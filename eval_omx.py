#!/usr/bin/env python
"""Offline-check or perform a short, supervised OMX ACT policy rollout."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.feature_utils import build_dataset_frame

from record_omx import (
    DATASET_ROOT,
    FOLLOWER_PORT,
    REPO_ID,
    TASK,
    _load_local_metadata,
    build_camera_configs,
)


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "train"
    / "omx_act_smoke_2026-11-10"
    / "checkpoints"
    / "000100"
    / "pretrained_model"
)
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 8767
CONTROL_URL = f"http://localhost:{CONTROL_PORT}"
MAX_LIVE_SECONDS = 3.0
MAX_RELATIVE_TARGET = 1.0
CONTROL_FPS = 30

_CONTROL_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OMX Policy Test Controls</title>
  <style>
    body { font: 18px system-ui, sans-serif; max-width: 720px; margin: 3rem auto; padding: 0 1rem; }
    .warning { border: 3px solid #ad2634; border-radius: 10px; padding: 1rem; font-weight: 800; }
    .buttons { display: grid; gap: 1rem; margin: 2rem 0; }
    button { border: 0; border-radius: 10px; padding: 1.2rem; color: white; font-size: 1.2rem; font-weight: 700; }
    button:disabled { opacity: .55; }
    #start { background: #16794b; }
    #stop { background: #ad2634; }
    #status { min-height: 1.5em; }
  </style>
</head>
<body>
  <h1>OMX policy test controls</h1>
  <p class="warning">SOFTWARE STOP ONLY — not a safety-rated emergency stop. It cannot interrupt an in-progress USB call or help if the browser, computer, process, or USB link hangs. Keep the physical power cutoff within reach.</p>
  <p>Policy output is limited to a maximum 3-second run. Remain beside the robot and use the physical cutoff for unexpected motion.</p>
  <div class="buttons">
    <button id="start">Start the short policy test</button>
    <button id="stop">SOFTWARE STOP</button>
  </div>
  <p id="status" role="status" aria-live="polite">Checking status...</p>
  <script>
    const status = document.getElementById("status");
    const buttons = [...document.querySelectorAll("button")];
    async function refreshStatus() {
      try {
        const response = await fetch("/status", {cache: "no-store"});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || response.statusText);
        status.textContent = `${result.state.toUpperCase()} — ${result.detail}`;
        document.getElementById("start").disabled = !result.can_start;
        document.getElementById("stop").disabled = !result.can_stop;
      } catch (error) {
        status.textContent = `Status unavailable: ${error.message}`;
      }
    }
    async function send(control) {
      if (control === "start" &&
          !window.confirm("Start the short policy test? Keep your hand at the physical cutoff.")) return;
      buttons.forEach(button => button.disabled = true);
      status.textContent = "Sending request...";
      try {
        const response = await fetch("/control", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({control})
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || response.statusText);
        status.textContent = result.message;
      } catch (error) {
        status.textContent = `Control failed: ${error.message}`;
      } finally {
        buttons.forEach(button => button.disabled = false);
        refreshStatus();
      }
    }
    document.getElementById("start").onclick = () => send("start");
    document.getElementById("stop").onclick = () => send("stop");
    refreshStatus();
    window.setInterval(refreshStatus, 500);
  </script>
</body>
</html>""".encode("utf-8")


class _PolicyControlPanel:
    def __init__(self) -> None:
        self.start_requested = threading.Event()
        self.stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._state = "starting"
        self._detail = "Loading checkpoint."
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/status":
                    payload = panel.get_status()
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path != "/":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_CONTROL_HTML)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(_CONTROL_HTML)

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/control":
                    self.respond(404, {"error": "Unknown endpoint."})
                    return
                if self.headers.get("Origin") not in (
                    None,
                    CONTROL_URL,
                    f"http://127.0.0.1:{CONTROL_PORT}",
                ):
                    self.respond(403, {"error": "Request did not come from the local panel."})
                    return
                if self.headers.get_content_type() != "application/json":
                    self.respond(415, {"error": "Expected application/json."})
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 128:
                        raise ValueError("Invalid request size.")
                    payload = json.loads(self.rfile.read(size))
                except (ValueError, json.JSONDecodeError) as error:
                    self.respond(400, {"error": f"Invalid request: {error}"})
                    return

                control = payload.get("control") if isinstance(payload, dict) else None
                if control == "start":
                    with panel._lock:
                        if panel._state != "waiting":
                            self.respond(409, {"error": "Policy is not waiting to start."})
                            return
                        panel.start_requested.set()
                        panel._state = "running"
                        panel._detail = "Policy rollout requested."
                    self.respond(200, {"message": "Short policy rollout requested."})
                    return
                if control == "stop":
                    with panel._lock:
                        if panel._state not in ("waiting", "running"):
                            self.respond(409, {"error": "No active policy test to stop."})
                            return
                        panel.stop_requested.set()
                        panel._state = "stopping"
                        panel._detail = "Software stop requested; a command already in progress may finish."
                    self.respond(200, {"message": "Software stop requested."})
                    return
                self.respond(400, {"error": "Control must be start or stop."})

            def respond(self, status: int, payload: dict[str, str]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        class LocalControlServer(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = LocalControlServer((CONTROL_HOST, CONTROL_PORT), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()
        print(f"Policy test controls: {CONTROL_URL}")

    def set_state(self, state: str, detail: str) -> None:
        with self._lock:
            self._state = state
            self._detail = detail

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "detail": self._detail,
                "can_start": self._state == "waiting",
                "can_stop": self._state in ("waiting", "running", "stopping"),
            }

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _load_policy(checkpoint: Path, metadata: LeRobotDatasetMetadata):
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory not found: {checkpoint}")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = make_policy(cfg=config, ds_meta=metadata)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    policy.eval()
    return policy, preprocessor, postprocessor


def _prepare_observation(raw_observation: dict[str, Any], dataset_features: dict) -> dict[str, Any]:
    frame = build_dataset_frame(dataset_features, raw_observation, prefix="observation")
    return frame


def check_policy(checkpoint: Path) -> None:
    metadata = _load_local_metadata()
    dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, download_videos=False)
    policy, preprocessor, postprocessor = _load_policy(checkpoint, metadata)

    sample = dataset[0]
    raw_observation: dict[str, Any] = {}
    state_names = metadata.features["observation.state"]["names"]
    state_values = sample["observation.state"].detach().cpu().numpy()
    if len(state_names) != len(state_values):
        raise ValueError(
            f"Dataset state names/values differ in length: {len(state_names)} != {len(state_values)}."
        )
    raw_observation.update(
        {name: float(value) for name, value in zip(state_names, state_values, strict=True)}
    )
    for camera_key in ("observation.images.overhead", "observation.images.wrist"):
        image = sample[camera_key].detach().cpu()
        if image.ndim != 3:
            raise ValueError(f"Expected a single image for {camera_key}; got {tuple(image.shape)}.")
        if image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        image_array = image.numpy()
        if np.issubdtype(image_array.dtype, np.floating):
            image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
        raw_observation[camera_key.removeprefix("observation.images.")] = image_array

    model_observation = build_inference_frame(
        raw_observation,
        torch.device(policy.config.device),
        metadata.features,
        task=TASK,
        robot_type="omx_follower",
    )
    model_observation = preprocessor(model_observation)
    policy.reset()
    with torch.inference_mode():
        action = postprocessor(policy.select_action(model_observation))
    values = action.detach().to("cpu").numpy()
    if values.ndim != 2 or values.shape[0] != 1 or not np.isfinite(values).all():
        raise ValueError(f"Expected one finite action vector; got shape {values.shape}.")
    expected_dim = len(metadata.features["action"]["names"])
    if values.shape[1] != expected_dim:
        raise ValueError(f"Policy returned {values.shape[1]} actions; dataset expects {expected_dim}.")
    print(f"Offline policy check passed using {policy.config.device}.")
    print(f"Checkpoint: {checkpoint}")
    print(f"Input: both camera images and {len(state_names)} joint values.")
    print(f"Output: one finite {values.shape[1]}-joint action; no hardware was opened.")


def run_live(checkpoint: Path, duration: float) -> None:
    if not 0 < duration <= MAX_LIVE_SECONDS:
        raise ValueError(f"Live test duration must be in (0, {MAX_LIVE_SECONDS}] seconds.")
    metadata = _load_local_metadata()
    policy, preprocessor, postprocessor = _load_policy(checkpoint, metadata)
    action_names = metadata.features["action"]["names"]
    follower_config = OmxFollowerConfig(
        port=FOLLOWER_PORT,
        id="omx_follower_arm",
        cameras=build_camera_configs(),
        max_relative_target=MAX_RELATIVE_TARGET,
        disable_torque_on_disconnect=True,
    )
    robot = make_robot_from_config(follower_config)
    if list(robot.action_features) != action_names:
        raise ValueError(
            "Policy dataset action order does not match the OMX follower motor order: "
            f"{action_names!r} != {list(robot.action_features)!r}"
        )

    panel = _PolicyControlPanel()
    panel.start()
    connected = False
    try:
        print(
            "LIVE MODE: this connects to the follower and can move it. "
            "Use only after installing and testing the physical power cutoff."
        )
        print(
            f"Policy output is capped to {MAX_RELATIVE_TARGET} normalized units per joint per command "
            f"and {duration:g} seconds total."
        )
        if not __import__("sys").stdin.isatty():
            raise RuntimeError("Live policy testing requires an interactive terminal.")
        if input("Type READY after checking the physical cutoff, clear area, and start pose: ").strip() != "READY":
            raise SystemExit("Live test cancelled; no robot was connected.")

        robot.connect(calibrate=False)
        connected = True
        policy.reset()
        panel.set_state("waiting", "Robot connected. Verify the setup, then start from the browser.")
        print(f"Open {CONTROL_URL}; click Start only when ready. The test stops after {duration:g} seconds.")

        while not panel.start_requested.wait(timeout=0.1):
            if panel.stop_requested.is_set():
                panel.set_state("stopped", "Software stop requested before motion.")
                return

        start_time = time.perf_counter()
        period = 1.0 / CONTROL_FPS
        sent = 0
        while sent < int(duration * CONTROL_FPS):
            if panel.stop_requested.is_set():
                break
            tick_start = time.perf_counter()
            raw_observation = robot.get_observation()
            model_observation = build_inference_frame(
                raw_observation,
                torch.device(policy.config.device),
                metadata.features,
                task=TASK,
                robot_type=robot.name,
            )
            model_observation = preprocessor(model_observation)
            with torch.inference_mode():
                action = postprocessor(policy.select_action(model_observation))
            action_values = action.detach().to("cpu").numpy()
            if action_values.ndim != 2 or action_values.shape[0] != 1:
                raise ValueError(f"Unexpected policy action shape: {action_values.shape}.")
            if action_values.shape[1] != len(action_names):
                raise ValueError(
                    f"Policy returned {action_values.shape[1]} actions; follower expects {len(action_names)}."
                )
            if not np.isfinite(action_values).all():
                raise ValueError("Policy produced a non-finite action; refusing to send it.")
            if panel.stop_requested.is_set():
                break
            if time.perf_counter() - start_time >= duration:
                break
            robot.send_action(make_robot_action(action, metadata.features))
            sent += 1
            delay = period - (time.perf_counter() - tick_start)
            if delay > 0:
                time.sleep(delay)
            if time.perf_counter() - start_time >= duration:
                break

        panel.set_state(
            "stopped",
            f"Policy commands ended after {sent} step(s); robot disconnect will disable torque.",
        )
        print(f"Policy test ended after {sent} command(s).")
    finally:
        panel.set_state("stopped", "Policy test ended.")
        if connected and robot.is_connected:
            robot.disconnect()
        panel.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="run one policy inference on a saved frame")
    live_parser = commands.add_parser("run", help="run a short supervised policy test on the follower")
    for subparser in (check_parser, live_parser):
        subparser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    live_parser.add_argument("--duration", type=float, default=MAX_LIVE_SECONDS)
    args = parser.parse_args()

    if args.command == "check":
        check_policy(args.checkpoint)
    else:
        run_live(args.checkpoint, args.duration)


if __name__ == "__main__":
    main()
