#!/usr/bin/env python
"""Record and inspect the OMX camera-based imitation-learning dataset."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lerobot.cameras import Cv2Backends, make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
import lerobot.scripts.lerobot_record as recording_module
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.utils.keyboard_input import apply_recording_control, init_keyboard_listener
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


TASK = "Pick up the gray cube and place it on top of the white cube."
REPO_ID = "local/omx_pick_place_v1"
DATASET_ROOT = Path(__file__).resolve().parent / "data" / "omx_pick_place_v1"
LEADER_PORT = "/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_2E2212C3503059384C2E3120FF052222-if00"
FOLLOWER_PORT = "/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_CA7CDE56503059384C2E3120FF081424-if00"
CAMERA_FEATURES = (
    "observation.images.overhead",
    "observation.images.wrist",
)
REQUIRED_FEATURES = {*CAMERA_FEATURES, "observation.state", "action"}
CONTROL_PANEL_HOST = "127.0.0.1"
CONTROL_PANEL_PORT = 8766
CONTROL_PANEL_URL = f"http://localhost:{CONTROL_PANEL_PORT}"

_CONTROL_PANEL_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeRobot Recording Controls</title>
  <style>
    body { font: 18px system-ui, sans-serif; max-width: 720px; margin: 3rem auto; padding: 0 1rem; }
    h1 { font-size: 1.6rem; }
    .buttons { display: grid; gap: 1rem; margin: 2rem 0; }
    button { border: 0; border-radius: 10px; padding: 1.2rem; color: white; font-size: 1.2rem; font-weight: 700; }
    button:disabled { opacity: .55; }
    #next { background: #16794b; }
    #retry { background: #a65c00; }
    #quit { background: #ad2634; }
    #status { min-height: 1.5em; }
  </style>
</head>
<body>
  <h1>LeRobot recording controls</h1>
  <p>These controls work only while recording is active.</p>
  <div class="buttons">
    <button id="next">Finish and save this episode (n)</button>
    <button id="retry">Discard this episode and retry (r)</button>
    <button id="quit">Stop recording (q)</button>
  </div>
  <p><strong>Stop recording saves the current partial episode.</strong>
     To discard it, choose &quot;Discard this episode and retry&quot; before stopping.</p>
  <p id="status" role="status" aria-live="polite">Checking recording status...</p>
  <script>
    const status = document.getElementById("status");
    const buttons = [...document.querySelectorAll("button")];
    async function refreshStatus() {
      try {
        const response = await fetch("/status", {cache: "no-store"});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || response.statusText);
        status.textContent = `${result.recording ? "RECORDING" : "NOT RECORDING"} — ${result.detail}`;
        status.dataset.state = result.state;
      } catch (error) {
        status.textContent = `Status unavailable: ${error.message}`;
        status.dataset.state = "unavailable";
      }
    }
    async function send(control, confirmation) {
      if (confirmation && !window.confirm(confirmation)) return;
      buttons.forEach(button => button.disabled = true);
      status.textContent = "Sending control...";
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
      }
    }
    refreshStatus();
    window.setInterval(refreshStatus, 1000);
    document.getElementById("next").onclick =
      () => send("next");
    document.getElementById("retry").onclick =
      () => send("retry", "Discard the current episode and record it again?");
    document.getElementById("quit").onclick =
      () => send("quit", "Stop recording? The current partial episode will be saved.");
  </script>
</body>
</html>""".encode("utf-8")


class _RecordingControlPanel:
    """Local browser controls that set LeRobot's existing recording event flags."""

    def __init__(self) -> None:
        self.events: dict[str, Any] | None = None
        self._control_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._state = "starting"
        self._detail = "Connecting to recording devices."
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if urlparse(self.path).path == "/status":
                    status = panel.get_status()
                    body = json.dumps(status).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if urlparse(self.path).path != "/":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_CONTROL_PANEL_HTML)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(_CONTROL_PANEL_HTML)

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/control":
                    self._respond(404, {"error": "Unknown control endpoint."})
                    return
                if self.headers.get("Origin") not in (
                    None,
                    CONTROL_PANEL_URL,
                    f"http://127.0.0.1:{CONTROL_PANEL_PORT}",
                ):
                    self._respond(403, {"error": "Requests must come from this local control panel."})
                    return
                if self.headers.get_content_type() != "application/json":
                    self._respond(415, {"error": "Expected an application/json request."})
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < content_length <= 128:
                        raise ValueError("Invalid request size.")
                    payload = json.loads(self.rfile.read(content_length))
                except (ValueError, json.JSONDecodeError) as error:
                    self._respond(400, {"error": f"Invalid control request: {error}"})
                    return

                control = payload.get("control") if isinstance(payload, dict) else None
                if not isinstance(control, str):
                    self._respond(400, {"error": "Control must be next, retry, or quit."})
                    return
                controls = {
                    "next": ("right", "Finish and save the current episode."),
                    "retry": ("left", "Discard the current episode and retry it."),
                    "quit": ("esc", "Stop recording; the current partial episode is saved."),
                }
                if control not in controls:
                    self._respond(400, {"error": "Control must be next, retry, or quit."})
                    return
                if panel.events is None:
                    self._respond(409, {"error": "Recording controls are not active yet."})
                    return

                key, message = controls[control]
                with panel._control_lock:
                    if panel.events["exit_early"]:
                        self._respond(409, {"error": "A recording control is already pending."})
                        return
                    apply_recording_control(key, panel.events)
                self._respond(200, {"message": message})

            def _respond(self, status: int, payload: dict[str, str]) -> None:
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

        self._server = LocalControlServer((CONTROL_PANEL_HOST, CONTROL_PANEL_PORT), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()
        print(f"Recording control buttons: {CONTROL_PANEL_URL}")

    def set_state(self, state: str, detail: str) -> None:
        with self._status_lock:
            self._state = state
            self._detail = detail

    def get_status(self) -> dict[str, Any]:
        with self._status_lock:
            return {
                "state": self._state,
                "recording": self._state == "recording",
                "session_active": self._state != "stopped",
                "detail": self._detail,
            }

    def attach_keyboard_listener(
        self, listener: Any, events: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        self.events = events

        class CombinedListener:
            def stop(self) -> None:
                if listener is not None:
                    listener.stop()

        return CombinedListener(), events

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def build_camera_configs() -> dict[str, OpenCVCameraConfig]:
    """Build the verified camera settings, pinned to the tested Linux V4L2 backend."""
    return {
        "overhead": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
            warmup_s=5,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
            warmup_s=5,
        ),
    }


def build_config(*, resume: bool = False) -> RecordConfig:
    """Build the exact recording configuration without opening any devices."""
    cameras = build_camera_configs()
    follower = OmxFollowerConfig(
        port=FOLLOWER_PORT,
        id="omx_follower_arm",
        cameras=cameras,
    )
    leader = OmxLeaderConfig(
        port=LEADER_PORT,
        id="omx_leader_arm",
    )
    dataset = DatasetRecordConfig(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        single_task=TASK,
        fps=30,
        num_episodes=3,
        episode_time_s=30,
        reset_time_s=15,
        streaming_encoding=True,
        encoder_threads=2,
        push_to_hub=False,
        no_stamp=True,
    )
    return RecordConfig(
        robot=follower,
        teleop=leader,
        dataset=dataset,
        display_data=True,
        display_mode="rerun",
        play_sounds=False,
        resume=resume,
    )


def check_cameras(*, display: bool = False, display_mode: str = "rerun") -> None:
    """Open both configured cameras without connecting the robot; optionally stream to a viewer."""
    cameras = make_cameras_from_configs(build_camera_configs())
    visualization_started = False
    try:
        for name, camera in cameras.items():
            camera.connect()
            frame = camera.async_read(timeout_ms=5000)
            print(f"{name}: captured frame with shape {frame.shape}")
        if not display:
            return

        init_visualization(display_mode, session_name="omx_camera_check")
        visualization_started = True
        if display_mode == "foxglove":
            print("Live camera feeds are streaming; connect Foxglove to ws://127.0.0.1:8765.")
        else:
            print("Live camera feeds are streaming to Rerun. Press Ctrl+C to stop.")
        try:
            while True:
                frames = {
                    name: camera.async_read(timeout_ms=5000)
                    for name, camera in cameras.items()
                }
                log_visualization_data(display_mode, observation=frames)
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopped live camera preview.")
    finally:
        try:
            if visualization_started:
                shutdown_visualization(display_mode)
        finally:
            for camera in cameras.values():
                if camera.is_connected:
                    camera.disconnect()


def _load_local_metadata() -> LeRobotDatasetMetadata:
    info_path = DATASET_ROOT / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata not found at {info_path}. "
            "Record a new dataset first; resume requires an existing LeRobot dataset."
        )
    return LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)


def _validate_resume() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Cannot resume: dataset directory does not exist: {DATASET_ROOT}"
        )
    metadata = _load_local_metadata()
    existing_tasks = set(metadata.tasks.index) if metadata.tasks is not None else set()
    missing_features = REQUIRED_FEATURES - set(metadata.features)
    if missing_features:
        raise ValueError(
            "Refusing to resume because the dataset schema is missing required "
            f"feature(s): {sorted(missing_features)!r}."
        )
    if metadata.robot_type != "omx_follower":
        raise ValueError(
            "Refusing to resume because the dataset was not recorded with "
            f"omx_follower (found {metadata.robot_type!r})."
        )

    if metadata.total_episodes == 0:
        if metadata.total_frames != 0 or existing_tasks:
            raise ValueError(
                "Refusing to resume inconsistent empty-dataset metadata: "
                f"{metadata.total_frames} frames and tasks {sorted(existing_tasks)!r}."
            )
        print(
            "Resuming the compatible empty dataset left by the failed startup; "
            "it contains no recorded demonstrations."
        )
        return

    if existing_tasks != {TASK}:
        raise ValueError(
            "Refusing to resume because the existing dataset task metadata does not "
            f"match {TASK!r}. Existing tasks: {sorted(existing_tasks)!r}. "
            "Do not relabel old demonstrations; choose a new dataset ID/root instead."
        )
    if metadata.total_frames < 1:
        raise ValueError(
            "Refusing to resume dataset metadata that reports completed episodes "
            "but contains no frames."
        )


def verify_dataset() -> None:
    """Validate saved task, feature schema, per-episode videos, and a decoded frame."""
    metadata = _load_local_metadata()
    if metadata.total_episodes < 1:
        raise ValueError(f"No completed episodes found in {DATASET_ROOT}.")

    existing_tasks = set(metadata.tasks.index) if metadata.tasks is not None else set()
    if existing_tasks != {TASK}:
        raise ValueError(
            "Dataset task metadata must contain only the requested task. "
            f"Found: {sorted(existing_tasks)!r}"
        )

    features = set(metadata.features)
    missing = REQUIRED_FEATURES - features
    if missing:
        raise ValueError(f"Dataset is missing required feature(s): {sorted(missing)!r}")

    missing_videos = []
    for episode_index in range(metadata.total_episodes):
        for camera_feature in CAMERA_FEATURES:
            video_path = DATASET_ROOT / metadata.get_video_file_path(episode_index, camera_feature)
            if not video_path.is_file() or video_path.stat().st_size == 0:
                missing_videos.append((episode_index, str(video_path)))
    if missing_videos:
        raise FileNotFoundError(f"Missing or empty camera video(s): {missing_videos!r}")

    dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, download_videos=False)
    for episode_index in range(metadata.total_episodes):
        frame_index = metadata.episodes["dataset_from_index"][episode_index]
        sample = dataset[int(frame_index)]
        empty_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in sample or sample[feature] is None or sample[feature].numel() == 0
        ]
        if empty_features:
            raise ValueError(
                f"Episode {episode_index} has missing or empty feature(s): {empty_features!r}"
            )

    print(f"Verified {metadata.total_episodes} completed episode(s) at {DATASET_ROOT}")
    print(f"Task: {TASK}")
    print("Verified schema: both camera streams, observation.state, and action")
    print("Verified non-empty video files for both cameras in every episode.")
    print("Decoded a frame from each episode containing all required features.")


def _confirm_recording() -> None:
    print(
        "WARNING: recording connects the leader and follower, enables robot control, "
        "and sends leader-derived actions to the follower. Clear the workspace and "
        "keep an emergency stop available."
    )
    if not sys.stdin.isatty():
        raise RuntimeError("Recording requires an interactive terminal confirmation.")
    if input("Type RECORD to continue: ").strip() != "RECORD":
        raise SystemExit("Recording cancelled; no robot was connected.")


def _record(*, resume: bool) -> None:
    if resume:
        _validate_resume()
    elif DATASET_ROOT.exists():
        raise FileExistsError(
            f"Refusing to use existing path {DATASET_ROOT}. "
            "Use --resume only if its task metadata matches, or select a new dataset path."
        )

    _confirm_recording()
    control_panel = _RecordingControlPanel()
    original_listener_factory = recording_module.init_keyboard_listener
    original_record_loop = recording_module.record_loop
    control_panel.start()
    recording_module.init_keyboard_listener = lambda: control_panel.attach_keyboard_listener(
        *original_listener_factory()
    )

    def report_recording_state(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("dataset") is None:
            control_panel.set_state("resetting", "Resetting between episodes.")
        else:
            control_panel.set_state("recording", "Teleoperating and saving the current episode.")
        try:
            return original_record_loop(*args, **kwargs)
        finally:
            control_panel.set_state("finishing", "Finishing the current recording phase.")

    recording_module.record_loop = report_recording_state
    try:
        print("Camera and robot telemetry will be shown in the Rerun viewer.")
        print("Browser buttons and the usual keyboard shortcuts both control recording.")
        record(build_config(resume=resume))
    finally:
        control_panel.set_state("stopped", "Recording has stopped.")
        recording_module.init_keyboard_listener = original_listener_factory
        recording_module.record_loop = original_record_loop
        control_panel.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check-config",
        help="validate LeRobot imports and configuration without connecting hardware",
    )
    commands.add_parser(
        "verify-dataset",
        help="check saved episodes and decode a frame without connecting hardware",
    )
    camera_parser = commands.add_parser(
        "check-cameras",
        help="test both configured camera streams without connecting the robot",
    )
    camera_parser.add_argument(
        "--display",
        action="store_true",
        help="stream live camera feeds to a viewer until Ctrl+C",
    )
    camera_parser.add_argument(
        "--display-mode",
        choices=("rerun", "foxglove"),
        default="rerun",
        help="viewer backend (defaults to Rerun; Foxglove requires its desktop app)",
    )
    record_parser = commands.add_parser("record", help="record teleoperated demonstrations")
    record_parser.add_argument(
        "--resume",
        action="store_true",
        help="explicitly append after checking the task/schema (also recovers a compatible empty dataset)",
    )
    args = parser.parse_args()

    if args.command == "check-config":
        config = build_config()
        print("LeRobot imports and configuration are valid; no hardware was opened.")
        print(f"Task: {config.dataset.single_task}")
        print(f"Dataset root: {config.dataset.root}")
        print("Cameras: V4L2, MJPG 640x480 at 30 FPS; 5-second warmup per camera")
    elif args.command == "verify-dataset":
        verify_dataset()
    elif args.command == "check-cameras":
        check_cameras(display=args.display, display_mode=args.display_mode)
    else:
        _record(resume=args.resume)


if __name__ == "__main__":
    main()
