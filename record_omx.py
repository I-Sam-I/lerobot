#!/usr/bin/env python
"""Record and inspect the OMX camera-based imitation-learning dataset."""

import argparse
import sys
from pathlib import Path

from lerobot.cameras import Cv2Backends, make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig


TASK = "Pick up the gray cube and place it on top of the white cube."
REPO_ID = "local/omx_pick_place_v1"
DATASET_ROOT = Path(__file__).resolve().parent / "data" / "omx_pick_place_v1"
CAMERA_FEATURES = (
    "observation.images.overhead",
    "observation.images.wrist",
)
REQUIRED_FEATURES = {*CAMERA_FEATURES, "observation.state", "action"}


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
        port="/dev/ttyACM1",
        id="omx_follower_arm",
        cameras=cameras,
    )
    leader = OmxLeaderConfig(
        port="/dev/ttyACM0",
        id="omx_leader_arm",
    )
    dataset = DatasetRecordConfig(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        single_task=TASK,
        fps=30,
        num_episodes=3,
        episode_time_s=30,
        reset_time_s=20,
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
        play_sounds=False,
        resume=resume,
    )


def check_cameras() -> None:
    """Open both configured cameras and read a frame without connecting the robot."""
    cameras = make_cameras_from_configs(build_camera_configs())
    try:
        for name, camera in cameras.items():
            camera.connect()
            frame = camera.async_read(timeout_ms=5000)
            print(f"{name}: captured frame with shape {frame.shape}")
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
    record(build_config(resume=resume))


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
    commands.add_parser(
        "check-cameras",
        help="test both configured camera streams without connecting the robot",
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
        check_cameras()
    else:
        _record(resume=args.resume)


if __name__ == "__main__":
    main()
