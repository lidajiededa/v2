from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import numpy.typing as npt
from typing import Any, Optional
import tempfile
import os
import warnings
import cv2
from PIL import Image

from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, OpenCVVideoBackend


def get_extracted_frame_indices(total_frames: int, original_fps: float, target_fps: float):
    """
    align with train extract image frames from video 
    """
    if total_frames <= 0:
        return []

    frame_step = original_fps / target_fps
    frame_indices = []
    idx = 0
    count_float = 0.0

    for raw_frame_idx in range(total_frames):
        count = round(count_float)
        idx += 1
        if idx <= count:
            continue
        frame_indices.append(raw_frame_idx)
        count_float += frame_step
    return frame_indices


class NPUOpenCVDynamicVideoBackend(OpenCVVideoBackend):
    @staticmethod
    def decode_single_frame(frame_idx: int, video_path: str, total_frames_num: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                return frame_rgb
            else:
                next_idx = frame_idx + 1
                while next_idx < total_frames_num:
                    ret_next, next_frame = cap.read()
                    if ret_next:
                        return cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
                    next_idx += 1
                return None
        finally:
            cap.release()

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = 32,
        sample_fps: int = 1,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        # logger.info("Using NPUOpenCVDynamicVideoBackend")
        import cv2
        temp_path = None
        backend = cls().get_cv2_video_api()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(data)
            temp_path = f.name
        cap = cv2.VideoCapture(temp_path) 
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of video frames
        original_fps = float(cap.get(cv2.CAP_PROP_FPS)) # Video fps 
        # The timestamp of the rightmost frame, cannot be used to calculate frame 0.
        total_duration = (total_frames_num - 1) / original_fps

        # `sample_fps` is the FPS parameter passed in for sampling,
        # -1 indicates that sampling can be performed directly without FPS limitation.
        if sample_fps > 0:
            extracted_frame_indices = get_extracted_frame_indices(
                total_frames=total_frames_num,
                original_fps=original_fps,
                target_fps=sample_fps,
            )
            extracted_total_frames = len(extracted_frame_indices)
            # align with train
            total_duration = (extracted_total_frames - 1) / sample_fps
            # Num_frames is the maximum number of frames to sample. 
            # If fewer frames are sampled at this sample_fps, the update duration will be longer.
            if num_frames >= int(total_duration * sample_fps) + 1:
                num_frames = int(total_duration * sample_fps) + 1
                # Under the new maximum frame rate, the video duration of the rightmost frame,
                # cannot be calculated for frame 0.
                total_duration = min(total_duration, (num_frames - 1) / sample_fps) 
            sample_frame_timestamps = np.linspace(0, total_duration, num_frames, dtype=float)
            sampled_seq_indices = [
                min(extracted_total_frames - 1, round(t * sample_fps))
                for t in sample_frame_timestamps
            ]
            # align with train:get the train offline extract frame index
            frames_indices = [extracted_frame_indices[idx] for idx in sampled_seq_indices]
        elif sample_fps == -1:
            # train sample_fps > 0,so keep the old logic 
            sample_frame_timestamps = np.linspace(0, total_duration, num_frames, dtype=float)
            frames_indices = [
                min(total_frames_num - 1, round(t * original_fps))
                for t in sample_frame_timestamps
            ]
        elif sample_fps != -1:
            raise ValueError(f"requires dataset fps is -1 or greater than 0 but got {sample_fps}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frames_indices), height, width, 3), dtype=np.uint8)
        valid_count = 0
        task = [(frame_idx, temp_path, total_frames_num) for frame_idx in frames_indices]
        decode_frame_thread_count = kwargs.pop("decode_frame_thread_count", 4)
        with ThreadPoolExecutor(max_workers=decode_frame_thread_count) as executor:
            results = list(executor.map(lambda args: cls.decode_single_frame(*args), task))
            for frame_data in results:
                if frame_data is not None:
                    frames[valid_count] = frame_data
                    valid_count += 1

        if valid_count != len(frames_indices):
            warnings.warn(
                f"Expected reading {len(frames_indices)} frames, "
                f"but only loaded {valid_count} frames from video.",
                UserWarning,
                stacklevel=2
            )
        if temp_path:
            os.remove(temp_path)

        # Use transformers transformers.video_utils.VideoMetadata format.
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": total_duration,
            "video_backend": "opencv_dynamic",
            "frames_indices": frames_indices,
            "do_sample_frames": False,
            "sample_frame_timestamps": sample_frame_timestamps
        }
        return frames, metadata


# registerbackend
VIDEO_LOADER_REGISTRY.register("npu_opencv_dynamic")(NPUOpenCVDynamicVideoBackend)
