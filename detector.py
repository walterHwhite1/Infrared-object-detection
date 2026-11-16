"""Preprocessing and prediction shared by the image and video demos."""

import csv
import io
import math
from pathlib import Path
from time import perf_counter

import cv2

BASELINE = "Original"
ENHANCED = "Grayscale + CLAHE"
CHECKPOINT = Path(__file__).with_name("yolov8n.pt")
REPORT_COLUMNS = (
    "mode", "frame", "time_seconds", "class", "confidence", "x1", "y1", "x2", "y2",
)


def load_detector():
    from ultralytics import YOLO

    return YOLO(str(CHECKPOINT) if CHECKPOINT.is_file() else CHECKPOINT.name)


def prepare_frame(frame, mode):
    if mode == BASELINE:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def detect(model, frame, mode, confidence, frame_index=0, seconds=None):
    model_input = prepare_frame(frame, mode)
    start = perf_counter()
    result = model.predict(
        model_input, conf=confidence, imgsz=640, device="cpu", verbose=False
    )[0]
    prediction_ms = round((perf_counter() - start) * 1000, 1)
    rows = []
    for box in result.boxes:
        coordinates = box.xyxy[0].tolist()
        rows.append({
            "mode": mode,
            "frame": frame_index,
            "time_seconds": seconds,
            "class": result.names[int(box.cls.item())],
            "confidence": round(box.conf.item(), 4),
            **dict(zip(("x1", "y1", "x2", "y2"), map(lambda x: round(x, 1), coordinates))),
        })
    return result.plot(img=frame.copy()), rows, prediction_ms


def video_frames(path, stride, duration_seconds=30, max_samples=60):
    """Yield samples from a bounded opening segment, using source frame indices."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Could not open this video. Try an H.264 MP4.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        has_fps = math.isfinite(fps) and fps > 0
        frame_limit = min(1800, math.ceil(fps * duration_seconds)) if has_fps else 900
        for index in range(frame_limit):
            if not capture.grab():
                break
            if index % stride:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                break
            yield index, round(index / fps, 3) if has_fps else None, frame
            if index // stride + 1 >= max_samples:
                break
    finally:
        capture.release()


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")
