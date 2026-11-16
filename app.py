"""Standalone Streamlit demo for image and sampled-video object detection."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import streamlit as st

from detector import BASELINE, ENHANCED, csv_bytes, detect, load_detector, video_frames

SAMPLE = Path(__file__).with_name("samples") / "bus.jpg"
MAX_PIXELS = 16_000_000

st.set_page_config(page_title="Infrared Object Detection", page_icon="◉", layout="wide")
theme = Path(__file__).with_name("theme.css").read_text(encoding="utf-8")
st.markdown(f"<style>{theme}</style>", unsafe_allow_html=True)
st.markdown(
    """
    <header class="ir-hero">
        <p class="ir-label">YOLOv8n / OpenCV</p>
        <h1>Infrared Object Detection</h1>
        <p>Run detection on an image or video clip and compare the original input with contrast enhancement.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

controls, workspace = st.columns([1, 2.25], gap="large")
with controls:
    with st.container(border=True):
        st.subheader("Input & settings")
        sources = ["Upload image", "Upload video"]
        if SAMPLE.is_file():
            sources.insert(0, "Sample image")
        source = st.radio("Input source", sources)
        upload = None
        stride = 15
        if source == "Upload image":
            upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        elif source == "Upload video":
            upload = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])
            stride = st.slider("Analyze every Nth frame", 5, 60, 15, 5)
            st.caption("Up to 60 samples from the first 30 seconds, with a 1,800-frame decode cap.")
        else:
            st.caption("Visible-light sample for checking the demo.")
        selection = st.radio("Preprocessing", ["Compare both", BASELINE, ENHANCED])
        confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)
        st.caption("Confidence is the model’s score for each detection.")
        run = st.button(
            "Run detection",
            type="primary",
            use_container_width=True,
            disabled=upload is None and source != "Sample image",
        )

modes = (BASELINE, ENHANCED) if selection == "Compare both" else (selection,)
config = (source, upload.file_id if upload else None, modes, confidence, stride)


@st.cache_resource
def model_resource():
    return load_detector()


def analyze_video(upload, model, progress):
    rows, previews = [], []
    with TemporaryDirectory() as directory:
        path = Path(directory) / ("input" + Path(upload.name).suffix.lower())
        path.write_bytes(upload.getbuffer())
        count = 0
        for index, seconds, frame in video_frames(path, stride):
            count += 1
            preview = []
            for mode in modes:
                annotation, detections, elapsed = detect(model, frame, mode, confidence, index, seconds)
                rows.extend(detections)
                if count <= 12:
                    height, width = annotation.shape[:2]
                    if width > 800:
                        annotation = cv2.resize(annotation, (800, max(1, round(height * 800 / width))))
                    success, encoded = cv2.imencode(".jpg", annotation)
                    if not success:
                        raise ValueError("Could not encode the preview frame.")
                    preview.append((mode, encoded.tobytes(), len(detections), elapsed))
            if preview:
                label = f"Frame {index}"
                if seconds is not None:
                    label += f" · {seconds:.2f} s"
                previews.append((label, preview))
            progress.caption(f"Analyzed {count} sampled frames…")
    if not count:
        raise ValueError("This video contains no readable frames.")
    return rows, previews, count


def analyze_image(upload, model):
    data = upload.getbuffer() if upload else SAMPLE.read_bytes()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode this image. Try a standard JPG or PNG.")
    if frame.shape[0] * frame.shape[1] > MAX_PIXELS:
        raise ValueError("Resize the image to 16 megapixels or less.")
    rows, preview = [], []
    for mode in modes:
        annotation, detections, elapsed = detect(model, frame, mode, confidence)
        rows.extend(detections)
        success, encoded = cv2.imencode(".png", annotation)
        if not success:
            raise ValueError("Could not encode the annotated image.")
        preview.append((mode, encoded.tobytes(), len(detections), elapsed))
    return rows, [("Image", preview)], 1


with workspace:
    st.subheader("Detection workspace")
    if run:
        st.session_state.pop("result", None)
        try:
            with st.spinner("Running detection. The first run may download YOLOv8n weights…"):
                model = model_resource()
                progress = st.empty()
                if source == "Upload video":
                    results = analyze_video(upload, model, progress)
                else:
                    results = analyze_image(upload, model)
                progress.empty()
            st.session_state.result = (config, results)
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error("Detection failed. Check that the dependencies and trusted YOLOv8n checkpoint are available.")
            with st.expander("Error details"):
                st.code(str(error))

    stored = st.session_state.get("result")
    if stored and stored[0] == config:
        rows, previews, frame_count = stored[1]
        st.markdown('<span class="ir-state">Analysis complete</span>', unsafe_allow_html=True)
        if source == "Upload video":
            st.caption(f"{frame_count} sampled frames analyzed · first {len(previews)} available for preview")
            chosen = st.selectbox("Preview frame", range(len(previews)), format_func=lambda i: previews[i][0])
        else:
            chosen = 0
        views = previews[chosen][1]
        for column, (mode, image, count, elapsed) in zip(st.columns(len(views)), views):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{mode}**")
                    st.image(image, use_container_width=True)
                    st.caption(f"{count} detections · {elapsed:.1f} ms prediction call")
                    if source != "Upload video":
                        name = "original" if mode == BASELINE else "clahe"
                        st.download_button(
                            "Download PNG", image, f"{name}.png", "image/png", key=name
                        )
        with st.expander("Detection data", expanded=bool(rows)):
            if rows:
                st.dataframe(
                    rows,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No detections met the confidence threshold.")
            st.download_button("Download detections CSV", csv_bytes(rows), "detections.csv", "text/csv")
    else:
        message = "Choose an input and select Run detection to see bounding boxes and confidence scores."
        if upload is None and source != "Sample image":
            message = "Choose a file in Input & settings to start."
        st.markdown(
            f'<div class="ir-empty"><strong>Ready for analysis</strong><span>{message}</span></div>',
            unsafe_allow_html=True,
        )
