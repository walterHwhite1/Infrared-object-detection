# Infrared Object Detection

A standalone Streamlit demo adapted from [Object-Detection-In-IR-](https://github.com/walterHwhite1/Object-Detection-In-IR-/tree/main), commit `e28ec38`. It keeps the original grayscale and CLAHE preprocessing and uses the COCO-pretrained YOLOv8n checkpoint already referenced by the original infrared video script. YOLOv8n is used for both images and videos to keep the comparison consistent and run on CPU.

## Run locally in VS Code

1. Install Python 3.12, extract the ZIP, and open the extracted `ir-object-detection` folder in VS Code. Open **Terminal → New Terminal**.
2. On Windows PowerShell run:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

   On macOS or Linux run:

   ```bash
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python -m streamlit run app.py
   ```

   On Debian or Ubuntu, install `libgl1` and the distribution's GLib runtime if importing OpenCV reports missing shared libraries. The repository's `packages.txt` is for Streamlit Community Cloud.
3. Open [http://localhost:8501](http://localhost:8501). Choose **Sample image** and click **Run detection**. Then upload a real IR image or short MP4 clip.
4. The included checkpoint allows models to load locally after the dependencies are installed. If you remove it, Ultralytics attempts to download the public YOLOv8n checkpoint on the first run.

Use **Compare both** to send identical source frames through the original input and the grayscale + CLAHE input at the same confidence threshold. Both modes draw boxes over the original image or frame. Images produce downloadable annotated PNG files. Video mode produces browsable annotated frames and a CSV covering all sampled frames. Uploads are capped at 40 MB. Video runs are bounded to the opening 30 seconds when FPS metadata is available, with additional frame and sample caps. Each CSV row represents one detection in one frame and mode. Repeated detections across frames are not counts of unique objects. For variable-frame-rate clips, timestamps estimated from nominal FPS are approximate.

## Deploy on Streamlit Community Cloud

1. Push this extracted project's source to a GitHub repository you can access. Commit `app.py`, `detector.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`, and `samples/bus.jpg`. The checkpoint is ignored by default, so the hosted app downloads it on its first run. If the deployment cannot fetch it, you may force-add the trusted checkpoint after confirming the applicable redistribution terms.
2. Sign in at [share.streamlit.io](https://share.streamlit.io). Select **Create app** and pick the repository, branch, and `app.py` as the entry point. Choose Python 3.12 in the advanced settings, then deploy.
3. Open the generated URL and run the sample and a short IR clip. Check the app logs if the dependency install or checkpoint download fails.

[Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Prepare for the defense

**Suggested title** · Evaluating CLAHE Preprocessing for YOLOv8 Object Detection on Infrared Footage

- Start with the research question. Does grayscale + CLAHE improve detection of the selected classes on the same infrared footage? A gain, a loss, and a mixed result are all meaningful if measured.
- Build a labeled test set from several short clips captured under different conditions. Report the sensor type, clip source, and target classes. Active near-infrared night vision and thermal imaging are distinct sensor modalities. Label them separately if you use both. The bundled visible-light photo is for setup only.
- Annotate ground-truth bounding boxes for one or two common classes, such as people and vehicles in a supervised campus or traffic recording. Evaluate both modes on identical images with the same model and input size. Report precision, recall, and AP at IoU 0.50 per class, plus mean AP if you have several classes. Report median per-image prediction latency after warm-up on the same hardware. The app shows the elapsed prediction call only, which excludes CLAHE and drawing and can include first-call setup.
- Label the current model accurately in the report. It uses COCO-pretrained weights without infrared fine-tuning. Remove the original “90% accuracy” heading unless you have a measured result with a defined metric and test set. A box score of 0.90 describes that detection's model confidence.
- Show one useful detection and one miss or false positive from the real test set. Discuss blur, distance, or contrast where relevant. For an extension, add a third grayscale-only condition to isolate the effect of CLAHE, then consider infrared fine-tuning with train and test footage split by source clip.

## Upstream assets

- [YOLOv8n checkpoint from Ultralytics Assets](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)
- [Visible-light smoke-test photo from Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg)
- [Ultralytics licensing information](https://www.ultralytics.com/license)
