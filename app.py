from __future__ import annotations

import io
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from yolov7_utils import LoadedModel, YoloV7MissingError, load_model, run_inference


st.set_page_config(page_title="YOLOv7 Viewer", layout="wide")

WEIGHTS_DIR = Path("weights")
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"


@st.cache_resource(show_spinner=False)
def _cached_model(weights_path: str, device: str, imgsz: int) -> LoadedModel:
    return load_model(Path(weights_path), device=device, imgsz=imgsz)


def ensure_model(weights_path: Path, device: str, imgsz: int) -> LoadedModel:
    if not weights_path.is_file():
        st.error(
            f"找不到模型權重檔 `{weights_path}`。\n"
            "請確認路徑正確，或將訓練完成的 `.pt` 檔案放到該位置後再試一次。"
        )
        st.stop()

    try:
        return _cached_model(str(weights_path), device, imgsz)
    except FileNotFoundError:
        st.error(
            f"讀取權重檔 `{weights_path}` 失敗。\n"
            "請確認 YOLOv7 的 `.pt` 模型位於專案中設定的路徑。"
        )
        st.stop()
    except YoloV7MissingError as exc:
        st.error(str(exc))
        st.info(
            "範例：`git clone https://github.com/WongKinYiu/yolov7.git yolov7`\n"
            "請確認 `yolov7` 資料夾位在專案根目錄。"
        )
        st.stop()


def _discover_available_weights(weights_dir: Path) -> list[Path]:
    if not weights_dir.exists():
        return []
    return sorted(
        (
            weight_path
            for weight_path in weights_dir.iterdir()
            if weight_path.suffix.lower() == ".pt" and weight_path.is_file()
        ),
        key=lambda path: path.name.lower(),
    )


def _render_image_inference(
    model: LoadedModel,
    conf_thres: float,
    iou_thres: float,
    per_class_conf: dict[str, float] | None = None,
) -> None:
    uploaded = st.file_uploader(
        "上傳圖片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )
    if not uploaded:
        st.info("請先選擇一張圖片。")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"無法讀取圖片：{exc}")
        return

    st.subheader("原始圖片")
    st.image(image, use_container_width=True)

    with st.spinner("推論中，請稍候..."):
        detections, annotated = run_inference(
            model,
            image=image,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            per_class_confidence=per_class_conf,
        )

    st.subheader("標註結果")
    st.image(annotated, use_container_width=True)

    if detections:
        records = [
            {
                "類別": det.label,
                "信心值": f"{det.confidence:.3f}",
                "框座標 (x1, y1, x2, y2)": det.bbox,
            }
            for det in detections
        ]
        st.dataframe(pd.DataFrame(records), use_container_width=True)

        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG")
        st.download_button(
            "下載標註圖片",
            data=buffer.getvalue(),
            file_name="annotated.png",
            mime="image/png",
        )
    else:
        st.warning("未偵測到任何物件，請調整參數後再試一次。")


def _render_video_inference(
    model: LoadedModel,
    conf_thres: float,
    iou_thres: float,
    per_class_conf: dict[str, float] | None = None,
) -> None:
    video_file = st.file_uploader(
        "上傳影片",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video_uploader",
        help="影片會逐格推論並輸出含標註結果的 MP4 檔案。",
    )
    if not video_file:
        st.info("請先選擇一段影片。")
        return

    suffix = Path(video_file.name).suffix or ".mp4"
    detections_summary: Counter[str] = Counter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
        tmp_input.write(video_file.getbuffer())
        input_path = Path(tmp_input.name)

    output_path: Path | None = None
    cap = cv2.VideoCapture(str(input_path))
    writer: cv2.VideoWriter | None = None

    try:
        if not cap.isOpened():
            st.error("無法開啟影片檔案，請確認格式是否受支援。")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            st.error("無法取得影片尺寸，可能檔案已損壞。")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output:
            output_path = Path(tmp_output.name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            st.error("建立輸出影片失敗，請確認系統支援 MP4 編碼。")
            return

        progress = st.progress(0.0) if total_frames else None
        frame_idx = 0

        with st.spinner("影片推論中，請稍候..."):
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_idx += 1

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                detections, annotated = run_inference(
                    model,
                    image=image,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    per_class_confidence=per_class_conf,
                )

                for det in detections:
                    detections_summary[det.label] += 1

                annotated_frame = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
                writer.write(annotated_frame)

                if progress is not None:
                    progress.progress(min(frame_idx / total_frames, 1.0))

        if progress is not None:
            progress.progress(1.0)

        # 確保釋放資源並完成影片寫入
        cap.release()
        cap = None
        if writer is not None:
            writer.release()
        writer = None

        if output_path is None or not output_path.exists():
            st.error("產生標註影片時發生問題。")
            return

        video_bytes = output_path.read_bytes()
        st.subheader("標註影片")
        st.video(video_bytes)
        st.download_button(
            "下載標註影片",
            data=video_bytes,
            file_name="annotated_video.mp4",
            mime="video/mp4",
        )

        if detections_summary:
            summary_df = pd.DataFrame(
                [
                    {"類別": label, "偵測次數": count}
                    for label, count in sorted(
                        detections_summary.items(), key=lambda item: item[1], reverse=True
                    )
                ]
            )
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.warning("影片中未偵測到任何物件，請調整參數後再試一次。")
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        input_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def main() -> None:
    st.title("YOLOv7 影像偵測檢視器")
    st.caption("上傳圖片或影片即可檢視訓練完成的 YOLOv7 模型輸出結果。")

    per_class_conf: dict[str, float] = {}

    with st.sidebar:
        st.header("權重設定")
        available_weights = _discover_available_weights(WEIGHTS_DIR)
        if not available_weights:
            st.error("請將 YOLOv7 `.pt` 權重檔放到專案的 `weights` 資料夾後再載入")
            st.stop()

        default_index = next(
            (idx for idx, path in enumerate(available_weights) if path == DEFAULT_WEIGHTS),
            0,
        )
        weights_path = st.selectbox(
            "選擇權重檔 (.pt)",
            options=available_weights,
            index=min(default_index, len(available_weights) - 1),
            format_func=lambda path: path.name,
            help="將訓練好的 `.pt` 權重檔放在 `weights/` 資料夾即可在這裡選擇。",
        )
        device = st.selectbox("裝置", options=["cpu", "0"], index=0, help="沒有 GPU 時可選擇 `cpu`。")
        imgsz = st.slider("輸入尺寸 (pix)", min_value=256, max_value=1280, value=640, step=64)
        conf_thres = st.slider("信心閾值", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        iou_thres = st.slider("IoU 閾值", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
        model = ensure_model(weights_path, device=device, imgsz=imgsz)
        class_names = [str(name) for name in model.class_names]
        overrides_state: dict[str, float] = st.session_state.setdefault(
            "class_conf_overrides", {}
        )

        if class_names:
            with st.expander("\u4f9d\u985e\u5225\u81ea\u8a02\u4fe1\u5fc3\u95be\u503c", expanded=bool(overrides_state)):
                st.caption("\u672a\u8a2d\u5b9a\u7684\u985e\u5225\u6703\u6cbf\u7528\u4e0a\u65b9\u7684\u5168\u57df\u4fe1\u5fc3\u95be\u503c\u3002")
                default_selection = [label for label in overrides_state if label in class_names]
                selected_labels = st.multiselect(
                    "\u9078\u64c7\u8981\u8abf\u6574\u7684\u985e\u5225",
                    options=class_names,
                    default=default_selection,
                )

                new_overrides: dict[str, float] = {}
                for label in selected_labels:
                    slider_key = f"class_conf_slider_{label}"
                    default_value = float(overrides_state.get(label, conf_thres))
                    new_overrides[label] = st.slider(
                        label,
                        min_value=0.05,
                        max_value=0.95,
                        value=default_value,
                        step=0.05,
                        key=slider_key,
                    )

                removed_labels = set(overrides_state) - set(new_overrides)
                for removed in removed_labels:
                    st.session_state.pop(f"class_conf_slider_{removed}", None)

                st.session_state["class_conf_overrides"] = new_overrides
                per_class_conf = new_overrides

    image_tab, video_tab = st.tabs(["圖片偵測", "影片偵測"])
    per_class_conf_arg = per_class_conf or None
    inference_conf = (
        min([conf_thres] + list(per_class_conf.values())) if per_class_conf else conf_thres
    )

    with image_tab:
        _render_image_inference(
            model,
            conf_thres=inference_conf,
            iou_thres=iou_thres,
            per_class_conf=per_class_conf_arg,
        )

    with video_tab:
        _render_video_inference(
            model,
            conf_thres=inference_conf,
            iou_thres=iou_thres,
            per_class_conf=per_class_conf_arg,
        )


if __name__ == "__main__":
    main()
