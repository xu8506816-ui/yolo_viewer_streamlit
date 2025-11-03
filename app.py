from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from yolov7_utils import LoadedModel, YoloV7MissingError, load_model, run_inference


st.set_page_config(page_title="YOLOv7 Viewer", layout="wide")

DEFAULT_WEIGHTS = Path("exp_custom_ciou_sgd") / "weights" / "best.pt"


@st.cache_resource(show_spinner=False)
def _cached_model(weights_path: str, device: str, imgsz: int) -> LoadedModel:
    return load_model(Path(weights_path), device=device, imgsz=imgsz)


def ensure_model(weights_path: Path, device: str, imgsz: int) -> LoadedModel:
    if not weights_path.is_file():
        st.error(
            f"找不到權重檔案：`{weights_path}`。\n"
            "請確認權重路徑輸入正確，或將訓練好的 `.pt` 檔案放到該位置。"
        )
        st.stop()

    try:
        return _cached_model(str(weights_path), device, imgsz)
    except FileNotFoundError:
        st.error(
            f"找不到權重檔案：`{weights_path}`。\n"
            "請確認 YOLOv7 訓練好的 `.pt` 檔案位在專案底下對應路徑。"
        )
        st.stop()
    except YoloV7MissingError as exc:
        st.error(str(exc))
        st.info(
            "範例：`git clone https://github.com/WongKinYiu/yolov7.git yolov7`，"
            "並確保資料夾位在此專案根目錄。"
        )
        st.stop()


def main() -> None:
    st.title("YOLOv7 影像偵測檢視器")
    st.caption("上傳照片後即時套用訓練完成的 YOLOv7 模型進行標註。")

    with st.sidebar:
        st.header("推論設定")
        weights_path = Path(
            st.text_input(
                "權重檔案路徑 (.pt)",
                value=str(DEFAULT_WEIGHTS),
                help="預設使用 `exp_custom_ciou_sgd/weights/best.pt`。",
            )
        ).expanduser()

        device = st.selectbox("裝置", options=["cpu", "0"], index=0, help="沒有 GPU 時選擇 `cpu`。")
        imgsz = st.slider("推論輸入尺寸 (pix)", min_value=256, max_value=1280, value=640, step=64)
        conf_thres = st.slider("信心閾值", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        iou_thres = st.slider("IoU 閾值", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

    uploaded = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png", "bmp", "webp"])
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

    model = ensure_model(weights_path, device=device, imgsz=imgsz)

    with st.spinner("推論中，請稍候..."):
        detections, annotated = run_inference(
            model, image=image, conf_thres=conf_thres, iou_thres=iou_thres
        )

    st.subheader("標註結果")
    st.image(annotated, use_container_width=True)

    if detections:
        records = [
            {
                "分類": det.label,
                "信心": f"{det.confidence:.3f}",
                "邊界框 (x1, y1, x2, y2)": det.bbox,
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
        st.warning("未偵測到任何物件，請調整參數或更換圖片試試。")


if __name__ == "__main__":
    main()
