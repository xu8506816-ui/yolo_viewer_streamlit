"""
Utility helpers for running YOLOv7 inference from a local checkout of the YOLOv7
repository.

This module expects a copy of the official YOLOv7 project to be available so the
model classes referenced inside the weight files can be imported. By default we
look for a `yolov7` directory next to this file, but the location can be
overridden with the `YOLOV7_REPO` environment variable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

try:
    import cv2  # noqa: F401  # Needed by YOLOv7 submodules
except ImportError as exc:  # pragma: no cover - dependencies validated at runtime
    raise ImportError(
        "OpenCV (cv2) is required because the YOLOv7 repository depends on it.\n"
        "Please install with `pip install opencv-python-headless`."
    ) from exc

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Detection:
    """Single detection output from YOLOv7."""

    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class YoloV7MissingError(RuntimeError):
    """Raised when we cannot locate a YOLOv7 source checkout."""


_FONT = ImageFont.load_default()


def _resolve_repo_path() -> Path:
    """Find the YOLOv7 repository path."""

    env_override = os.getenv("YOLOV7_REPO")
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if not candidate.exists():
            raise YoloV7MissingError(
                f"Configured YOLOv7 repository '{candidate}' does not exist."
            )
        return candidate

    default_path = Path(__file__).parent / "yolov7"
    if default_path.exists():
        return default_path.resolve()

    raise YoloV7MissingError(
        "Could not find a YOLOv7 repository. Please clone "
        "`https://github.com/WongKinYiu/yolov7` into "
        f"`{default_path}` or set the environment variable YOLOV7_REPO "
        "to point at your local checkout."
    )


def _import_yolov7_dependencies() -> None:
    """Ensure YOLOv7 modules are importable."""

    repo_path = _resolve_repo_path()
    if str(repo_path) not in sys.path:
        sys.path.append(str(repo_path))


def _load_dependencies():
    """Import YOLOv7 utilities after the repository path is configured."""

    from models.experimental import attempt_load  # type: ignore
    from utils.general import check_img_size, non_max_suppression, scale_coords  # type: ignore
    from utils.torch_utils import select_device  # type: ignore

    return attempt_load, check_img_size, non_max_suppression, scale_coords, select_device


@dataclass
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    stride: int
    class_names: Sequence[str]
    imgsz: int


def load_model(weights_path: Path, device: str = "cpu", imgsz: int = 640) -> LoadedModel:
    """
    Load a YOLOv7 model from weights.

    Parameters
    ----------
    weights_path:
        Path to the `.pt` checkpoint (e.g. the `best.pt` file from training).
    device:
        Device identifier understood by YOLOv7's `select_device`. Use `"cpu"`
        for CPU inference or `"0"` for the first CUDA device.
    imgsz:
        Preferred inference image size. This is validated against the model
        stride so the final value may be adjusted.
    """

    _import_yolov7_dependencies()
    attempt_load, check_img_size, _, _, select_device = _load_dependencies()

    device_obj = select_device(device)

    try:
        from torch.serialization import add_safe_globals
    except ImportError:
        add_safe_globals = None  # type: ignore[assignment]
    if add_safe_globals is not None:
        safe_items = []
        try:
            reconstruct = getattr(np.core.multiarray, "_reconstruct")
        except AttributeError:
            reconstruct = None
        if reconstruct is not None:
            safe_items.append(reconstruct)
        safe_items.append(np.ndarray)
        safe_items.append(np.dtype)
        add_safe_globals(safe_items)
    weights = str(weights_path)
    model = attempt_load(weights, map_location=device_obj)
    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    imgsz_validated = check_img_size(imgsz, s=stride)
    class_names = model.names if hasattr(model, "names") else getattr(model, "module", model).names

    model.to(device_obj)
    model.eval()

    return LoadedModel(
        model=model,
        device=device_obj,
        stride=stride,
        class_names=class_names,
        imgsz=imgsz_validated,
    )


def _preprocess_image(
    image: Image.Image,
    imgsz: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Prepare an image for YOLOv7 inference.

    Returns the original RGB array, the normalized tensor input, and the ratio/padding
    values required to map predictions back to the original size.
    """

    img0 = np.array(image.convert("RGB"))
    img, ratio, pad = _letterbox(img0, new_shape=imgsz, stride=stride)
    img = img.transpose(2, 0, 1)  # channel-first
    img = np.ascontiguousarray(img)
    return img0, img, ratio, pad


def run_inference(
    loaded: LoadedModel,
    image: Image.Image,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> Tuple[List[Detection], Image.Image]:
    """
    Execute a forward pass on a PIL image and return detections with an annotated copy.
    """

    attempt_load, check_img_size, non_max_suppression, scale_coords, _ = _load_dependencies()
    del attempt_load, check_img_size  # silence linters for unused imports

    img0, img, ratio, pad = _preprocess_image(image, loaded.imgsz, loaded.stride)

    im = torch.from_numpy(img).to(loaded.device)
    im = im.float()  # uint8 to fp32
    im /= 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    with torch.no_grad():
        pred = loaded.model(im)[0]

    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

    detections: List[Detection] = []
    annotated = image.copy()

    if len(pred) and pred[0] is not None and len(pred[0]):
        det = pred[0]
        det[:, :4] = scale_coords(
            im.shape[2:], det[:, :4], img0.shape, ratio_pad=(ratio, pad)
        ).round()

        for *xyxy, conf, cls in det.tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            class_idx = int(cls)
            label = loaded.class_names[class_idx] if class_idx < len(loaded.class_names) else str(class_idx)
            detections.append(Detection(label=label, confidence=float(conf), bbox=(x1, y1, x2, y2)))
            _draw_box(annotated, (x1, y1, x2, y2), label, float(conf))
    return detections, annotated


def _draw_box(
    image_rgb: Image.Image,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    """Render a labelled bounding box onto an image in-place."""

    x1, y1, x2, y2 = bbox
    color = (255, 128, 0)
    outline_width = 2

    draw = ImageDraw.Draw(image_rgb)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=outline_width)

    label_text = f"{label} {confidence:.2f}"
    font = _FONT

    try:
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
    except AttributeError:
        tw, th = draw.textsize(label_text, font=font)
        text_bbox = (x1, y1, x1 + tw, y1 + th)

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = x1
    text_y = y1 - text_height - 4
    if text_y < 0:
        text_y = y1 + 4

    draw.rectangle(
        (text_x, text_y, text_x + text_width, text_y + text_height),
        fill=color,
    )
    draw.text((text_x, text_y), label_text, fill="white", font=font)


def _letterbox(
    image: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    stride: int = 32,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize and pad image to fit the desired shape while preserving aspect ratio.

    This re-implements YOLOv7's letterbox logic using Pillow instead of OpenCV.
    """

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    shape = image.shape[:2]  # current (h, w)
    if shape[0] == 0 or shape[1] == 0:
        raise ValueError("Invalid image with zero dimension.")

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw = dw % stride
    dh = dh % stride
    dw /= 2
    dh /= 2

    resized = Image.fromarray(image).resize(new_unpad, Image.BILINEAR)
    resized_np = np.array(resized)

    top = int(np.floor(dh))
    bottom = int(np.ceil(dh))
    left = int(np.floor(dw))
    right = int(np.ceil(dw))

    height = new_shape[0]
    width = new_shape[1]
    canvas = np.full((height, width, resized_np.shape[2]), color, dtype=resized_np.dtype)
    canvas[top : top + resized_np.shape[0], left : left + resized_np.shape[1]] = resized_np

    return np.ascontiguousarray(canvas), (r, r), (dw, dh)


__all__ = ["Detection", "LoadedModel", "YoloV7MissingError", "load_model", "run_inference"]
