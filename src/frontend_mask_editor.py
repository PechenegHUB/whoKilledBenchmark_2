from __future__ import annotations

import base64
import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import streamlit.components.v1 as components
from PIL import Image

_COMPONENT_DIR = Path(__file__).resolve().parent / "frontend_mask_editor_component"
_COMPONENT_DIR.mkdir(parents=True, exist_ok=True)

_component = components.declare_component("frontend_mask_editor", path=str(_COMPONENT_DIR))


def _pil_to_data_url(image: Image.Image | np.ndarray | None) -> str | None:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        arr = image.astype(np.uint8)
        if arr.ndim == 2:
            image = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            image = Image.fromarray(arr[:, :, :3], mode="RGB")
    else:
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def frontend_mask_editor(
    background_image,
    *,
    original_image=None,
    expert_background_image=None,
    model_class_background_image=None,
    expert_class_background_image=None,
    has_expert_mask: bool = False,
    has_expert_class_mask: bool = False,
    initial_base_choice: str = "model",
    review_context: dict[str, Any] | None = None,
    ore_classes: list[str] | None = None,
    key: str | None = None,
    storage_key: str | None = None,
    height: int = 980,
):
    """Browser-side multiclass mask editor.

    Returns a dict with:
      - edit_layer: PNG data URL of transparent stroke layer
      - base_choice: 'model' or 'expert'
      - action: 'apply' or 'save'

    The editor is intentionally single-mode: it edits semantic class_mask 0/1/2/3.
    """
    review_context = dict(review_context or {})
    ore_classes = ore_classes or ["рядовая руда", "труднообогатимая руда", "оталькованная руда"]

    args = {
        "originalImage": _pil_to_data_url(original_image),
        "modelTalcImage": _pil_to_data_url(background_image),
        "expertTalcImage": _pil_to_data_url(expert_background_image),
        "modelClassImage": _pil_to_data_url(model_class_background_image) or _pil_to_data_url(background_image),
        "expertClassImage": _pil_to_data_url(expert_class_background_image),
        "hasExpertMask": bool(has_expert_mask),
        "hasExpertClassMask": bool(has_expert_class_mask),
        "initialBaseChoice": initial_base_choice,
        "reviewContext": review_context,
        "oreClasses": ore_classes,
        "storageKey": storage_key or key or f"mask_editor_{uuid.uuid4().hex}",
    }
    return _component(args=args, default=None, key=key, height=height)


def decode_editor_image(editor_value: dict[str, Any] | None) -> np.ndarray | None:
    """Decode edit_layer data URL returned by frontend_mask_editor into RGBA ndarray."""
    if not editor_value:
        return None
    data_url = editor_value.get("edit_layer") or editor_value.get("image")
    if not data_url or not isinstance(data_url, str):
        return None
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        img = Image.open(BytesIO(raw)).convert("RGBA")
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None
