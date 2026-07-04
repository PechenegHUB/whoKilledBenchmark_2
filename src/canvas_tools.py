from __future__ import annotations

import cv2
import numpy as np

# RGB colors used by the browser editor.
COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # фон / ластик
    (0, 210, 90): 1,     # обычные / плотные сульфиды
    (255, 50, 50): 2,    # тонкие / замещённые
    (0, 96, 255): 3,     # тальк
}


def _resize_canvas(canvas_image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize RGBA/RGB canvas layer to mask size with nearest interpolation."""
    target_h, target_w = target_shape
    arr = np.asarray(canvas_image)
    if arr.shape[:2] == (target_h, target_w):
        return arr
    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _nearest_class_id(rgb: np.ndarray) -> np.ndarray:
    """Map drawn RGB pixels to semantic class ids by nearest palette color."""
    rgb_f = rgb.astype(np.float32)
    palette = np.array(list(COLOR_TO_CLASS.keys()), dtype=np.float32)
    class_ids = np.array(list(COLOR_TO_CLASS.values()), dtype=np.uint8)
    # (H,W,1,3) - (1,1,K,3) -> (H,W,K)
    dist = ((rgb_f[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
    return class_ids[dist.argmin(axis=-1)]


def apply_canvas_edits(
    base_mask: np.ndarray,
    canvas_image: np.ndarray,
    original_shape: tuple[int, int],
    *,
    edit_mode: str = "talc",
) -> np.ndarray:
    """
    Apply browser editor strokes to either a binary talc mask or a multiclass class_mask.

    Parameters
    ----------
    base_mask:
        bool mask for edit_mode='talc' or uint8 class mask for edit_mode='multiclass'.
    canvas_image:
        RGBA/RGB image returned from the browser editor. Transparent pixels mean untouched.
    original_shape:
        Target output shape (H, W).
    edit_mode:
        'talc' for backward-compatible binary editing, 'multiclass' for 0/1/2/3 class editing.
    """
    h, w = original_shape
    base = np.asarray(base_mask)
    if base.shape[:2] != (h, w):
        interp = cv2.INTER_NEAREST
        base = cv2.resize(base.astype(np.uint8), (w, h), interpolation=interp)
        if base_mask.dtype == bool:
            base = base > 0
        else:
            base = base.astype(np.uint8)
    else:
        base = base.copy()

    if canvas_image is None:
        return base

    layer = _resize_canvas(np.asarray(canvas_image), (h, w))
    if layer.ndim == 2:
        rgb = np.stack([layer, layer, layer], axis=-1).astype(np.uint8)
        alpha = layer > 0
    else:
        rgb = layer[:, :, :3].astype(np.uint8)
        if layer.shape[2] >= 4:
            alpha = layer[:, :, 3] > 10
        else:
            alpha = rgb.sum(axis=-1) > 10

    if not np.any(alpha):
        return base

    mode = str(edit_mode or "talc").lower()
    if mode in {"class", "classes", "multiclass", "semantic"}:
        out = base.astype(np.uint8).copy()
        painted_classes = _nearest_class_id(rgb)
        out[alpha] = painted_classes[alpha]
        return out

    # Backward-compatible talc binary mode.
    out = base.astype(bool).copy()
    painted_classes = _nearest_class_id(rgb)
    # Blue/class 3 means add talc; black/class 0 means erase talc.
    out[alpha & (painted_classes == 3)] = True
    out[alpha & (painted_classes == 0)] = False
    return out
