from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# Цвета по ТЗ: зеленый = обычные, красный = тонкие, синий = тальк.
ORDINARY_GREEN_RGB = (0, 210, 90)
THIN_RED_RGB = (255, 50, 50)
TALC_BLUE_RGB = (0, 96, 255)
TALC_BLUE_HEX = "#0060FF"
MASK_CLASS_COLORS = {
    1: ORDINARY_GREEN_RGB,
    2: THIN_RED_RGB,
    3: TALC_BLUE_RGB,
}

# Native colors of the new superpixel segmentation model:
# 0=unmarked, 1=talc, 2=dense_sulfide, 3=thin_sulfide,
# 4=magnetite_filler, 5=background_silicates, 6=uncertain.
SUPERPIXEL_CLASS_COLORS = {
    1: (0, 90, 255),
    2: (0, 200, 90),
    3: (255, 60, 40),
    4: (255, 190, 0),
    5: (180, 180, 180),
    6: (210, 0, 255),
}


def mask_to_pil(mask: np.ndarray, color=TALC_BLUE_RGB) -> Image.Image:
    h, w = mask.shape[:2]
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[mask.astype(bool)] = color
    return Image.fromarray(arr, mode="RGB")


def class_mask_to_rgb(class_mask: np.ndarray) -> np.ndarray:
    h, w = class_mask.shape[:2]
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in MASK_CLASS_COLORS.items():
        arr[class_mask == class_id] = color
    return arr




def superpixel_mask_to_rgb(native_mask: np.ndarray) -> np.ndarray:
    h, w = native_mask.shape[:2]
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in SUPERPIXEL_CLASS_COLORS.items():
        arr[native_mask == class_id] = color
    return arr


def make_superpixel_overlay_np(image_rgb: np.ndarray, native_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image_rgb.astype(np.float32).copy()
    colors = superpixel_mask_to_rgb(native_mask).astype(np.float32)
    active = native_mask.astype(np.uint8) > 0
    out = image.copy()
    out[active] = image[active] * (1.0 - alpha) + colors[active] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_superpixel_overlay_pil(image_rgb: np.ndarray, native_mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    return Image.fromarray(make_superpixel_overlay_np(image_rgb, native_mask, alpha=alpha), mode="RGB")


def make_overlay_np(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45, color=TALC_BLUE_RGB) -> np.ndarray:
    image = image_rgb.astype(np.float32).copy()
    overlay = image.copy()
    overlay[mask.astype(bool)] = np.array(color, dtype=np.float32)
    out = image * (1.0 - alpha) + overlay * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_overlay_pil(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45, color=TALC_BLUE_RGB) -> Image.Image:
    return Image.fromarray(make_overlay_np(image_rgb, mask, alpha=alpha, color=color), mode="RGB")


def make_class_overlay_np(image_rgb: np.ndarray, class_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image_rgb.astype(np.float32).copy()
    colors = class_mask_to_rgb(class_mask).astype(np.float32)
    active = class_mask.astype(np.uint8) > 0
    out = image.copy()
    out[active] = image[active] * (1.0 - alpha) + colors[active] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_class_overlay_pil(image_rgb: np.ndarray, class_mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    return Image.fromarray(make_class_overlay_np(image_rgb, class_mask, alpha=alpha), mode="RGB")


def make_heatmap_pil(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay для карт уверенности/Grad-CAM/talc-score."""
    if heatmap is None:
        return Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim == 3:
        hm = hm[..., 0]
    finite = hm[np.isfinite(hm)]
    if finite.size == 0:
        hm01 = np.zeros(hm.shape, dtype=np.float32)
    else:
        lo, hi = np.percentile(finite, [1, 99])
        hm01 = np.clip((hm - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    if hm01.shape[:2] != image_rgb.shape[:2]:
        hm01 = cv2.resize(hm01, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap((hm01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image_rgb.astype(np.float32)
    out = image * (1.0 - alpha) + colored * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def make_diff_overlay_pil(image_rgb: np.ndarray, model_mask: np.ndarray, expert_mask: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """
    Синий — совпадение модели и эксперта.
    Зелёный — эксперт добавил.
    Красный — эксперт удалил.
    """
    image = image_rgb.astype(np.float32).copy()
    color = image.copy()
    model = model_mask.astype(bool)
    expert = expert_mask.astype(bool)
    both = model & expert
    added = expert & ~model
    removed = model & ~expert
    color[both] = np.array(TALC_BLUE_RGB, dtype=np.float32)
    color[added] = np.array([0, 210, 90], dtype=np.float32)
    color[removed] = np.array([255, 40, 40], dtype=np.float32)
    out = image * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def resize_for_display(image_rgb: np.ndarray, mask: np.ndarray | None = None, max_side: int = 1100):
    h, w = image_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_pil = Image.fromarray(image_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
    if mask is None:
        return np.asarray(img_pil), None, scale
    mask_arr = np.asarray(mask)
    if mask_arr.dtype == bool:
        mask_pil = Image.fromarray(mask_arr.astype(np.uint8) * 255).resize((new_w, new_h), Image.Resampling.NEAREST)
        return np.asarray(img_pil), np.asarray(mask_pil) > 127, scale
    mask_pil = Image.fromarray(mask_arr.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.NEAREST)
    return np.asarray(img_pil), np.asarray(mask_pil).astype(mask_arr.dtype), scale
