from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps


def read_image_any(path: str | Path) -> np.ndarray:
    """Read PNG/JPG/TIFF-like image as RGB uint8 numpy array."""
    path = Path(path)
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(path)
    return path


def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (mask.astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(path)
    return path


def load_mask(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img) > 127


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    allowed = []
    for ch in stem:
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("_")
    out = "".join(allowed).strip("._")
    return out or "image"


def short_hash_bytes(data: bytes, n: int = 10) -> str:
    return hashlib.sha1(data).hexdigest()[:n]


def persist_uploaded_file(uploaded_file, output_dir: str | Path) -> tuple[Path, str]:
    """Save Streamlit uploaded file and return local path plus stable sample_id."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    sample_id = f"{safe_stem(uploaded_file.name)}_{short_hash_bytes(data)}"
    path = output_dir / f"{sample_id}{suffix}"
    path.write_bytes(data)
    return path, sample_id


def copy_to(path: str | Path, dst_dir: str | Path, new_name: str | None = None) -> Path:
    path = Path(path)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (new_name or path.name)
    shutil.copy2(path, dst)
    return dst


def save_class_mask(path: str | Path, class_mask: np.ndarray) -> Path:
    """Save uint8 semantic mask: 0=фон, 1=обычные, 2=тонкие, 3=тальк."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_mask.astype(np.uint8), mode="L").save(path)
    return path


def load_class_mask(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def save_float_map(path: str | Path, value_map: np.ndarray) -> Path:
    """Save float map 0..1 as an 8-bit PNG for reports/preview."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(value_map, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        out = np.zeros(arr.shape[:2], dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [1, 99])
        out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        out = (out * 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)
    return path
