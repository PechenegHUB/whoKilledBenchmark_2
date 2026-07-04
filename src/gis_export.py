from __future__ import annotations

import csv
import json
import math
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Common class mask used by the wrapper. This is intentionally small and aligned with the TЗ colors.
GIS_CLASS_INFO: dict[int, dict[str, Any]] = {
    1: {"name": "ordinary_sulfide", "name_ru": "обычные / плотные сульфиды", "color": "#34c759"},
    2: {"name": "thin_sulfide", "name_ru": "тонкие / замещённые сульфиды", "color": "#ff3b30"},
    3: {"name": "talc", "name_ru": "тальк", "color": "#0a84ff"},
}


def parse_microns_per_pixel(row: dict[str, Any]) -> float | None:
    """Extract microns-per-pixel scale from row metadata if it exists."""
    candidates: list[Any] = []
    for key in ("microns_per_pixel", "scale_microns_per_pixel", "scale_um_per_px"):
        if row.get(key) not in (None, ""):
            candidates.append(row.get(key))

    metadata_raw = row.get("metadata_json") or ""
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
            for key in ("microns_per_pixel", "scale_microns_per_pixel", "scale_um_per_px"):
                if metadata.get(key) not in (None, ""):
                    candidates.append(metadata.get(key))
        except Exception:
            pass

    for value in candidates:
        try:
            text = str(value).strip().replace(",", ".")
            if not text:
                continue
            parsed = float(text)
            if math.isfinite(parsed) and parsed > 0:
                return parsed
        except Exception:
            continue
    return None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _safe_feature_id(sample_id: str, class_id: int, idx: int) -> str:
    base = str(sample_id or "sample").replace(" ", "_")
    return f"{base}_class{class_id}_{idx:04d}"


def _connected_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from skimage.measure import label

        labels = label(binary.astype(bool), connectivity=1).astype(np.int32)
        return labels, int(labels.max())
    except Exception:
        # Tiny fallback without skimage.measure.label. Slow, but safe for small masks.
        h, w = binary.shape
        labels = np.zeros((h, w), dtype=np.int32)
        current = 0
        for y in range(h):
            for x in range(w):
                if not binary[y, x] or labels[y, x] != 0:
                    continue
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
        return labels, current


def _mask_component_to_bbox(component: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(component)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _contours_for_component(component: np.ndarray, simplify_tolerance: float) -> list[np.ndarray]:
    try:
        from skimage.measure import approximate_polygon, find_contours

        contours = find_contours(component.astype(np.float32), level=0.5)
        out: list[np.ndarray] = []
        for contour in contours:
            if simplify_tolerance > 0:
                contour = approximate_polygon(contour, tolerance=simplify_tolerance)
            if contour.shape[0] >= 4:
                out.append(contour)
        return out
    except Exception:
        # Conservative fallback: bbox polygon around component.
        bbox = _mask_component_to_bbox(component)
        if bbox is None:
            return []
        y0, y1, x0, x1 = bbox
        return [np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0], [y0, x0]], dtype=np.float32)]


def _to_geo_coords(
    contour_rc: np.ndarray,
    offset_y: int,
    offset_x: int,
    microns_per_pixel: float | None,
) -> list[list[float]]:
    scale = float(microns_per_pixel) if microns_per_pixel else 1.0
    coords: list[list[float]] = []
    for row, col in contour_rc:
        x = (float(col) + offset_x) * scale
        # GeoJSON/Y-up viewers display this more naturally if image rows go downward as negative Y.
        y = -(float(row) + offset_y) * scale
        coords.append([round(x, 4), round(y, 4)])
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def build_mask_geojson(
    mask: np.ndarray,
    row: dict[str, Any] | None = None,
    source_mask: str = "class_mask",
    min_area_px: int = 25,
    simplify_tolerance_px: float = 1.2,
) -> dict[str, Any]:
    """Vectorize a common class mask into GeoJSON polygons in local image coordinates.

    Coordinates are local, not georeferenced: X is pixel or micrometer coordinate from the left edge;
    Y is negative pixel/micrometer coordinate from the top edge. This keeps the image visually upright in
    ordinary GIS viewers even without a real CRS.
    """
    row = row or {}
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"GIS export expects a 2D class mask, got shape={mask_arr.shape}")

    h, w = mask_arr.shape
    sample_id = str(row.get("sample_id") or "sample")
    original_name = str(row.get("original_name") or "")
    microns_per_pixel = parse_microns_per_pixel(row)
    units = "micrometers" if microns_per_pixel else "pixels"

    features: list[dict[str, Any]] = []
    total_area_px = max(1, int(h * w))

    for class_id, info in GIS_CLASS_INFO.items():
        binary = mask_arr == int(class_id)
        labels, n_labels = _connected_components(binary)
        for label_id in range(1, n_labels + 1):
            component = labels == label_id
            area_px = int(component.sum())
            if area_px < int(min_area_px):
                continue

            bbox = _mask_component_to_bbox(component)
            if bbox is None:
                continue
            y0, y1, x0, x1 = bbox
            # Pad the crop so contours can close cleanly at component borders.
            py0, py1 = max(0, y0 - 1), min(h, y1 + 1)
            px0, px1 = max(0, x0 - 1), min(w, x1 + 1)
            crop = component[py0:py1, px0:px1]
            contours = _contours_for_component(crop, simplify_tolerance=float(simplify_tolerance_px))

            for contour_idx, contour in enumerate(contours, start=1):
                coords = _to_geo_coords(contour, offset_y=py0, offset_x=px0, microns_per_pixel=microns_per_pixel)
                if len(coords) < 4:
                    continue
                feature_index = len(features) + 1
                area_um2 = area_px * microns_per_pixel * microns_per_pixel if microns_per_pixel else None
                features.append({
                    "type": "Feature",
                    "id": _safe_feature_id(sample_id, class_id, feature_index),
                    "properties": {
                        "sample_id": sample_id,
                        "original_name": original_name,
                        "source_mask": source_mask,
                        "class_id": int(class_id),
                        "class_name": info["name"],
                        "class_name_ru": info["name_ru"],
                        "color": info["color"],
                        "component_id": int(label_id),
                        "contour_id": int(contour_idx),
                        "area_px": area_px,
                        "area_percent": round(area_px / total_area_px * 100.0, 6),
                        "area_um2": round(area_um2, 6) if area_um2 is not None else None,
                        "bbox_px": [int(x0), int(y0), int(x1), int(y1)],
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                })

    return {
        "type": "FeatureCollection",
        "name": f"{sample_id}_{source_mask}",
        "features": features,
        "metadata": {
            "format": "GeoJSON",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_id": sample_id,
            "original_name": original_name,
            "source_mask": source_mask,
            "image_width_px": int(w),
            "image_height_px": int(h),
            "coordinate_system": "local_image_coordinates",
            "coordinate_units": units,
            "microns_per_pixel": microns_per_pixel,
            "x_axis": "left_to_right",
            "y_axis": "top_to_bottom_represented_as_negative_y",
            "note": "GeoJSON is exported in local image coordinates, not in a geographic CRS. Use microns_per_pixel metadata for real physical scale.",
        },
    }


def save_mask_geojson(
    mask: np.ndarray,
    row: dict[str, Any],
    out_path: str | Path,
    source_mask: str = "class_mask",
    min_area_px: int = 25,
    simplify_tolerance_px: float = 1.2,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geojson = build_mask_geojson(
        mask,
        row=row,
        source_mask=source_mask,
        min_area_px=min_area_px,
        simplify_tolerance_px=simplify_tolerance_px,
    )
    out_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_gis_readme(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """GIS export for ore thin-section analysis\n\n"
        "Формат: GeoJSON FeatureCollection.\n"
        "Координаты: локальная система изображения, не географическая CRS.\n"
        "X считается от левого края изображения. Y записан отрицательным значением от верхнего края, "
        "чтобы слой визуально не переворачивался в обычных GIS-просмотрщиках.\n\n"
        "Классы common class_mask:\n"
        "1 — ordinary_sulfide / обычные или плотные сульфиды / green\n"
        "2 — thin_sulfide / тонкие или замещённые сульфиды / red\n"
        "3 — talc / тальк / blue\n\n"
        "Если в метаданных партии указан масштаб мкм/px, координаты и area_um2 считаются в микрометрах. "
        "Иначе координаты остаются в пикселях.\n"
        """,
        encoding="utf-8",
    )
    return path


def write_errors_csv(path: str | Path, errors: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "original_name", "error"])
        writer.writeheader()
        for row in errors:
            writer.writerow(row)
    return path
