from __future__ import annotations

import cv2
import numpy as np


ORE_CLASSES = ["рядовая руда", "труднообогатимая руда", "оталькованная руда"]
MASK_CLASS_NAMES = {
    0: "фон / матрица",
    1: "обычные срастания",
    2: "тонкие срастания",
    3: "тальк",
}


def estimate_valid_area(image_rgb: np.ndarray) -> np.ndarray:
    """
    Анализируемую область отделяем от явных полей/рамок.

    Для большинства OM-фото это почти вся картинка, но простая проверка по яркости
    убирает полностью черные поля и сильно снижает риск искажения процентов.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    if valid.sum() < image_rgb.shape[0] * image_rgb.shape[1] * 0.2:
        return np.ones(image_rgb.shape[:2], dtype=bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    return valid


def estimate_sulfide_masks(image_rgb: np.ndarray) -> dict:
    """
    Heuristic CV-сегментация сульфидов для обертки и fallback-режима.

    Возвращает две маски: обычные срастания и тонкие/замещенные срастания.
    Это не замена финальной ML-сегментации, но стабильный контракт для UI:
    class_mask всегда существует даже при отсутствии весов.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    valid = estimate_valid_area(image_rgb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    if valid.sum() < 100:
        h, w = image_rgb.shape[:2]
        empty = np.zeros((h, w), dtype=bool)
        return {
            "ordinary_mask": empty,
            "thin_mask": empty,
            "sulfide_mask": empty,
            "sulfide_area": 0,
            "ordinary_area": 0,
            "thin_area": 0,
            "sulfide_percent": 0.0,
            "ordinary_share_among_sulfides": 0.0,
            "thin_share_among_sulfides": 0.0,
        }

    try:
        otsu_thr = cv2.threshold(g[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        floor_thr = np.percentile(g[valid], 74)
        thr = max(float(otsu_thr), float(floor_thr))
    except Exception:
        thr = float(np.percentile(g[valid], 80))

    bright = ((g >= thr) & valid).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    valid_area = int(valid.sum())
    min_area = max(30, int(valid_area * 0.00001))
    large_threshold = max(300, int(valid_area * 0.00008))

    ordinary_mask = np.zeros_like(valid, dtype=bool)
    thin_mask = np.zeros_like(valid, dtype=bool)

    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area:
            continue

        component = labels == i
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = float(area) / hull_area
        elongation = max(w, h) / max(1, min(w, h))

        # Тонкие/замещенные срастания эвристически выглядят мельче,
        # вытянутее или менее цельными.
        is_thin = (area < large_threshold) or (solidity < 0.70) or (elongation > 4.0)
        if is_thin:
            thin_mask[component] = True
        else:
            ordinary_mask[component] = True

    sulfide_mask = ordinary_mask | thin_mask
    ordinary_area = int(ordinary_mask.sum())
    thin_area = int(thin_mask.sum())
    sulfide_area = ordinary_area + thin_area
    ordinary_share = ordinary_area / max(sulfide_area, 1) * 100.0 if sulfide_area else 0.0
    thin_share = thin_area / max(sulfide_area, 1) * 100.0 if sulfide_area else 0.0

    return {
        "ordinary_mask": ordinary_mask,
        "thin_mask": thin_mask,
        "sulfide_mask": sulfide_mask,
        "sulfide_area": sulfide_area,
        "ordinary_area": ordinary_area,
        "thin_area": thin_area,
        "sulfide_percent": sulfide_area / max(valid_area, 1) * 100.0,
        "ordinary_share_among_sulfides": ordinary_share,
        "thin_share_among_sulfides": thin_share,
    }


def estimate_sulfide_features(image_rgb: np.ndarray) -> dict:
    masks = estimate_sulfide_masks(image_rgb)
    return {k: v for k, v in masks.items() if not isinstance(v, np.ndarray)}


def make_class_mask(ordinary_mask: np.ndarray, thin_mask: np.ndarray, talc_mask: np.ndarray) -> np.ndarray:
    """0=фон, 1=обычные срастания, 2=тонкие срастания, 3=тальк."""
    class_mask = np.zeros(ordinary_mask.shape[:2], dtype=np.uint8)
    class_mask[ordinary_mask.astype(bool)] = 1
    class_mask[thin_mask.astype(bool)] = 2
    # Тальк важнее сульфидных эвристик: если пересеклось, показываем как тальк.
    class_mask[talc_mask.astype(bool)] = 3
    return class_mask


def class_mask_metrics(class_mask: np.ndarray, valid_mask: np.ndarray) -> dict:
    valid_area = int(valid_mask.sum())
    ordinary_area = int(((class_mask == 1) & valid_mask).sum())
    thin_area = int(((class_mask == 2) & valid_mask).sum())
    talc_area = int(((class_mask == 3) & valid_mask).sum())
    sulfide_area = ordinary_area + thin_area
    return {
        "valid_area_pixels": valid_area,
        "ordinary_area": ordinary_area,
        "thin_area": thin_area,
        "sulfide_area": sulfide_area,
        "talc_pixels": talc_area,
        "ordinary_percent": ordinary_area / max(valid_area, 1) * 100.0,
        "thin_percent": thin_area / max(valid_area, 1) * 100.0,
        "sulfide_percent": sulfide_area / max(valid_area, 1) * 100.0,
        "talc_percent": talc_area / max(valid_area, 1) * 100.0,
        "ordinary_share_among_sulfides": ordinary_area / max(sulfide_area, 1) * 100.0 if sulfide_area else 0.0,
        "thin_share_among_sulfides": thin_area / max(sulfide_area, 1) * 100.0 if sulfide_area else 0.0,
    }


def classify_ore(
    talc_percent: float,
    sulfide_features: dict,
    talc_threshold: float = 10.0,
) -> tuple[str, str]:
    if talc_percent > talc_threshold:
        ore_class = "оталькованная руда"
        conclusion = (
            f"Руда классифицирована как оталькованная: доля талька "
            f"составляет {talc_percent:.2f}% от анализируемой площади, что выше порога {talc_threshold:.1f}%."
        )
        return ore_class, conclusion

    thin = float(sulfide_features.get("thin_share_among_sulfides", 0.0))
    ordinary = float(sulfide_features.get("ordinary_share_among_sulfides", 0.0))

    if thin > ordinary:
        ore_class = "труднообогатимая руда"
        reason = (
            f"доля талька ниже порога ({talc_percent:.2f}%), "
            f"а признаки тонких срастаний преобладают ({thin:.2f}% против {ordinary:.2f}%)."
        )
    else:
        ore_class = "рядовая руда"
        reason = (
            f"доля талька ниже порога ({talc_percent:.2f}%), "
            f"а признаки обычных срастаний преобладают ({ordinary:.2f}% против {thin:.2f}%)."
        )

    return ore_class, f"Руда классифицирована как {ore_class}: {reason}"


def review_reason(
    talc_percent: float,
    mean_confidence: float | None = None,
    talc_threshold: float = 10.0,
    thin_share: float | None = None,
    ordinary_share: float | None = None,
) -> tuple[bool, str]:
    reasons = []
    if abs(talc_percent - talc_threshold) <= 2.0:
        reasons.append(f"доля талька близко к порогу {talc_threshold:.1f}%")
    if mean_confidence is not None and mean_confidence < 0.75:
        reasons.append("низкая средняя уверенность модели")
    if thin_share is not None and ordinary_share is not None and abs(float(thin_share) - float(ordinary_share)) <= 7.0:
        reasons.append("обычные и тонкие срастания близки по доле")
    if not reasons:
        return False, "автоматическая проверка не требуется"
    return True, "; ".join(reasons)
