from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from src.classification import (
    estimate_valid_area,
    estimate_sulfide_masks,
    classify_ore,
    review_reason,
    make_class_mask,
    class_mask_metrics,
)
from src.ml_adapter import (
    ModelBundle,
    load_model_bundle,
    predict_with_cnn_bundle,
    predict_with_segmentation_bundle,
    predict_with_superpixel_bundle,
)


def load_segmentation_model(model_path: str, device: str = "auto"):
    """
    Backward-compatible wrapper для app.py.

    Раньше эта функция грузила только сегментацию. Теперь она грузит единый ML-bundle:
    - CNN-классификатор из нового notebook-решения;
    - старую SMP/Unet-сегментацию;
    - CV-baseline, если весов нет или зависимости не установлены.
    """
    return load_model_bundle(model_path=model_path, device=device)


def cv_talc_zone_baseline(image_rgb: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV-baseline returns mask, pseudo-confidence map, talc_score."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if valid_mask is not None and valid_mask.any():
        dark_thr = np.percentile(gray_blur[valid_mask], 28)
    else:
        dark_thr = np.percentile(gray_blur, 28)
    dark = (gray_blur < dark_thr).astype(np.float32)

    density = cv2.GaussianBlur(dark, (0, 0), sigmaX=13, sigmaY=13)
    zone = (density > 0.33).astype(np.uint8)

    kernel = np.ones((9, 9), np.uint8)
    zone = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, kernel, iterations=2)
    zone = cv2.morphologyEx(zone, cv2.MORPH_OPEN, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(zone, connectivity=8)
    cleaned = np.zeros_like(zone)
    min_area = max(150, int(image_rgb.shape[0] * image_rgb.shape[1] * 0.00005))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    conf = np.clip(np.abs(density - 0.33) / 0.33, 0, 1)
    conf = 0.55 + 0.4 * conf
    return cleaned.astype(bool), conf.astype(np.float32), np.clip(density, 0, 1).astype(np.float32)


def _resolve_ore_class(
    model_class: str,
    talc_percent: float,
    metrics: dict,
    talc_threshold: float,
) -> tuple[str, str]:
    """
    Предметная логика ТЗ важнее классификационного класса:
    если доля талька > порога, финальный класс обязан быть оталькованным.
    Если талька мало — выбираем рядовая/трудная по доле тонких и обычных срастаний,
    а CNN-класс используем как дополнительный сигнал в тексте/таблице.
    """
    rule_class, conclusion = classify_ore(talc_percent, metrics, talc_threshold=talc_threshold)
    if model_class and model_class != rule_class:
        conclusion += f" ML-классификатор предварительно предсказал: {model_class}."
    return rule_class, conclusion


def _analysis_option(options: dict | None, key: str, default: bool = True) -> bool:
    if not isinstance(options, dict):
        return bool(default)
    return bool(options.get(key, default))


def analyze_image(
    image_rgb: np.ndarray,
    model: ModelBundle | None = None,
    runtime_info: dict | None = None,
    talc_threshold: float = 10.0,
    analysis_options: dict | None = None,
) -> dict:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    if runtime_info is None:
        runtime_info = {"mode": "cv_baseline", "message": "runtime_info не передан"}

    run_superpixel = _analysis_option(analysis_options, "run_superpixel", True)
    compute_detailed_talc_map = _analysis_option(analysis_options, "compute_detailed_talc_map", True)
    compute_gradcam = _analysis_option(analysis_options, "compute_gradcam", True)
    compute_confidence_map = _analysis_option(analysis_options, "compute_confidence_map", True)
    save_segmentation_comparison = _analysis_option(analysis_options, "save_segmentation_comparison", True)

    step_started = time.perf_counter()
    valid_mask = estimate_valid_area(image_rgb)
    timings["valid_area_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    sulfide = estimate_sulfide_masks(image_rgb)
    timings["sulfide_cv_masks_sec"] = round(time.perf_counter() - step_started, 4)

    model_payload: dict = {}
    warnings: list[str] = []

    # 1) Build the previous hybrid/CV/CNN mask.
    if model is None:
        step_started = time.perf_counter()
        talc_mask, confidence_map, talc_score = cv_talc_zone_baseline(image_rgb, valid_mask=valid_mask)
        timings["cv_talc_baseline_sec"] = round(time.perf_counter() - step_started, 4)

        gradcam = None
        talc_product = talc_score
        class_probs = {}
        model_pred_class = ""

        step_started = time.perf_counter()
        class_mask = make_class_mask(sulfide["ordinary_mask"], sulfide["thin_mask"], talc_mask)
        timings["hybrid_class_mask_build_sec"] = round(time.perf_counter() - step_started, 4)
    elif getattr(model, "mode", "") == "segmentation":
        step_started = time.perf_counter()
        model_payload = predict_with_segmentation_bundle(image_rgb, model)
        timings["torch_segmentation_call_sec"] = round(time.perf_counter() - step_started, 4)
        timings.update(model_payload.get("debug_timings_sec", {}) or {})

        class_mask = model_payload.get("class_mask")
        if class_mask is None:
            step_started = time.perf_counter()
            class_mask = make_class_mask(sulfide["ordinary_mask"], sulfide["thin_mask"], model_payload["talc_mask"])
            timings["hybrid_class_mask_build_sec"] = round(time.perf_counter() - step_started, 4)
        talc_mask = (class_mask == 3) | model_payload["talc_mask"]
        confidence_map = model_payload["confidence_map"]
        gradcam = model_payload.get("gradcam")
        talc_score = model_payload.get("talc_score")
        talc_product = model_payload.get("talc_product")
        class_probs = model_payload.get("class_probs", {})
        model_pred_class = model_payload.get("predicted_class", "")
        warnings.extend(model_payload.get("warnings", []))
    else:
        # For CNN modes this is CNN + CV/Grad-CAM pseudo-segmentation.
        # For pure superpixel mode there is no CNN, so start from CV-baseline and replace it below.
        if getattr(model, "cnn_model", None) is not None:
            step_started = time.perf_counter()
            model_payload = predict_with_cnn_bundle(
                image_rgb,
                model,
                valid_mask=valid_mask,
                analysis_options={
                    "compute_detailed_talc_map": compute_detailed_talc_map,
                    "compute_gradcam": compute_gradcam,
                    "compute_confidence_map": compute_confidence_map,
                },
            )
            timings["cnn_bundle_call_sec"] = round(time.perf_counter() - step_started, 4)
            timings.update(model_payload.get("debug_timings_sec", {}) or {})

            talc_mask = model_payload["talc_mask"]
            confidence_map = model_payload["confidence_map"]
            gradcam = model_payload.get("gradcam")
            talc_score = model_payload.get("talc_score")
            talc_product = model_payload.get("talc_product")
            class_probs = model_payload.get("class_probs", {})
            model_pred_class = model_payload.get("predicted_class", "")
            warnings.extend(model_payload.get("warnings", []))

            step_started = time.perf_counter()
            class_mask = make_class_mask(sulfide["ordinary_mask"], sulfide["thin_mask"], talc_mask)
            timings["hybrid_class_mask_build_sec"] = round(time.perf_counter() - step_started, 4)
        else:
            step_started = time.perf_counter()
            talc_mask, confidence_map, talc_score = cv_talc_zone_baseline(image_rgb, valid_mask=valid_mask)
            timings["cv_talc_baseline_sec"] = round(time.perf_counter() - step_started, 4)

            gradcam = None
            talc_product = talc_score
            class_probs = {}
            model_pred_class = ""

            step_started = time.perf_counter()
            class_mask = make_class_mask(sulfide["ordinary_mask"], sulfide["thin_mask"], talc_mask)
            timings["hybrid_class_mask_build_sec"] = round(time.perf_counter() - step_started, 4)

    hybrid_class_mask = class_mask.copy()
    segmentation_payload: dict = {}
    segmentation_source = "hybrid_cv_cnn"

    # 2) If the new superpixel segmentation model is loaded, run it too.
    # It becomes the main class_mask, while the previous hybrid mask is kept for side-by-side comparison.
    if run_superpixel and model is not None and getattr(model, "superpixel_model", None) is not None:
        try:
            step_started = time.perf_counter()
            segmentation_payload = predict_with_superpixel_bundle(
                image_rgb,
                model,
                talc_thr=talc_threshold / 100.0,
                compute_confidence_map=compute_confidence_map,
            )
            timings["superpixel_bundle_call_sec"] = round(time.perf_counter() - step_started, 4)
            timings.update(segmentation_payload.get("debug_timings_sec", {}) or {})

            class_mask = segmentation_payload["class_mask"].astype(np.uint8)
            talc_mask = segmentation_payload["talc_mask"].astype(bool)
            confidence_map = segmentation_payload["confidence_map"].astype(np.float32)
            model_pred_class = segmentation_payload.get("predicted_class") or model_pred_class
            segmentation_source = "superpixel_segmentation"
            warnings.extend(segmentation_payload.get("warnings", []))
        except Exception as exc:
            warnings.append(f"superpixel segmentation unavailable: {type(exc).__name__}: {exc}")
            segmentation_source = "hybrid_cv_cnn"

    step_started = time.perf_counter()
    metrics = class_mask_metrics(class_mask, valid_mask)
    talc_percent = float(metrics["talc_percent"])
    mean_confidence = float(confidence_map[valid_mask].mean()) if int(valid_mask.sum()) else 0.0
    timings["final_metrics_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    ore_class, conclusion = _resolve_ore_class(
        model_pred_class,
        talc_percent,
        metrics,
        talc_threshold=talc_threshold,
    )
    needs_review, reason = review_reason(
        talc_percent,
        mean_confidence,
        talc_threshold=talc_threshold,
        thin_share=metrics.get("thin_share_among_sulfides"),
        ordinary_share=metrics.get("ordinary_share_among_sulfides"),
    )
    timings["decision_logic_sec"] = round(time.perf_counter() - step_started, 4)

    if warnings:
        needs_review = True
        reason = (reason + "; " if reason and reason != "автоматическая проверка не требуется" else "") + "есть предупреждения ML-адаптера"

    timings["analyze_internal_total_sec"] = round(time.perf_counter() - total_started, 4)

    display_confidence_map = confidence_map.astype(np.float32) if compute_confidence_map else None
    display_talc_score = talc_score.astype(np.float32) if (talc_score is not None and compute_detailed_talc_map) else None
    display_gradcam = gradcam.astype(np.float32) if (gradcam is not None and compute_gradcam) else None
    display_hybrid_class_mask = hybrid_class_mask.astype(np.uint8) if save_segmentation_comparison else None
    display_superpixel_native_mask = segmentation_payload.get("superpixel_native_mask") if save_segmentation_comparison else None
    display_superpixel_confidence_map = (
        segmentation_payload.get("superpixel_confidence_map")
        if compute_confidence_map and save_segmentation_comparison
        else None
    )

    return {
        "status": "success",
        "talc_mask": talc_mask.astype(bool),
        "class_mask": class_mask.astype(np.uint8),
        "hybrid_class_mask": display_hybrid_class_mask,
        "segmentation_class_mask": segmentation_payload.get("class_mask") if save_segmentation_comparison else None,
        "superpixel_native_mask": display_superpixel_native_mask,
        "superpixel_confidence_map": display_superpixel_confidence_map,
        "superpixel_ratios": segmentation_payload.get("superpixel_ratios", {}),
        "superpixel_metrics": segmentation_payload.get("superpixel_metrics", {}),
        "segmentation_source": segmentation_source,
        "confidence_map": display_confidence_map,
        "gradcam": display_gradcam,
        "talc_score": display_talc_score,
        "talc_product": None if talc_product is None else talc_product.astype(np.float32),
        "talc_pixels": int(metrics["talc_pixels"]),
        "valid_area_pixels": int(metrics["valid_area_pixels"]),
        "talc_percent": talc_percent,
        "ore_class": ore_class,
        "model_pred_class": model_pred_class,
        "class_probs": class_probs,
        "conclusion": conclusion,
        "runtime_mode": runtime_info.get("mode", "unknown"),
        "model_message": runtime_info.get("message", ""),
        "mean_confidence": mean_confidence,
        "needs_review": bool(needs_review),
        "review_reason": reason,
        "adapter_warnings": warnings,
        "analysis_options": {
            "run_superpixel": run_superpixel,
            "compute_detailed_talc_map": compute_detailed_talc_map,
            "compute_gradcam": compute_gradcam,
            "compute_confidence_map": compute_confidence_map,
            "save_segmentation_comparison": save_segmentation_comparison,
        },
        "debug_timings_sec": timings,
        **{k: v for k, v in metrics.items() if k not in {"talc_percent", "talc_pixels", "valid_area_pixels"}},
    }
