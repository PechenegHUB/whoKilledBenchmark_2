from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES_RU = ["рядовая руда", "труднообогатимая руда", "оталькованная руда"]
CLASS_NAMES_SHORT = ["row", "difficult", "talc"]


@dataclass
class ModelBundle:
    mode: str
    device: str
    cnn_model: Any | None = None
    segmentation_model: Any | None = None
    superpixel_model: Any | None = None
    ensemble_pipeline: Any | None = None
    model_path: str = ""
    message: str = ""


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _torch_load(path: Path, device: str):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state", "model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _looks_like_efficientnet_state(state: dict) -> bool:
    keys = list(state.keys())[:50]
    joined = " ".join(keys)
    return "classifier" in joined or "features" in joined


def _load_cnn_classifier(path: Path, device: str):
    import torch
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    checkpoint = _torch_load(path, device)
    state = _extract_state_dict(checkpoint)
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state_dict")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Если state_dict совсем не похож на EfficientNet, strict=False мог проглотить слишком много.
    loaded_keys = len(set(model.state_dict().keys()) & set(state.keys()))
    if loaded_keys < 10 and not _looks_like_efficientnet_state(state):
        raise ValueError("checkpoint does not look like EfficientNet-B0 classifier")

    model.to(device)
    model.eval()
    return model, {"missing": len(missing), "unexpected": len(unexpected), "loaded_keys": loaded_keys}


def _load_segmentation_model(path: Path, device: str):
    import torch
    import segmentation_models_pytorch as smp

    checkpoint = _torch_load(path, device)
    state = _extract_state_dict(checkpoint)
    model = smp.Unet(
        encoder_name="efficientnet-b5",
        encoder_weights=None,
        in_channels=3,
        classes=4,
    )
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _find_model_files(model_path: str | Path) -> tuple[Path | None, Path | None, Path | None]:
    """Return candidate paths: CNN checkpoint, ensemble pipeline, superpixel segmentation joblib."""
    path = Path(model_path).expanduser()
    if path.is_dir():
        cnn_candidates = [
            path / "best_intergrowth_3class_b0_256.pt",
            path / "cnn.pt",
            path / "classifier.pt",
            path / "model.pt",
            path / "best_model.pth",
        ]
        ens_candidates = [
            path / "ensemble_pipeline.joblib",
            path / "ensemble.joblib",
            path / "meta_classifier.joblib",
        ]
        superpixel_candidates = [
            path / "superpixel_model.joblib",
            path / "segmentation_superpixel.joblib",
            path / "superpixel_segmentation.joblib",
            path / "model_superpixel.joblib",
            path / "model.joblib",
        ]
        cnn_path = next((p for p in cnn_candidates if p.exists()), None)
        ens_path = next((p for p in ens_candidates if p.exists()), None)
        superpixel_path = next((p for p in superpixel_candidates if p.exists()), None)
        return cnn_path, ens_path, superpixel_path
    if path.exists():
        if path.suffix.lower() == ".joblib":
            return None, None, path
        return path, None, None
    return None, None, None


def _load_superpixel_model(path: Path):
    """Load the new joblib superpixel segmentation model.

    Expected object follows inference_superpixel.py: {"clf": ..., "feature_cols": [...]}.
    """
    import joblib
    obj = joblib.load(path.expanduser().resolve())
    if isinstance(obj, dict) and "clf" in obj and "feature_cols" in obj:
        return obj
    raise ValueError("joblib object is not a superpixel segmentation bundle: expected keys 'clf' and 'feature_cols'")


def load_model_bundle(model_path: str | Path, device: str = "auto") -> tuple[ModelBundle | None, dict]:
    """
    Единая точка загрузки для обертки.

    Поддерживает:
    1) новое notebook-решение: EfficientNet-B0 3-class checkpoint;
    2) старую SMP/Unet-сегментацию, если передали такие веса;
    3) отсутствие весов: CV-baseline без падения интерфейса.
    """
    device = resolve_device(device)
    path, ensemble_path, superpixel_path = _find_model_files(model_path)

    runtime_info = {
        "device": device,
        "mode": "cv_baseline",
        "model_path": str(model_path),
        "message": "Файл весов не найден — используется CV-baseline для демонстрации обёртки.",
    }

    superpixel_model = None
    superpixel_error = ""
    if superpixel_path is not None:
        try:
            superpixel_model = _load_superpixel_model(superpixel_path)
        except Exception as exc:
            superpixel_error = f"SUPERPIXEL: {type(exc).__name__}: {exc}"

    if path is None and superpixel_model is None:
        if superpixel_error:
            runtime_info["message"] += " " + superpixel_error
        return None, runtime_info

    # Сначала пробуем как классификатор из notebook-решения.
    cnn_error = "CNN: не найден checkpoint классификатора"
    if path is not None:
        try:
            cnn_model, load_stats = _load_cnn_classifier(path, device)
            ensemble = None
            if ensemble_path is not None:
                try:
                    import joblib
                    ensemble = joblib.load(ensemble_path)
                except Exception:
                    ensemble = None

            if superpixel_model is not None and ensemble is not None:
                mode = "cnn_morph_ensemble_superpixel"
            elif superpixel_model is not None:
                mode = "cnn_morph_superpixel"
            elif ensemble is not None:
                mode = "cnn_morph_ensemble"
            else:
                mode = "cnn_morph"

            msg = f"Загружен CNN-классификатор из {path.name}."
            if ensemble is not None:
                msg += f" Подключен ансамбль {ensemble_path.name}."
            else:
                msg += " Ансамбль не найден — решение объединяется с CV-признаками эвристически."
            if superpixel_model is not None:
                msg += f" Подключена superpixel-сегментация {superpixel_path.name}."
            elif superpixel_error:
                msg += " " + superpixel_error

            bundle = ModelBundle(
                mode=mode,
                device=device,
                cnn_model=cnn_model,
                superpixel_model=superpixel_model,
                ensemble_pipeline=ensemble,
                model_path=str(path),
                message=msg,
            )
            runtime_info.update({"mode": mode, "message": msg, "load_stats": load_stats})
            return bundle, runtime_info
        except Exception as cnn_exc:
            cnn_error = f"CNN: {type(cnn_exc).__name__}: {cnn_exc}"

    # Если передали только superpixel joblib — это полноценный режим сегментации без CNN.
    if superpixel_model is not None:
        msg = f"Загружена superpixel-сегментация из {superpixel_path.name}."
        if path is not None:
            msg += f" CNN-классификатор не подключен ({cnn_error})."
        bundle = ModelBundle(
            mode="superpixel_segmentation",
            device=device,
            superpixel_model=superpixel_model,
            model_path=str(superpixel_path),
            message=msg,
        )
        runtime_info.update({"mode": "superpixel_segmentation", "message": msg})
        return bundle, runtime_info

    # Если не классификатор и не superpixel — пробуем старый SMP/Unet формат.
    if path is not None:
        try:
            seg_model = _load_segmentation_model(path, device)
            msg = f"Загружена torch-сегментационная модель из {path.name}."
            bundle = ModelBundle(mode="segmentation", device=device, segmentation_model=seg_model, model_path=str(path), message=msg)
            runtime_info.update({"mode": "segmentation", "message": msg})
            return bundle, runtime_info
        except Exception as seg_exc:
            runtime_info["message"] = (
                "Не удалось загрузить веса ни как CNN-классификатор, ни как сегментацию. "
                "Используется CV-baseline. "
                f"{cnn_error}; SEG: {type(seg_exc).__name__}: {seg_exc}"
            )
            return None, runtime_info

    runtime_info["message"] = "Не удалось загрузить модель. Используется CV-baseline. " + superpixel_error
    return None, runtime_info


def _resize_keep_aspect(img: np.ndarray, max_side: int | None):
    if max_side is None:
        return img, 1.0
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    scale = max_side / m
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def _norm01(x, p_low=1, p_high=99, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(vals, [p_low, p_high])
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1).astype(np.float32)


def _sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def _low_value_score(x, q=0.45, softness=0.25):
    x = np.asarray(x, dtype=np.float32)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    thr = np.quantile(vals, q)
    iqr = np.quantile(vals, 0.75) - np.quantile(vals, 0.25)
    s = max(float(softness * iqr), 1e-6)
    return _sigmoid((thr - x) / s).astype(np.float32)


def _high_value_score(x, q=0.75, softness=0.25):
    x = np.asarray(x, dtype=np.float32)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    thr = np.quantile(vals, q)
    iqr = np.quantile(vals, 0.75) - np.quantile(vals, 0.25)
    s = max(float(softness * iqr), 1e-6)
    return _sigmoid((x - thr) / s).astype(np.float32)


def _quantile_band_score(x, q_low=0.18, q_high=0.72, softness=0.18):
    x = np.asarray(x, dtype=np.float32)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.quantile(vals, [q_low, q_high])
    width = max(float(hi - lo), 1e-6)
    s = softness * width + 1e-6
    return (_sigmoid((x - lo) / s) * _sigmoid((hi - x) / s)).astype(np.float32)


def _local_mean_std(x, k=9):
    from scipy import ndimage
    x = x.astype(np.float32)
    mean = ndimage.uniform_filter(x, size=k, mode="reflect")
    mean2 = ndimage.uniform_filter(x * x, size=k, mode="reflect")
    var = np.maximum(mean2 - mean * mean, 0)
    return mean, np.sqrt(var).astype(np.float32)


def _structure_tensor_largest_eigenvalue(x, sigma=11.0):
    xf = x.astype(np.float32) / 255.0
    gx = cv2.Sobel(xf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(xf, cv2.CV_32F, 0, 1, ksize=3)
    Axx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    Axy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    Ayy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    trace = Axx + Ayy
    diff = Axx - Ayy
    root = np.sqrt(diff * diff + 4 * Axy * Axy)
    return (0.5 * (trace + root)).astype(np.float32)


def talc_superpixel_score(image_rgb: np.ndarray, max_side: int = 1800) -> np.ndarray:
    """Скоринговая карта талька из нового notebook-решения. При ошибках вызывающий код делает fallback."""
    from skimage import feature
    from skimage.filters import rank, threshold_multiotsu
    from skimage.morphology import disk
    from skimage.segmentation import slic

    rgb0 = np.asarray(image_rgb, dtype=np.uint8)
    h0, w0 = rgb0.shape[:2]
    rgb, _ = _resize_keep_aspect(rgb0, max_side=max_side)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.uint8)
    bg = cv2.GaussianBlur(L.astype(np.float32), (0, 0), sigmaX=45, sigmaY=45)
    L_corr = np.clip(L.astype(np.float32) - bg + np.median(bg), 0, 255).astype(np.uint8)

    try:
        segments = slic(rgb, n_segments=550, compactness=9, sigma=1, start_label=0, channel_axis=-1)
    except TypeError:
        segments = slic(rgb, n_segments=550, compactness=9, sigma=1, start_label=0)

    # superpixel q03 map
    sp_L_q03 = np.zeros_like(L, dtype=np.float32)
    for sid in np.unique(segments):
        mask = segments == sid
        sp_L_q03[mask] = float(np.quantile(L[mask], 0.03))
    dark_q3_score = _low_value_score(sp_L_q03, q=0.45, softness=0.28)

    _, local_std9 = _local_mean_std(L_corr, k=9)
    _, std_of_std9 = _local_mean_std(local_std9, k=33)
    entropy5 = rank.entropy(np.clip(L_corr, 0, 255).astype(np.uint8), disk(5)).astype(np.float32)

    std9_band_score = _quantile_band_score(local_std9, q_low=0.18, q_high=0.72, softness=0.20)
    std_stability_score = _low_value_score(std_of_std9, q=0.55, softness=0.35)
    stable_microtexture_score = np.sqrt(np.clip(std9_band_score * std_stability_score, 0, 1)).astype(np.float32)
    entropy_band_score = _quantile_band_score(entropy5, q_low=0.15, q_high=0.70, softness=0.22)

    try:
        thresholds = threshold_multiotsu(L, classes=4)
    except Exception:
        thresholds = np.quantile(L, [0.25, 0.5, 0.75])
    multiotsu_L = np.digitize(L, bins=thresholds).astype(np.uint8)
    multiotsu_dark_class = (multiotsu_L == 0).astype(np.float32)

    canny2 = feature.canny(L_corr.astype(np.float32) / 255.0, sigma=2.0).astype(np.float32)
    canny_pressure = cv2.GaussianBlur(canny2, (0, 0), sigmaX=2.0, sigmaY=2.0)
    canny_pressure = _norm01(canny_pressure, p_low=0, p_high=99)

    tensor_l1 = _structure_tensor_largest_eigenvalue(L_corr, sigma=11.0)
    tensor_pressure = _high_value_score(tensor_l1, q=0.86, softness=0.30)
    line_pressure = np.maximum(canny_pressure, tensor_pressure).astype(np.float32)
    no_line_score = 1.0 - line_pressure

    positive_soft = (
        1.35 * dark_q3_score
        + 1.05 * stable_microtexture_score
        + 0.75 * entropy_band_score
        + 1.00 * multiotsu_dark_class
    ) / (1.35 + 1.05 + 0.75 + 1.00)

    raw_candidate = positive_soft - 0.65 * line_pressure
    score_pixel = _norm01(raw_candidate, p_low=2, p_high=98)
    score_pixel = score_pixel * (0.40 + 0.60 * no_line_score)
    score_pixel = _norm01(score_pixel, p_low=1, p_high=99)

    score_candidate = np.zeros_like(score_pixel, dtype=np.float32)
    for sid in np.unique(segments):
        mask = segments == sid
        score_candidate[mask] = float(np.median(score_pixel[mask]))

    if score_candidate.shape != (h0, w0):
        score_candidate = cv2.resize(score_candidate, (w0, h0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return score_candidate.astype(np.float32)


def make_talc_mask_from_cam_and_score(
    cam: np.ndarray,
    talc_score: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    product_quantile: float = 0.86,
    min_score_quantile: float = 0.55,
    min_product_abs: float = 0.18,
    min_area: int = 35,
    morph_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    cam_n = _norm01(cam, p_low=1, p_high=99)
    score_n = _norm01(talc_score, p_low=1, p_high=99)
    if cam_n.shape != score_n.shape:
        cam_n = cv2.resize(cam_n, (score_n.shape[1], score_n.shape[0]), interpolation=cv2.INTER_LINEAR)
    product = (cam_n * score_n).astype(np.float32)
    if valid is None:
        valid = np.ones(product.shape, dtype=bool)
    else:
        valid = valid.astype(bool)
        if valid.shape != product.shape:
            valid = cv2.resize(valid.astype(np.uint8), (product.shape[1], product.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    vals = product[valid & np.isfinite(product)]
    score_vals = score_n[valid & np.isfinite(score_n)]
    if vals.size < 100 or score_vals.size < 100:
        return np.zeros_like(product, dtype=bool), product

    product_thr = max(float(np.quantile(vals, product_quantile)), float(min_product_abs))
    score_thr = float(np.quantile(score_vals, min_score_quantile))
    mask = (product >= product_thr) & (score_n >= score_thr) & valid

    if morph_radius and morph_radius > 0:
        k = 2 * int(morph_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        clean = np.zeros_like(mask, dtype=bool)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = True
        mask = clean
    return mask, product


def _preprocess_for_cnn(image_rgb: np.ndarray, size: int = 256):
    import torch
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def _gradcam_for_prediction(model, x, pred_idx: int, out_size: tuple[int, int]) -> np.ndarray:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    target_layer = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    activations = []
    gradients = []

    def forward_hook(_m, _i, o):
        activations.append(o.detach())

    def backward_hook(_m, _gi, go):
        gradients.append(go[0].detach())

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        score = logits[0, pred_idx]
        score.backward()
        if not activations or not gradients:
            return np.zeros(out_size, dtype=np.float32)
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations[0]).sum(dim=1, keepdim=True))[0, 0].detach().cpu().numpy()
        cam = _norm01(cam, p_low=0, p_high=100)
        cam = cv2.resize(cam, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)
        return _norm01(cam, p_low=1, p_high=99)
    finally:
        h1.remove(); h2.remove()



def _analysis_option(options: dict | None, key: str, default: bool = True) -> bool:
    if not isinstance(options, dict):
        return bool(default)
    return bool(options.get(key, default))


def _threshold_float_map_to_mask(
    score: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    quantile: float = 0.86,
    min_abs: float = 0.18,
    min_area: int = 35,
    morph_radius: int = 2,
) -> np.ndarray:
    """Fast fallback mask from one float map when detailed fusion maps are disabled."""
    score_n = _norm01(score, p_low=1, p_high=99)
    if valid is None:
        valid = np.ones(score_n.shape, dtype=bool)
    else:
        valid = valid.astype(bool)
        if valid.shape != score_n.shape:
            valid = cv2.resize(valid.astype(np.uint8), (score_n.shape[1], score_n.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    vals = score_n[valid & np.isfinite(score_n)]
    if vals.size < 100:
        return np.zeros_like(score_n, dtype=bool)

    thr = max(float(np.quantile(vals, quantile)), float(min_abs))
    mask = (score_n >= thr) & valid
    if morph_radius and morph_radius > 0:
        k = 2 * int(morph_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        clean = np.zeros_like(mask, dtype=bool)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = True
        mask = clean
    return mask.astype(bool)


def _cheap_talc_score(image_rgb: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    vals = gray_blur[valid_mask] if valid_mask is not None and valid_mask.any() else gray_blur
    dark = (gray_blur < np.percentile(vals, 28)).astype(np.float32)
    return cv2.GaussianBlur(dark, (0, 0), sigmaX=13, sigmaY=13).astype(np.float32)

def predict_with_cnn_bundle(
    image_rgb: np.ndarray,
    bundle: ModelBundle,
    valid_mask: np.ndarray | None = None,
    analysis_options: dict | None = None,
) -> dict:
    import torch

    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    model = bundle.cnn_model
    if model is None:
        raise ValueError("CNN model is not loaded")

    h, w = image_rgb.shape[:2]

    step_started = time.perf_counter()
    x = _preprocess_for_cnn(image_rgb, size=256).to(bundle.device)
    timings["cnn_preprocess_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    if str(bundle.device).startswith("cuda"):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    timings["cnn_forward_sec"] = round(time.perf_counter() - step_started, 4)

    pred_idx = int(np.argmax(probs))
    max_prob = float(np.max(probs))

    compute_gradcam = _analysis_option(analysis_options, "compute_gradcam", True)
    compute_detailed_talc_map = _analysis_option(analysis_options, "compute_detailed_talc_map", True)
    compute_confidence_map = _analysis_option(analysis_options, "compute_confidence_map", True)

    warnings: list[str] = []
    cam = None
    if compute_gradcam:
        # Grad-CAM требует backward, поэтому отдельный проход без no_grad.
        step_started = time.perf_counter()
        cam = _gradcam_for_prediction(model, x, pred_idx, (h, w))
        if str(bundle.device).startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        timings["gradcam_sec"] = round(time.perf_counter() - step_started, 4)

    talc_score = None
    talc_mask = None
    talc_product = None
    if compute_detailed_talc_map:
        try:
            step_started = time.perf_counter()
            talc_score = talc_superpixel_score(image_rgb, max_side=1800)
            timings["cnn_talc_score_sec"] = round(time.perf_counter() - step_started, 4)
        except Exception as exc:
            warnings.append(f"talc_superpixel_score unavailable: {type(exc).__name__}: {exc}")

    step_started = time.perf_counter()
    if talc_score is not None and cam is not None:
        talc_mask, talc_product = make_talc_mask_from_cam_and_score(
            cam,
            talc_score,
            valid=valid_mask,
            product_quantile=0.86,
            min_score_quantile=0.55,
            min_product_abs=0.18,
            min_area=35,
            morph_radius=2,
        )
    elif talc_score is not None:
        talc_mask = _threshold_float_map_to_mask(talc_score, valid=valid_mask, quantile=0.86, min_abs=0.18)
        talc_product = talc_score
    elif cam is not None:
        talc_mask = _threshold_float_map_to_mask(cam, valid=valid_mask, quantile=0.90, min_abs=0.18)
        talc_product = cam
    else:
        cheap_score = _cheap_talc_score(image_rgb, valid_mask=valid_mask)
        talc_mask = cheap_score > 0.33
        talc_product = cheap_score
    # Если классификатор почти уверен, что талька нет, не даём слабой карте талька переопределять всё.
    if float(probs[2]) < 0.28 and pred_idx != 2:
        talc_mask[:] = False
    timings["cnn_talc_fusion_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    if compute_confidence_map:
        confidence_source = talc_product if talc_product is not None else np.zeros((h, w), dtype=np.float32)
        if cam is not None:
            confidence_source = np.maximum(cam, confidence_source)
        confidence_map = np.clip(0.35 + 0.65 * max_prob * (0.55 + 0.45 * _norm01(confidence_source)), 0, 1).astype(np.float32)
    else:
        # Для таблицы всё равно нужен scalar mean_confidence; карта не сохраняется и не показывается.
        confidence_map = np.full((h, w), max(0.35, max_prob), dtype=np.float32)
    timings["cnn_confidence_sec"] = round(time.perf_counter() - step_started, 4)
    timings["cnn_total_sec"] = round(time.perf_counter() - total_started, 4)

    return {
        "predicted_class_idx": pred_idx,
        "predicted_class": CLASS_NAMES_RU[pred_idx],
        "class_probs": {
            "row": float(probs[0]),
            "difficult": float(probs[1]),
            "talc": float(probs[2]),
        },
        "gradcam": None if cam is None or not compute_gradcam else cam.astype(np.float32),
        "talc_score": None if talc_score is None or not compute_detailed_talc_map else talc_score.astype(np.float32),
        "talc_product": None if talc_product is None else talc_product.astype(np.float32),
        "talc_mask": talc_mask.astype(bool),
        "confidence_map": confidence_map,
        "display_confidence_map": confidence_map if compute_confidence_map else None,
        "warnings": warnings,
        "debug_timings_sec": timings,
    }

def predict_with_segmentation_bundle(image_rgb: np.ndarray, bundle: ModelBundle) -> dict:
    import torch

    timings: dict[str, float] = {}
    total_started = time.perf_counter()
    model = bundle.segmentation_model
    if model is None:
        raise ValueError("segmentation model is not loaded")

    h, w = image_rgb.shape[:2]
    max_side = 1536

    step_started = time.perf_counter()
    scale = min(1.0, max_side / max(h, w))
    new_h, new_w = max(32, int(h * scale)), max(32, int(w * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(bundle.device)
    timings["torch_segmentation_preprocess_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.shape[1] == 1:
            prob_talc = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            mask_small = prob_talc > 0.5
            conf_small = np.maximum(prob_talc, 1.0 - prob_talc)
            class_mask_small = np.zeros(mask_small.shape, dtype=np.uint8)
            class_mask_small[mask_small] = 3
        else:
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = np.argmax(probs, axis=0).astype(np.uint8)
            conf_small = np.max(probs, axis=0)
            class_mask_small = pred.astype(np.uint8)
            mask_small = class_mask_small == (3 if logits.shape[1] >= 4 else 1)
    if str(bundle.device).startswith("cuda"):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    timings["torch_segmentation_forward_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    class_mask = cv2.resize(class_mask_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    talc_mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    confidence_map = cv2.resize(conf_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    timings["torch_segmentation_postprocess_sec"] = round(time.perf_counter() - step_started, 4)
    timings["torch_segmentation_total_sec"] = round(time.perf_counter() - total_started, 4)
    return {
        "predicted_class_idx": None,
        "predicted_class": "",
        "class_probs": {},
        "class_mask": class_mask,
        "talc_mask": talc_mask,
        "confidence_map": np.clip(confidence_map, 0, 1),
        "gradcam": None,
        "talc_score": None,
        "talc_product": None,
        "warnings": [],
        "debug_timings_sec": timings,
    }

# ============================================================
# Superpixel segmentation adapter
# Based on inference_superpixel.py supplied by the ML team.
# Expected joblib bundle: {"clf": classifier, "feature_cols": list[str]}.
# Native classes:
# 0 unmarked, 1 talc, 2 dense_sulfide, 3 thin_sulfide,
# 4 magnetite_filler, 5 background_silicates, 6 uncertain.
# ============================================================

SUPERPIXEL_CLASS_NAMES = {
    0: "unmarked",
    1: "talc",
    2: "dense_sulfide",
    3: "thin_sulfide",
    4: "magnetite_filler",
    5: "background_silicates",
    6: "uncertain",
}


def _sp_safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _sp_resize_keep_aspect(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8), float(scale)


def _sp_make_superpixels(arr: np.ndarray, seg_px: int = 60, compactness: float = 10.0, sigma: float = 1.2) -> np.ndarray:
    from skimage.segmentation import slic
    from skimage.util import img_as_float

    h, w = arr.shape[:2]
    n_segments = int((h * w) / max(10, seg_px * seg_px))
    n_segments = max(30, min(2500, n_segments))
    return slic(
        img_as_float(arr),
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=1,
        convert2lab=True,
        enforce_connectivity=True,
        slic_zero=True,
    ).astype(np.uint32)


def _sp_rgb_to_spaces(rgb_u8: np.ndarray):
    from skimage.color import rgb2lab, rgb2hsv

    rgb = rgb_u8.astype(np.float32) / 255.0
    lab = rgb2lab(rgb).astype(np.float32)
    hsv = rgb2hsv(rgb).astype(np.float32)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return rgb, lab, hsv, gray, grad


def _sp_add_stats(feats: dict, prefix: str, values: np.ndarray):
    values = values.astype(np.float32)
    if values.size == 0:
        for name in ["mean", "std", "min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max"]:
            feats[f"{prefix}_{name}"] = 0.0
        return

    feats[f"{prefix}_mean"] = float(np.mean(values))
    feats[f"{prefix}_std"] = float(np.std(values))
    feats[f"{prefix}_min"] = float(np.min(values))
    qs = np.percentile(values, [5, 10, 25, 50, 75, 90, 95])
    for q, val in zip(["p05", "p10", "p25", "p50", "p75", "p90", "p95"], qs):
        feats[f"{prefix}_{q}"] = float(val)
    feats[f"{prefix}_max"] = float(np.max(values))


def _sp_segment_features_for_ids(rgb_u8: np.ndarray, segments: np.ndarray, segment_ids=None):
    import pandas as pd

    h, w = segments.shape
    rgb, lab, hsv, gray, grad = _sp_rgb_to_spaces(rgb_u8)
    if segment_ids is None:
        segment_ids = np.unique(segments)
    segment_ids = [int(x) for x in segment_ids if int(x) != 0]

    rows = []
    global_grad_p90 = float(np.percentile(grad, 90))

    for sid in segment_ids:
        mask = segments == sid
        area = int(mask.sum())
        if area == 0:
            continue

        ys, xs = np.nonzero(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        bbox_h = y1 - y0
        bbox_w = x1 - x0

        feats = {
            "segment_id": sid,
            "area": area,
            "area_frac": area / float(h * w),
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "bbox_area_frac": (bbox_w * bbox_h) / float(h * w),
            "extent": _sp_safe_div(area, bbox_w * bbox_h),
            "aspect": _sp_safe_div(bbox_w, bbox_h),
            "cx_norm": float(xs.mean() / max(1, w - 1)),
            "cy_norm": float(ys.mean() / max(1, h - 1)),
        }

        sub = mask[y0:y1, x0:x1]
        boundary = np.zeros_like(sub, dtype=bool)
        boundary[:-1, :] |= sub[:-1, :] != sub[1:, :]
        boundary[1:, :] |= sub[1:, :] != sub[:-1, :]
        boundary[:, :-1] |= sub[:, :-1] != sub[:, 1:]
        boundary[:, 1:] |= sub[:, 1:] != sub[:, :-1]
        boundary &= sub
        feats["boundary_frac"] = _sp_safe_div(int(boundary.sum()), area)

        for ci, name in enumerate(["r", "g", "b"]):
            _sp_add_stats(feats, name, rgb[..., ci][mask])
        for ci, name in enumerate(["lab_l", "lab_a", "lab_b"]):
            _sp_add_stats(feats, name, lab[..., ci][mask])
        for ci, name in enumerate(["hsv_h", "hsv_s", "hsv_v"]):
            _sp_add_stats(feats, name, hsv[..., ci][mask])

        _sp_add_stats(feats, "gray", gray[mask])
        _sp_add_stats(feats, "grad", grad[mask])

        r = rgb[..., 0][mask]
        g = rgb[..., 1][mask]
        b = rgb[..., 2][mask]
        eps = 1e-6
        feats["rg_mean"] = float(np.mean(r / (g + eps)))
        feats["gb_mean"] = float(np.mean(g / (b + eps)))
        feats["rb_mean"] = float(np.mean(r / (b + eps)))
        feats["yellow_score_mean"] = float(np.mean(((r + g) * 0.5) - b))

        vals_gray = gray[mask]
        vals_grad = grad[mask]
        feats["dark_frac_10"] = float(np.mean(vals_gray < 0.10))
        feats["dark_frac_15"] = float(np.mean(vals_gray < 0.15))
        feats["dark_frac_20"] = float(np.mean(vals_gray < 0.20))
        feats["bright_frac_60"] = float(np.mean(vals_gray > 0.60))
        feats["bright_frac_75"] = float(np.mean(vals_gray > 0.75))
        feats["edge_frac_global_p90"] = float(np.mean(vals_grad > global_grad_p90))
        rows.append(feats)

    return pd.DataFrame(rows)


def _sp_pred_to_mask(segments: np.ndarray, seg_ids: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.zeros(segments.shape, dtype=np.uint8)
    for sid, cls in zip(seg_ids, pred):
        out[segments == int(sid)] = int(cls)
    return out


def _sp_conf_to_map(segments: np.ndarray, seg_ids: np.ndarray, conf: np.ndarray) -> np.ndarray:
    out = np.zeros(segments.shape, dtype=np.float32)
    for sid, val in zip(seg_ids, conf):
        out[segments == int(sid)] = float(val)
    return out


def _sp_native_to_tz_mask(native_mask: np.ndarray) -> np.ndarray:
    """Map native superpixel classes to interface class_mask 0/1/2/3.

    1 talc -> 3, 2 dense_sulfide -> 1, 3 thin_sulfide -> 2,
    4 magnetite_filler is treated as thin/replacement for the simplified TЗ mask.
    Background, unmarked and uncertain are transparent in the simplified mask.
    """
    out = np.zeros(native_mask.shape, dtype=np.uint8)
    out[native_mask == 2] = 1
    out[(native_mask == 3) | (native_mask == 4)] = 2
    out[native_mask == 1] = 3
    return out


def _sp_classify_ore(ratios: dict, talc_thr: float = 0.10, thin_thr: float = 0.38) -> str:
    talc = ratios.get(1, 0.0)
    dense = ratios.get(2, 0.0)
    thin = ratios.get(3, 0.0) + ratios.get(4, 0.0)
    sulf = dense + thin
    thin_share = thin / max(sulf, 1e-6)
    if talc >= talc_thr:
        return "оталькованная руда"
    if sulf > 0.01 and thin_share >= thin_thr:
        return "труднообогатимая руда"
    return "рядовая руда"


def predict_with_superpixel_bundle(
    image_rgb: np.ndarray,
    bundle: ModelBundle,
    *,
    max_side: int = 1300,
    seg_px: int = 60,
    compactness: float = 10.0,
    sigma: float = 1.2,
    talc_thr: float = 0.10,
    thin_thr: float = 0.38,
    compute_confidence_map: bool = True,
) -> dict:
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    model_obj = bundle.superpixel_model
    if model_obj is None:
        raise ValueError("superpixel segmentation model is not loaded")

    step_started = time.perf_counter()
    clf = model_obj["clf"]
    feature_cols = list(model_obj["feature_cols"])
    timings["superpixel_prepare_model_sec"] = round(time.perf_counter() - step_started, 4)

    h, w = image_rgb.shape[:2]

    step_started = time.perf_counter()
    rgb_small, scale = _sp_resize_keep_aspect(np.asarray(image_rgb, dtype=np.uint8), max_side=max_side)
    timings["superpixel_resize_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    segments = _sp_make_superpixels(rgb_small, seg_px=seg_px, compactness=compactness, sigma=sigma)
    timings["superpixel_slic_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    feats = _sp_segment_features_for_ids(rgb_small, segments)
    timings["superpixel_features_sec"] = round(time.perf_counter() - step_started, 4)
    if feats.empty:
        raise ValueError("superpixel model produced no segments/features")

    step_started = time.perf_counter()
    seg_ids = feats["segment_id"].values.astype(np.uint32)
    X = feats.reindex(columns=feature_cols, fill_value=0)
    X = X.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    timings["superpixel_feature_table_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    pred = clf.predict(X).astype(np.uint8)
    timings["superpixel_predict_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    native_small = _sp_pred_to_mask(segments, seg_ids, pred)
    timings["superpixel_mask_build_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    if compute_confidence_map and hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X)
            seg_conf = np.max(proba, axis=1).astype(np.float32)
        except Exception:
            seg_conf = np.full(len(seg_ids), 0.78, dtype=np.float32)
    else:
        seg_conf = np.full(len(seg_ids), 0.78, dtype=np.float32)
    conf_small = _sp_conf_to_map(segments, seg_ids, seg_conf)
    timings["superpixel_confidence_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    if native_small.shape != (h, w):
        native_mask = cv2.resize(native_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        confidence_map = cv2.resize(conf_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        native_mask = native_small.astype(np.uint8)
        confidence_map = conf_small.astype(np.float32)
    timings["superpixel_resize_back_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    class_mask = _sp_native_to_tz_mask(native_mask)
    ratios = {cls: float(np.mean(native_mask == cls)) for cls in SUPERPIXEL_CLASS_NAMES}
    dense = ratios.get(2, 0.0)
    thin = ratios.get(3, 0.0) + ratios.get(4, 0.0)
    sulf = dense + thin
    timings["superpixel_metrics_sec"] = round(time.perf_counter() - step_started, 4)
    timings["superpixel_total_sec"] = round(time.perf_counter() - total_started, 4)

    return {
        "predicted_class_idx": None,
        "predicted_class": _sp_classify_ore(ratios, talc_thr=talc_thr, thin_thr=thin_thr),
        "class_probs": {},
        "class_mask": class_mask.astype(np.uint8),
        "talc_mask": (native_mask == 1),
        "confidence_map": np.clip(confidence_map, 0, 1).astype(np.float32),
        "superpixel_native_mask": native_mask.astype(np.uint8),
        "superpixel_confidence_map": np.clip(confidence_map, 0, 1).astype(np.float32) if compute_confidence_map else None,
        "superpixel_scale_from_original": float(scale),
        "superpixel_ratios": ratios,
        "superpixel_metrics": {
            "superpixel_talc_percent": 100.0 * ratios.get(1, 0.0),
            "superpixel_dense_sulfide_percent": 100.0 * dense,
            "superpixel_thin_sulfide_percent": 100.0 * ratios.get(3, 0.0),
            "superpixel_magnetite_percent": 100.0 * ratios.get(4, 0.0),
            "superpixel_background_percent": 100.0 * ratios.get(5, 0.0),
            "superpixel_uncertain_percent": 100.0 * ratios.get(6, 0.0),
            "superpixel_sulfide_percent": 100.0 * sulf,
            "superpixel_thin_share_in_sulfides": 100.0 * thin / max(sulf, 1e-6),
        },
        "gradcam": None,
        "talc_score": None,
        "talc_product": None,
        "warnings": [],
        "debug_timings_sec": timings,
    }
