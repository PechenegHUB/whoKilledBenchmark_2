from __future__ import annotations

from pathlib import Path, PureWindowsPath
from datetime import datetime
import time
from io import BytesIO
import base64
import html
import zipfile
import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.classification import ORE_CLASSES
from src.image_io import persist_uploaded_file, read_image_any, load_mask, load_class_mask, safe_stem
from src.inference import analyze_image, load_segmentation_model
from src.review_store import OutputStore
from src.visualization import make_overlay_pil, make_diff_overlay_pil, resize_for_display, make_class_overlay_pil, make_heatmap_pil, make_superpixel_overlay_pil
from src.reporting import generate_pdf_report
from src.canvas_tools import apply_canvas_edits
from src.frontend_mask_editor import frontend_mask_editor, decode_editor_image
from src.gis_export import save_mask_geojson, write_errors_csv, write_gis_readme

CANVAS_AVAILABLE = True


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ZIP_EXTS = {".zip"}
MAX_ZIP_IMAGE_FILES = 500
MAX_ZIP_MEMBER_BYTES = 512 * 1024 * 1024


class InMemoryUploadedFile:
    """Small Streamlit UploadedFile-like wrapper for images extracted from ZIP."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _is_supported_image_name(name: str) -> bool:
    return Path(str(name)).suffix.lower() in IMAGE_EXTS


def _is_zip_name(name: str) -> bool:
    return Path(str(name)).suffix.lower() in ZIP_EXTS


def _expand_batch_uploads(uploaded_items) -> tuple[list, list[str]]:
    """
    Convert direct image uploads and ZIP archives into one flat list of image-like files.

    ZIP members are read into memory and wrapped as objects with .name and .getvalue(),
    so the rest of the pipeline can keep using _run_analysis unchanged.
    """
    files: list = []
    warnings_list: list[str] = []

    for item in uploaded_items or []:
        item_name = getattr(item, "name", "uploaded_file")
        suffix = Path(item_name).suffix.lower()

        if suffix in IMAGE_EXTS:
            files.append(item)
            continue

        if suffix not in ZIP_EXTS:
            warnings_list.append(f"{item_name}: пропущен неподдерживаемый формат.")
            continue

        try:
            archive_bytes = item.getvalue()
            with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
                image_infos = []
                for info in zf.infolist():
                    member_name = info.filename.replace("\\", "/")
                    base_name = Path(member_name).name
                    if info.is_dir():
                        continue
                    if member_name.startswith("__MACOSX/") or base_name.startswith("."):
                        continue
                    if not _is_supported_image_name(member_name):
                        continue
                    image_infos.append(info)

                if not image_infos:
                    warnings_list.append(f"{item_name}: в архиве не найдено PNG/JPG/TIFF изображений.")
                    continue

                if len(image_infos) > MAX_ZIP_IMAGE_FILES:
                    warnings_list.append(
                        f"{item_name}: найдено {len(image_infos)} изображений, "
                        f"будут взяты первые {MAX_ZIP_IMAGE_FILES}."
                    )
                    image_infos = image_infos[:MAX_ZIP_IMAGE_FILES]

                for info in image_infos:
                    member_name = info.filename.replace("\\", "/")
                    if info.file_size > MAX_ZIP_MEMBER_BYTES:
                        warnings_list.append(
                            f"{item_name}/{member_name}: файл больше "
                            f"{MAX_ZIP_MEMBER_BYTES // (1024 * 1024)} МБ, пропущен."
                        )
                        continue
                    try:
                        data = zf.read(info)
                    except Exception as exc:
                        warnings_list.append(f"{item_name}/{member_name}: не удалось прочитать из ZIP ({exc}).")
                        continue

                    # Keep archive name in original_name for traceability, but do not extract paths to disk.
                    files.append(InMemoryUploadedFile(name=f"{Path(item_name).name}/{member_name}", data=data))
        except zipfile.BadZipFile:
            warnings_list.append(f"{item_name}: это невалидный ZIP-архив.")
        except Exception as exc:
            warnings_list.append(f"{item_name}: ошибка чтения ZIP ({type(exc).__name__}: {exc}).")

    return files, warnings_list


def _human_size(num_bytes: int | float | None) -> str:
    """Format byte size for ZIP/direct upload preview."""
    try:
        size = float(num_bytes or 0)
    except Exception:
        return "—"
    units = ["Б", "КБ", "МБ", "ГБ"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.1f} {units[idx]}"


def _uploaded_size(uploaded_file) -> int:
    """Return size for Streamlit UploadedFile or in-memory ZIP member."""
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        try:
            return int(size)
        except Exception:
            pass
    try:
        return len(uploaded_file.getvalue())
    except Exception:
        return 0


def _batch_preview_df(batch_files: list, max_rows: int = 80) -> pd.DataFrame:
    """Build a compact preview of images that will be sent to batch analysis."""
    rows = []
    for idx, file in enumerate(batch_files[:max_rows], start=1):
        name = getattr(file, "name", f"image_{idx}")
        in_zip = "/" in str(name) and str(name).lower().split("/")[0].endswith(".zip")
        rows.append({
            "№": idx,
            "Файл": name,
            "Источник": "ZIP" if in_zip else "Прямая загрузка",
            "Формат": Path(str(name)).suffix.lower().replace(".", "").upper(),
            "Размер": _human_size(_uploaded_size(file)),
        })
    if len(batch_files) > max_rows:
        rows.append({
            "№": "…",
            "Файл": f"и ещё {len(batch_files) - max_rows} изображений",
            "Источник": "—",
            "Формат": "—",
            "Размер": "—",
        })
    return pd.DataFrame(rows)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
STORE = OutputStore(OUTPUT_ROOT)

st.set_page_config(
    page_title="Скажи мне, кто твой шлиф",
    page_icon="🔬",
    layout="wide",
)

# Анти-мерцание для Streamlit. При rerun Streamlit помечает старые элементы как stale
# и визуально снижает opacity. На canvas это выглядит как постоянное затемнение во время
# рисования. CSS ниже не отменяет rerun, но убирает именно визуальное приглушение.
st.markdown(
    """
    <style>
    div[data-testid="stStatusWidget"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {
        visibility: hidden !important;
        display: none !important;
    }

    .stale-element,
    .element-container:has(.stale-element),
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] *,
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        opacity: 1 !important;
        filter: none !important;
    }

    [data-testid="stAppViewContainer"] * {
        transition-property: none !important;
    }

    iframe[title*="mask_editor"],
    iframe[title*="component"] {
        opacity: 1 !important;
        filter: none !important;
    }

    .fit-image-card {
        width: 100%;
        margin: 0.25rem 0 0.75rem 0;
    }

    .fit-image-card img {
        display: block;
        width: auto;
        max-width: 100%;
        max-height: min(78vh, 860px);
        object-fit: contain;
        margin: 0 auto;
        border-radius: 10px;
    }

    .fit-image-caption {
        margin-top: 0.35rem;
        text-align: center;
        color: rgba(250, 250, 250, 0.72);
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str, device: str):
    return load_segmentation_model(model_path=model_path, device=device)


def _bool(v) -> bool:
    return str(v).lower() in {"true", "1", "yes", "да"}


ANALYSIS_OPTION_DEFAULTS = {
    "run_superpixel": True,
    "compute_detailed_talc_map": True,
    "compute_gradcam": True,
    "compute_confidence_map": True,
    "save_segmentation_comparison": True,
    "save_visual_overlays": True,
}


def _normalize_analysis_options(options: dict | None = None) -> dict:
    """Return safe calculation/output switches for single and batch runs.

    Core outputs are always calculated: input image, binary talc mask, final class_mask,
    geological percentages and final ore class. The switches below control heavier
    optional blocks and optional visual artifacts.
    """
    normalized = dict(ANALYSIS_OPTION_DEFAULTS)
    if isinstance(options, dict):
        for key in normalized:
            if key in options:
                normalized[key] = bool(options[key])

    # Without superpixel there is nothing meaningful to compare against except the final mask.
    if not normalized["run_superpixel"]:
        normalized["save_segmentation_comparison"] = False
    return normalized


def _analysis_options_panel(prefix: str, *, expanded: bool = True) -> dict:
    """Render a compact block with calculation switches and return selected options."""
    with st.expander("Что считать в этом запуске", expanded=expanded):
        st.checkbox(
            "Базовые метрики, итоговый класс и итоговая маска",
            value=True,
            disabled=True,
            key=f"{prefix}_core_metrics_locked",
            help="Это обязательная часть: без неё нельзя получить долю талька, класс руды и очередь проверки.",
        )
        left, right = st.columns(2)
        with left:
            run_superpixel = st.checkbox(
                "Superpixel-сегментация",
                value=ANALYSIS_OPTION_DEFAULTS["run_superpixel"],
                key=f"{prefix}_run_superpixel",
                help="Самый тяжёлый CPU-блок. Можно выключить для быстрого демо или массового прогона.",
            )
            compute_detailed_talc_map = st.checkbox(
                "Подробная карта talc-score / тальк-уверенности",
                value=ANALYSIS_OPTION_DEFAULTS["compute_detailed_talc_map"],
                key=f"{prefix}_compute_detailed_talc_map",
                help="Считает подробную тепловую карту вероятных тальковых зон. Если выключить, останется быстрая маска и проценты.",
            )
            compute_gradcam = st.checkbox(
                "Grad-CAM для CNN",
                value=ANALYSIS_OPTION_DEFAULTS["compute_gradcam"],
                key=f"{prefix}_compute_gradcam",
                help="Дополнительная explainability-карта. Требует отдельного backward-прохода CNN.",
            )
        with right:
            compute_confidence_map = st.checkbox(
                "Карта общей уверенности",
                value=ANALYSIS_OPTION_DEFAULTS["compute_confidence_map"],
                key=f"{prefix}_compute_confidence_map",
                help="Сохраняет тепловую карту уверенности. Средняя уверенность для таблицы всё равно считается облегчённо.",
            )
            save_segmentation_comparison = st.checkbox(
                "Сравнение сегментаций",
                value=ANALYSIS_OPTION_DEFAULTS["save_segmentation_comparison"],
                disabled=not run_superpixel,
                key=f"{prefix}_save_segmentation_comparison",
                help="Сохраняет дополнительные hybrid/superpixel-маски для просмотра рядом. Не влияет на итоговые проценты.",
            )
            save_visual_overlays = st.checkbox(
                "Overlay-картинки для ZIP/PDF",
                value=ANALYSIS_OPTION_DEFAULTS["save_visual_overlays"],
                key=f"{prefix}_save_visual_overlays",
                help="Если выключить, будут сохранены маски и таблицы, а overlay можно строить в браузере при просмотре.",
            )

        options = _normalize_analysis_options({
            "run_superpixel": run_superpixel,
            "compute_detailed_talc_map": compute_detailed_talc_map,
            "compute_gradcam": compute_gradcam,
            "compute_confidence_map": compute_confidence_map,
            "save_segmentation_comparison": save_segmentation_comparison,
            "save_visual_overlays": save_visual_overlays,
        })

        disabled = [name for name, enabled in {
            "superpixel": options["run_superpixel"],
            "talc-score": options["compute_detailed_talc_map"],
            "Grad-CAM": options["compute_gradcam"],
            "карта уверенности": options["compute_confidence_map"],
            "сравнение сегментаций": options["save_segmentation_comparison"],
            "overlay": options["save_visual_overlays"],
        }.items() if not enabled]
        if disabled:
            st.caption("Будет пропущено: " + ", ".join(disabled) + ".")
        else:
            st.caption("Включён полный режим: считаются все маски, карты и визуальные артефакты.")
        return options


def _resize_mask_to_shape(mask, target_shape: tuple[int, int]):
    """Resize binary/class mask to image shape with nearest-neighbor sampling.

    This prevents overlay crashes when the editor preview image and the mask
    were resized with different max_side values.
    """
    arr = np.asarray(mask)
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if arr.shape[:2] == (target_h, target_w):
        return arr

    src_dtype = arr.dtype
    pil = Image.fromarray(arr.astype(np.uint8), mode="L")
    pil = pil.resize((target_w, target_h), Image.Resampling.NEAREST)
    out = np.asarray(pil)
    if src_dtype == np.bool_ or src_dtype == bool:
        return out > 0
    return out.astype(src_dtype, copy=False)


def _format_pct(v) -> str:
    try:
        return f"{float(v):.2f}%"
    except Exception:
        return "—"


def _analysis_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = [
        "sample_id", "batch_id", "original_name", "status", "model_class", "raw_model_class", "final_class",
        "talc_percent_model", "sulfide_percent", "ordinary_share_among_sulfides", "thin_share_among_sulfides",
        "mean_confidence", "needs_review", "review_reason", "review_status", "final_source",
        "processing_time_sec", "runtime_mode", "segmentation_source", "superpixel_talc_percent", "error_message",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def _result_metrics_df(result: dict) -> pd.DataFrame:
    rows = [
        ("Доля талька", result.get("talc_percent", 0.0), "%"),
        ("Общая доля сульфидов", result.get("sulfide_percent", 0.0), "%"),
        ("Обычные срастания от площади", result.get("ordinary_percent", 0.0), "%"),
        ("Тонкие срастания от площади", result.get("thin_percent", 0.0), "%"),
        ("Обычные среди сульфидов", result.get("ordinary_share_among_sulfides", 0.0), "%"),
        ("Тонкие среди сульфидов", result.get("thin_share_among_sulfides", 0.0), "%"),
        ("Средняя уверенность", result.get("mean_confidence", 0.0), ""),
    ]
    return pd.DataFrame([
        {"Метрика": name, "Значение": f"{float(value):.2f}{unit}" if unit else f"{float(value):.2f}"}
        for name, value, unit in rows
    ])


def _class_probs_df(result: dict) -> pd.DataFrame:
    probs = result.get("class_probs") or {}
    labels = {"row": "рядовая", "difficult": "труднообогатимая", "talc": "оталькованная"}
    return pd.DataFrame([
        {"Класс CNN": labels.get(k, k), "Вероятность": f"{float(v):.3f}"}
        for k, v in probs.items()
    ])


def _debug_timing_df(result: dict) -> pd.DataFrame:
    """Build a detailed timing table for single-image debugging."""
    timings = result.get("debug_timings_sec") or {}
    if not isinstance(timings, dict):
        timings = {}

    labels = {
        "persist_upload_sec": "Сохранение загруженного файла",
        "read_image_sec": "Чтение изображения",
        "analyze_image_sec": "Инференс + построение масок — всего",
        "valid_area_sec": "  ├─ Маска валидной области",
        "sulfide_cv_masks_sec": "  ├─ CV-поиск сульфидов",
        "cv_talc_baseline_sec": "  ├─ CV-baseline талька",
        "cnn_bundle_call_sec": "  ├─ CNN/Grad-CAM блок — всего",
        "cnn_preprocess_sec": "  │  ├─ CNN preprocess",
        "cnn_forward_sec": "  │  ├─ CNN forward",
        "gradcam_sec": "  │  ├─ Grad-CAM backward",
        "cnn_talc_score_sec": "  │  ├─ Talc-score для CNN",
        "cnn_talc_fusion_sec": "  │  ├─ Fusion Grad-CAM × talc-score",
        "cnn_talc_fallback_sec": "  │  ├─ Fallback talc-score",
        "cnn_confidence_sec": "  │  └─ Карта уверенности CNN",
        "cnn_total_sec": "  │  Итого CNN-блок",
        "hybrid_class_mask_build_sec": "  ├─ Сборка hybrid class_mask",
        "torch_segmentation_call_sec": "  ├─ Torch-сегментация — всего",
        "torch_segmentation_preprocess_sec": "  │  ├─ Torch preprocess",
        "torch_segmentation_forward_sec": "  │  ├─ Torch forward",
        "torch_segmentation_postprocess_sec": "  │  └─ Torch postprocess",
        "torch_segmentation_total_sec": "  │  Итого torch-сегментация",
        "superpixel_bundle_call_sec": "  ├─ Superpixel-сегментация — всего",
        "superpixel_prepare_model_sec": "  │  ├─ Подготовка clf/features",
        "superpixel_resize_sec": "  │  ├─ Resize для SLIC",
        "superpixel_slic_sec": "  │  ├─ SLIC superpixels",
        "superpixel_features_sec": "  │  ├─ Расчёт признаков superpixels",
        "superpixel_feature_table_sec": "  │  ├─ Подготовка feature table",
        "superpixel_predict_sec": "  │  ├─ sklearn/joblib predict",
        "superpixel_mask_build_sec": "  │  ├─ Сборка native mask",
        "superpixel_confidence_sec": "  │  ├─ Predict_proba / confidence map",
        "superpixel_resize_back_sec": "  │  ├─ Resize маски к исходнику",
        "superpixel_metrics_sec": "  │  └─ Метрики superpixel",
        "superpixel_total_sec": "  │  Итого superpixel-блок",
        "final_metrics_sec": "  ├─ Финальные метрики по class_mask",
        "decision_logic_sec": "  └─ Логика класса и review_reason",
        "analyze_internal_total_sec": "  Итого внутри analyze_image",
        "save_outputs_sec": "Сохранение outputs",
        "total_sec": "Итого",
    }
    comments = {
        "persist_upload_sec": "диск / временное сохранение",
        "read_image_sec": "PIL/TIFF/PNG/JPG → RGB",
        "analyze_image_sec": "этапы ниже выполняются в основном последовательно",
        "valid_area_sec": "CPU, определение рабочей области изображения",
        "sulfide_cv_masks_sec": "CPU/OpenCV, обычные и тонкие зоны для hybrid fallback",
        "cv_talc_baseline_sec": "CPU/OpenCV fallback, если нет CNN/superpixel",
        "cnn_bundle_call_sec": "PyTorch; на CUDA ускоряется только этот блок",
        "cnn_preprocess_sec": "CPU→tensor→device",
        "cnn_forward_sec": "CUDA/CPU forward EfficientNet",
        "gradcam_sec": "CUDA/CPU backward для карты внимания",
        "cnn_talc_score_sec": "CPU/skimage/OpenCV, не CUDA",
        "cnn_talc_fusion_sec": "CPU/numpy/OpenCV",
        "cnn_confidence_sec": "CPU/numpy",
        "hybrid_class_mask_build_sec": "CPU/numpy, сборка 0/1/2/3",
        "torch_segmentation_call_sec": "если подключена torch-сегментация",
        "superpixel_bundle_call_sec": "CPU pipeline: SLIC → признаки → sklearn",
        "superpixel_slic_sec": "CPU/skimage; обычно один из самых тяжёлых этапов",
        "superpixel_features_sec": "CPU/numpy/skimage; признаки по каждому суперпикселю",
        "superpixel_predict_sec": "CPU/sklearn/joblib predict",
        "superpixel_resize_back_sec": "CPU/OpenCV, возврат к размеру исходника",
        "final_metrics_sec": "CPU/numpy, проценты талька/сульфидов",
        "decision_logic_sec": "быстрая экспертная логика ТЗ",
        "analyze_internal_total_sec": "суммарно внутри analyze_image",
        "save_outputs_sec": "PNG-маски, overlay, CSV/JSON/GIS paths",
        "total_sec": "полное время одиночного анализа",
    }

    ordered_keys = [
        "persist_upload_sec",
        "read_image_sec",
        "analyze_image_sec",
        "valid_area_sec",
        "sulfide_cv_masks_sec",
        "cv_talc_baseline_sec",
        "cnn_bundle_call_sec",
        "cnn_preprocess_sec",
        "cnn_forward_sec",
        "gradcam_sec",
        "cnn_talc_score_sec",
        "cnn_talc_fusion_sec",
        "cnn_talc_fallback_sec",
        "cnn_confidence_sec",
        "cnn_total_sec",
        "hybrid_class_mask_build_sec",
        "torch_segmentation_call_sec",
        "torch_segmentation_preprocess_sec",
        "torch_segmentation_forward_sec",
        "torch_segmentation_postprocess_sec",
        "torch_segmentation_total_sec",
        "superpixel_bundle_call_sec",
        "superpixel_prepare_model_sec",
        "superpixel_resize_sec",
        "superpixel_slic_sec",
        "superpixel_features_sec",
        "superpixel_feature_table_sec",
        "superpixel_predict_sec",
        "superpixel_mask_build_sec",
        "superpixel_confidence_sec",
        "superpixel_resize_back_sec",
        "superpixel_metrics_sec",
        "superpixel_total_sec",
        "final_metrics_sec",
        "decision_logic_sec",
        "analyze_internal_total_sec",
        "save_outputs_sec",
        "total_sec",
    ]
    ordered_keys += [k for k in timings.keys() if k not in ordered_keys]

    total = timings.get("total_sec") or result.get("processing_time_sec") or 0
    try:
        total = float(total)
    except Exception:
        total = 0.0

    rows = []
    for key in ordered_keys:
        if key not in timings:
            continue
        try:
            value = float(timings.get(key) or 0.0)
        except Exception:
            continue
        rows.append({
            "Этап": labels.get(key, str(key).replace("_", " ")),
            "Время, с": f"{value:.3f}",
            "% от итого": f"{(value / total * 100):.1f}%" if total > 0 and key != "total_sec" else "—",
            "Комментарий": comments.get(key, "дополнительный тайминг адаптера"),
        })

    if not rows and result.get("processing_time_sec") not in (None, ""):
        try:
            value = float(result.get("processing_time_sec") or 0.0)
            rows.append({"Этап": "Итого", "Время, с": f"{value:.3f}", "% от итого": "—", "Комментарий": "полное время одиночного анализа"})
        except Exception:
            pass

    return pd.DataFrame(rows)

def _make_outputs_zip(batch_id: str | None = None) -> Path:
    """Build ZIP with either all outputs or only artifacts belonging to one batch."""
    safe_batch = str(batch_id or "").strip()
    zip_name = f"nornickel_outputs_{safe_batch}.zip" if safe_batch else "nornickel_outputs.zip"
    zip_path = OUTPUT_ROOT / zip_name
    if zip_path.exists():
        zip_path.unlink()

    rows = STORE.read_rows()
    selected_rows = rows
    if safe_batch:
        selected_rows = [r for r in rows if str(r.get("batch_id", "")) == safe_batch]

    artifact_keys = [
        "image_path", "model_mask_path", "class_mask_path", "hybrid_class_mask_path", "superpixel_mask_path", "expert_mask_path",
        "overlay_path", "class_overlay_path", "hybrid_class_overlay_path", "superpixel_overlay_path", "diff_overlay_path",
        "confidence_map_path", "superpixel_confidence_path", "gradcam_path", "talc_score_path",
    ]
    added: set[Path] = set()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if safe_batch:
            # Keep a batch-specific CSV inside the archive.
            batch_csv = OUTPUT_ROOT / "reviews" / f"analysis_results_{safe_batch}.csv"
            _analysis_table(selected_rows).to_csv(batch_csv, index=False, encoding="utf-8-sig")
            zf.write(batch_csv, batch_csv.relative_to(OUTPUT_ROOT))
            added.add(batch_csv.resolve())

            for row in selected_rows:
                expected = STORE.result_paths(row.get("sample_id", ""))
                for key in artifact_keys:
                    fallback = expected.get(key) if key in expected else None
                    path = _resolve_existing_artifact(row.get(key, ""), fallback)
                    if path and path.is_file() and path.resolve() not in added and path != zip_path:
                        zf.write(path, path.relative_to(OUTPUT_ROOT) if path.is_relative_to(OUTPUT_ROOT) else path.name)
                        added.add(path.resolve())
        else:
            for path in OUTPUT_ROOT.rglob("*"):
                if path.is_file() and path != zip_path:
                    zf.write(path, path.relative_to(OUTPUT_ROOT))
    return zip_path




def _resolve_gis_mask_for_row(row: dict) -> tuple[Path | None, str]:
    """Return preferred mask for GIS export: expert class mask first, then final class mask."""
    sample_id = str(row.get("sample_id") or "")
    expected = STORE.result_paths(sample_id) if sample_id else {}
    candidates = [
        ("expert_class_mask_path", "expert_class_mask"),
        ("class_mask_path", "class_mask"),
    ]
    for key, source_name in candidates:
        fallback = expected.get(key) if isinstance(expected, dict) else None
        path = _resolve_existing_artifact(row.get(key, ""), fallback)
        if path is not None and path.is_file():
            return path, source_name
    return None, ""


def _make_single_gis_export(row: dict) -> Path:
    """Export one sample mask to GeoJSON in local image coordinates."""
    sample_id = safe_stem(str(row.get("sample_id") or "sample"))
    mask_path, source_name = _resolve_gis_mask_for_row(row)
    if mask_path is None:
        raise RuntimeError(
            "Для GIS-экспорта не найдена мультиклассовая маска. "
            "Нужен class_mask_path или expert_class_mask_path."
        )

    mask = load_class_mask(mask_path)
    gis_dir = OUTPUT_ROOT / "gis"
    gis_dir.mkdir(parents=True, exist_ok=True)
    out_path = gis_dir / f"{sample_id}_{source_name}.geojson"
    return save_mask_geojson(mask, row=row, out_path=out_path, source_mask=source_name)


def _make_gis_export_zip(rows: list[dict], batch_id: str | None = None) -> Path:
    """Create a ZIP with GeoJSON files for all selected successful rows."""
    safe_batch = str(batch_id or "").strip()
    gis_dir = OUTPUT_ROOT / "gis"
    gis_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"gis_export_{safe_batch}_{stamp}.zip" if safe_batch else f"gis_export_all_{stamp}.zip"
    zip_path = gis_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()

    selected_rows = rows
    if safe_batch:
        selected_rows = [r for r in rows if str(r.get("batch_id", "")) == safe_batch]

    created: list[Path] = []
    errors: list[dict] = []
    for row in selected_rows:
        if str(row.get("status", "success")).lower() == "error":
            continue
        try:
            geojson_path = _make_single_gis_export(row)
            created.append(geojson_path)
        except Exception as exc:
            errors.append({
                "sample_id": row.get("sample_id", ""),
                "original_name": row.get("original_name", ""),
                "error": f"{type(exc).__name__}: {exc}",
            })

    readme_path = write_gis_readme(gis_dir / "README_GIS.txt")
    errors_path = write_errors_csv(gis_dir / f"gis_export_errors_{safe_batch or 'all'}_{stamp}.csv", errors)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme_path, "README_GIS.txt")
        if errors:
            zf.write(errors_path, errors_path.name)
        for path in created:
            zf.write(path, f"geojson/{path.name}")

    if not created:
        raise RuntimeError(
            "Не удалось сформировать ни одного GeoJSON. "
            "Проверьте, что у выбранных образцов сохранены class_mask или expert_class_mask."
        )
    return zip_path

def _make_pdf_report(rows: list[dict], batch_id: str | None = None, title: str | None = None) -> Path:
    """Generate PDF report from current journal rows."""
    reports_dir = OUTPUT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_batch = str(batch_id or "").strip()
    if safe_batch:
        report_path = reports_dir / f"report_{safe_batch}.pdf"
        report_title = title or "Отчет по партии шлифов"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"report_all_{stamp}.pdf"
        report_title = title or "Сводный отчет по анализу шлифов"
    return generate_pdf_report(
        rows,
        output_root=OUTPUT_ROOT,
        project_root=PROJECT_ROOT,
        report_path=report_path,
        title=report_title,
        batch_id=safe_batch or None,
        max_samples=30,
    )


def _make_single_pdf_report(row: dict) -> Path:
    """Generate a PDF report for one analyzed sample."""
    reports_dir = OUTPUT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    sample_id = safe_stem(str(row.get("sample_id") or "sample"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"report_{sample_id}_{stamp}.pdf"
    return generate_pdf_report(
        [row],
        output_root=OUTPUT_ROOT,
        project_root=PROJECT_ROOT,
        report_path=report_path,
        title="Отчет по одному шлифу",
        batch_id=None,
        max_samples=1,
    )


def _basename_from_any_path(path_value) -> str:
    """Return filename from POSIX or Windows-like path stored in CSV."""
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    if "\\" in raw:
        return PureWindowsPath(raw).name
    return Path(raw).name


def _resolve_existing_artifact(path_value, fallback_path: Path | None = None) -> Path | None:
    """
    Resolve an artifact path saved in CSV.

    Why this is needed: older CSV rows may contain absolute paths from another
    folder or another OS, for example C:/old_project/outputs/masks/... .
    The app should still open files from the current PROJECT_ROOT/outputs.
    """
    candidates: list[Path] = []
    raw = str(path_value or "").strip()

    if raw:
        raw_path = Path(raw).expanduser()
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.append(PROJECT_ROOT / raw_path)
            candidates.append(OUTPUT_ROOT / raw_path)

        basename = _basename_from_any_path(raw)
        if basename and fallback_path is not None:
            candidates.append(fallback_path.parent / basename)

    if fallback_path is not None:
        candidates.append(Path(fallback_path))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _read_saved_float_map(path: Path | None) -> np.ndarray | None:
    """Read a saved 8-bit PNG float-like map as 0..1 array for preview overlays."""
    if path is None or not path.is_file():
        return None
    try:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return arr
    except Exception:
        return None


def _load_batch_preview_payload(row: dict, heatmap_kind: str = "confidence", mask_kind: str = "final") -> dict:
    """Load saved artifacts for post-batch visual inspection."""
    sample_id = str(row.get("sample_id", ""))
    expected = STORE.result_paths(sample_id)

    image_path = _resolve_existing_artifact(row.get("image_path", ""), expected["image_path"])
    if image_path is None:
        raise FileNotFoundError("Не найдено сохранённое исходное изображение для выбранной строки.")
    image_rgb = read_image_any(image_path)

    class_overlay_path = _resolve_existing_artifact(row.get("class_overlay_path", ""), expected["class_overlay_path"])
    overlay_path = _resolve_existing_artifact(row.get("overlay_path", ""), expected["overlay_path"])
    class_mask_path = _resolve_existing_artifact(row.get("class_mask_path", ""), expected["class_mask_path"])
    model_mask_path = _resolve_existing_artifact(row.get("model_mask_path", ""), expected["model_mask_path"])
    hybrid_overlay_path = _resolve_existing_artifact(row.get("hybrid_class_overlay_path", ""), expected.get("hybrid_class_overlay_path"))
    hybrid_mask_path = _resolve_existing_artifact(row.get("hybrid_class_mask_path", ""), expected.get("hybrid_class_mask_path"))
    superpixel_overlay_path = _resolve_existing_artifact(row.get("superpixel_overlay_path", ""), expected.get("superpixel_overlay_path"))
    superpixel_mask_path = _resolve_existing_artifact(row.get("superpixel_mask_path", ""), expected.get("superpixel_mask_path"))

    mask_preview = None
    mask_caption = "Мультиклассовая маска: зелёный = обычные, красный = тонкие, синий = тальк"

    if mask_kind == "superpixel":
        mask_caption = "Superpixel-модель: синий = тальк, зелёный = плотные сульфиды, красный = тонкие, жёлтый = магнетит/замещение, серый = фон, фиолетовый = спорное"
        if superpixel_overlay_path is not None:
            mask_preview = read_image_any(superpixel_overlay_path)
        elif superpixel_mask_path is not None:
            mask_preview = make_superpixel_overlay_pil(image_rgb, load_class_mask(superpixel_mask_path), alpha=1.0)
    elif mask_kind == "hybrid":
        mask_caption = "CV/CNN-гибрид: зелёный = обычные, красный = тонкие, синий = тальк"
        if hybrid_overlay_path is not None:
            mask_preview = read_image_any(hybrid_overlay_path)
        elif hybrid_mask_path is not None:
            mask_preview = make_class_overlay_pil(image_rgb, load_class_mask(hybrid_mask_path), alpha=1.0)
    elif mask_kind == "talc":
        mask_caption = "Только тальк: синяя зона = тальк / оталькованные участки"
        if overlay_path is not None:
            mask_preview = read_image_any(overlay_path)
        elif model_mask_path is not None:
            mask_preview = make_overlay_pil(image_rgb, load_mask(model_mask_path), alpha=1.0)
    else:
        if class_overlay_path is not None:
            mask_preview = read_image_any(class_overlay_path)
        elif class_mask_path is not None:
            mask_preview = make_class_overlay_pil(image_rgb, load_class_mask(class_mask_path), alpha=1.0)
        elif overlay_path is not None:
            mask_preview = read_image_any(overlay_path)
            mask_caption = "Маска талька / fallback-overlay"
        elif model_mask_path is not None:
            mask_preview = make_overlay_pil(image_rgb, load_mask(model_mask_path), alpha=1.0)
            mask_caption = "Синяя зона = тальк / fallback-mask"

    heatmap_options = {
        "confidence": ("confidence_map_path", "confidence_map_path", "Карта уверенности"),
        "superpixel_confidence": ("superpixel_confidence_path", "superpixel_confidence_path", "Уверенность superpixel-сегментации"),
        "gradcam": ("gradcam_path", "gradcam_path", "Grad-CAM: куда смотрел CNN"),
        "talc_score": ("talc_score_path", "talc_score_path", "Talc-score: вероятные тальковые зоны"),
    }
    row_key, expected_key, heatmap_caption = heatmap_options.get(heatmap_kind, heatmap_options["confidence"])
    heatmap_path = _resolve_existing_artifact(row.get(row_key, ""), expected.get(expected_key))
    heatmap = _read_saved_float_map(heatmap_path)
    heatmap_preview = make_heatmap_pil(image_rgb, heatmap, alpha=1.0) if heatmap is not None else None

    return {
        "image_rgb": image_rgb,
        "mask_preview": mask_preview,
        "mask_caption": mask_caption,
        "heatmap_preview": heatmap_preview,
        "heatmap_caption": heatmap_caption,
        "image_path": image_path,
        "heatmap_path": heatmap_path,
    }


def _batch_viewer_labels(view_df: pd.DataFrame) -> list[str]:
    labels = []
    for _, r in view_df.iterrows():
        original = str(r.get("original_name", ""))
        sample_id = str(r.get("sample_id", ""))
        ore_class = str(r.get("model_class", r.get("final_class", "")))
        talc = _format_pct(r.get("talc_percent_model", ""))
        short_name = original if len(original) <= 72 else "..." + original[-69:]
        labels.append(f"{sample_id} · {ore_class} · тальк {talc} · {short_name}")
    return labels


def _as_display_pil(image) -> Image.Image:
    """Convert numpy/PIL image to RGB PIL for lightweight browser preview."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    arr = np.asarray(image)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def _pil_to_data_url(pil_img: Image.Image) -> str:
    """Encode an already prepared PIL image as PNG data URL."""
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _resize_for_visual_panel(image, max_long_side: int = 1500) -> tuple[Image.Image, tuple[int, int]]:
    """Resize any image to the exact size used by visual tabs.

    Source image, mask overlay and heatmap must pass through the same function,
    otherwise Streamlit tabs look as if masks have a different scale.
    """
    pil = _as_display_pil(image)
    w, h = pil.size
    scale = min(1.0, float(max_long_side) / max(w, h))
    if scale < 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        pil = pil.resize(new_size, Image.Resampling.LANCZOS)
        w, h = new_size
    return pil, (w, h)


def _image_to_data_url(image, max_long_side: int = 1500) -> str:
    """Make a resized data-url so large panoramas do not overload reruns."""
    pil, _ = _resize_for_visual_panel(image, max_long_side=max_long_side)
    return _pil_to_data_url(pil)


def _show_fit_image(image, caption: str = "", max_long_side: int = 1500, max_height_vh: int = 78) -> None:
    """Show a plain image with the same visual sizing as overlay tabs."""
    pil, (w, h) = _resize_for_visual_panel(image, max_long_side=max_long_side)
    data_url = _pil_to_data_url(pil)
    safe_caption = html.escape(caption)
    caption_html = f'<div class="fit-image-caption">{safe_caption}</div>' if caption else ""
    st.markdown(
        f'<div class="fit-image-card" style="max-width:{w}px; margin-left:auto; margin-right:auto;">'
        f'<img src="{data_url}" style="display:block; width:100%; height:auto; max-height:none; object-fit:contain; border-radius:10px;" />'
        f'{caption_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _paired_overlay_data_urls(base_image, overlay_image, max_long_side: int = 1500) -> tuple[str, str, tuple[int, int]]:
    """Prepare same-size base/overlay images for a browser-side opacity slider."""
    base_raw = _as_display_pil(base_image)
    overlay_raw = _as_display_pil(overlay_image)
    if overlay_raw.size != base_raw.size:
        overlay_raw = overlay_raw.resize(base_raw.size, Image.Resampling.NEAREST)

    # Resize both through the same scale so original/mask/heatmap tabs keep identical size.
    w0, h0 = base_raw.size
    scale = min(1.0, float(max_long_side) / max(w0, h0))
    if scale < 1.0:
        new_size = (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale))))
        base = base_raw.resize(new_size, Image.Resampling.LANCZOS)
        overlay = overlay_raw.resize(new_size, Image.Resampling.NEAREST)
    else:
        base = base_raw
        overlay = overlay_raw
    w, h = base.size
    return _pil_to_data_url(base), _pil_to_data_url(overlay), (w, h)


def _show_browser_overlay(
    base_image,
    overlay_image,
    *,
    caption: str = "",
    slider_label: str = "Прозрачность слоя",
    alpha_default: float = 0.45,
    max_long_side: int = 1500,
    height: int | None = None,
) -> None:
    """
    Show image + overlay with an opacity slider inside the browser.

    Unlike st.slider, this does not rerun Streamlit, so mask/heatmap opacity changes instantly.
    The iframe height is computed from the prepared image height to avoid visual cropping.
    """
    base_url, overlay_url, (w, h) = _paired_overlay_data_urls(base_image, overlay_image, max_long_side=max_long_side)
    alpha_default = max(0.0, min(0.95, float(alpha_default)))
    safe_caption = html.escape(caption)
    safe_label = html.escape(slider_label)
    iframe_height = int(height) if height is not None else int(h + 92)
    iframe_height = max(260, min(max(iframe_height, h + 92), 1120))
    html_code = f"""
    <div style="font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: rgba(250,250,250,.88); width: 100%;">
      <div style="display:flex; align-items:center; gap:14px; margin: 0 0 10px 0;">
        <div style="font-weight:700; min-width: 160px;">{safe_label}</div>
        <input id="alpha" type="range" min="0" max="0.95" step="0.01" value="{alpha_default:.2f}"
               style="flex:1; accent-color:#ff4b5c; cursor:pointer;">
        <div id="alpha_value" style="width:50px; text-align:right; color:#ff6b78; font-weight:700;">{alpha_default:.2f}</div>
      </div>
      <div style="position:relative; width:100%; max-width:{w}px; margin:0 auto; border-radius:10px; overflow:hidden; background:#111827; line-height:0;">
        <img src="{base_url}" style="display:block; width:100%; height:auto; user-select:none; -webkit-user-drag:none;">
        <img id="overlay" src="{overlay_url}" style="position:absolute; inset:0; width:100%; height:100%; opacity:{alpha_default:.2f}; user-select:none; -webkit-user-drag:none; pointer-events:none;">
      </div>
      <div style="margin-top:8px; text-align:center; color:rgba(250,250,250,.66); font-size:13px;">{safe_caption}</div>
      <script>
        const slider = document.getElementById('alpha');
        const overlay = document.getElementById('overlay');
        const value = document.getElementById('alpha_value');
        slider.addEventListener('input', () => {{
          overlay.style.opacity = slider.value;
          value.textContent = Number(slider.value).toFixed(2);
        }});
      </script>
    </div>
    """
    components.html(html_code, height=iframe_height, scrolling=False)


def _run_analysis(
    uploaded_file,
    model,
    runtime_info,
    talc_threshold: float,
    batch_id: str = "",
    metadata: dict | None = None,
    *,
    collect_timing: bool = False,
    analysis_options: dict | None = None,
) -> dict:
    started = time.perf_counter()
    debug_timings: dict[str, float] = {}
    analysis_options = _normalize_analysis_options(analysis_options)

    step_started = time.perf_counter()
    raw_path, sample_id = persist_uploaded_file(uploaded_file, STORE.images)
    if collect_timing:
        debug_timings["persist_upload_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    image_rgb = read_image_any(raw_path)
    if collect_timing:
        debug_timings["read_image_sec"] = round(time.perf_counter() - step_started, 4)

    step_started = time.perf_counter()
    result = analyze_image(
        image_rgb=image_rgb,
        model=model,
        runtime_info=runtime_info,
        talc_threshold=talc_threshold,
        analysis_options=analysis_options,
    )
    if collect_timing:
        debug_timings["analyze_image_sec"] = round(time.perf_counter() - step_started, 4)
        internal_timings = result.get("debug_timings_sec")
        if isinstance(internal_timings, dict):
            debug_timings.update(internal_timings)

    # Keep the old semantics for CSV: processing_time_sec is time until inference is ready.
    result["processing_time_sec"] = round(time.perf_counter() - started, 3)

    step_started = time.perf_counter()
    metadata_for_row = dict(metadata or {})
    metadata_for_row["analysis_options"] = analysis_options
    row = STORE.save_analysis(
        sample_id=sample_id,
        original_name=uploaded_file.name,
        image_rgb=image_rgb,
        result=result,
        batch_id=batch_id,
        metadata=metadata_for_row,
        output_options=analysis_options,
    )
    if collect_timing:
        debug_timings["save_outputs_sec"] = round(time.perf_counter() - step_started, 4)
        debug_timings["total_sec"] = round(time.perf_counter() - started, 4)
        result["debug_timings_sec"] = debug_timings

    st.session_state["last_sample_id"] = sample_id
    return {"row": row, "image_rgb": image_rgb, "result": result}


with st.sidebar:
    st.header("⚙️ Настройки")
    default_model = PROJECT_ROOT / "models"
    model_path_text = st.text_input("Путь к папке models или весам .pt/.pth/.joblib", str(default_model))
    model_path = Path(model_path_text).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if model_path.exists():
        st.success("Путь к модели найден")
    else:
        st.warning("Путь к модели не найден — будет CV-baseline")

    device = st.selectbox("Устройство", ["auto", "cuda", "cpu"], index=0)
    talc_threshold = st.slider("Порог оталькованности, %", 5.0, 20.0, 10.0, 0.5)
    st.divider()
    st.header("🧑‍🔬 Экспертный контур")
    st.caption("Подтверждение класса, ручная коррекция маски и сохранение новой разметки для active learning.")


st.title("🔬 Скажи мне, кто твой шлиф")
st.caption("Интерпретируемая обёртка: анализ OM-изображений, зона талька, классификация руды и Human-in-the-loop доразметка")

model, runtime_info = cached_model(str(model_path), device)
with st.expander("Статус модели", expanded=False):
    st.write(runtime_info.get("message", ""))
    st.json({"mode": runtime_info.get("mode"), "device": runtime_info.get("device"), "model_path": str(model_path)})


tab_analyze, tab_batch, tab_review, tab_logs, tab_help = st.tabs([
    "1 · Анализ одного шлифа",
    "2 · Пакетная обработка",
    "3 · Экспертная проверка и доразметка",
    "4 · Журнал и экспорт",
    "5 · Инструкция",
])


with tab_analyze:
    st.subheader("Анализ одного изображения")
    single_analysis_options = _analysis_options_panel("single", expanded=True)
    uploaded = st.file_uploader("Загрузите PNG/JPG/TIFF изображение", type=["png", "jpg", "jpeg", "tif", "tiff"], key="single_upload")

    if uploaded is None:
        st.info("Загрузите изображение, чтобы запустить анализ.")
    else:
        if st.button("Запустить анализ", type="primary", key="single_run"):
            with st.spinner("Анализируем изображение..."):
                payload = _run_analysis(
                    uploaded,
                    model,
                    runtime_info,
                    talc_threshold,
                    collect_timing=True,
                    analysis_options=single_analysis_options,
                )
            st.success("Анализ завершён. Результат сохранён в журнал и доступен во вкладке экспертной проверки.")
            st.session_state["single_result_payload"] = payload

        payload = st.session_state.get("single_result_payload")
        if payload:
            image_rgb = payload["image_rgb"]
            result = payload["result"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Итоговый класс", result["ore_class"])
            c2.metric("Доля талька", f"{result['talc_percent']:.2f}%")
            c3.metric("Уверенность", f"{result.get('mean_confidence', 0):.2f}")
            c4.metric("Проверка", "нужна" if result.get("needs_review") else "не нужна")

            if result.get("needs_review"):
                st.warning(f"Изображение попало в очередь проверки: {result.get('review_reason')}")
            if result.get("adapter_warnings"):
                st.info("Предупреждения адаптера: " + "; ".join(result.get("adapter_warnings", [])))

            st.markdown("### Объяснение")
            st.write(result["conclusion"])

            st.markdown("### Таблица геологических метрик")
            st.dataframe(_result_metrics_df(result), use_container_width=True, hide_index=True)

            timing_df = _debug_timing_df(result)
            if not timing_df.empty:
                with st.expander("Отладка времени одиночного анализа", expanded=True):
                    st.dataframe(timing_df, use_container_width=True, hide_index=True)
                    st.caption(
                        "Эта таблица показывается только для одиночного запуска. "
                        "В пакетной обработке тайминги по этапам не собираются, чтобы не захламлять интерфейс."
                    )

            probs_df = _class_probs_df(result)
            if not probs_df.empty:
                with st.expander("Вероятности CNN-классификатора", expanded=False):
                    st.dataframe(probs_df, use_container_width=True, hide_index=True)

            view_tabs = st.tabs(["Исходник", "Итоговая маска", "Сравнение сегментаций", "Тальк", "Уверенность", "Grad-CAM / talc-score"])
            with view_tabs[0]:
                _show_fit_image(image_rgb, caption="Исходное изображение")
            with view_tabs[1]:
                _show_browser_overlay(
                    image_rgb,
                    make_class_overlay_pil(image_rgb, result["class_mask"], alpha=1.0),
                    caption="Итоговая маска: зелёный = обычные · красный = тонкие · синий = тальк",
                    slider_label="Прозрачность маски",
                    alpha_default=0.45,
                    height=760,
                )
            with view_tabs[2]:
                col_h, col_s = st.columns(2)
                with col_h:
                    if result.get("hybrid_class_mask") is not None:
                        _show_browser_overlay(
                            image_rgb,
                            make_class_overlay_pil(image_rgb, result["hybrid_class_mask"], alpha=1.0),
                            caption="CV/CNN-гибридная сегментация",
                            slider_label="Прозрачность маски",
                            alpha_default=0.45,
                            max_long_side=900,
                            height=560,
                        )
                    else:
                        st.info("Гибридная маска недоступна.")
                with col_s:
                    if result.get("superpixel_native_mask") is not None:
                        _show_browser_overlay(
                            image_rgb,
                            make_superpixel_overlay_pil(image_rgb, result["superpixel_native_mask"], alpha=1.0),
                            caption="Superpixel-модель: синий тальк, зелёный плотные сульфиды, красный тонкие, жёлтый магнетит/замещение",
                            slider_label="Прозрачность маски",
                            alpha_default=0.45,
                            max_long_side=900,
                            height=560,
                        )
                    else:
                        st.info("Superpixel-сегментация недоступна: положите model.joblib / superpixel_model.joblib в папку models.")
            with view_tabs[3]:
                _show_browser_overlay(
                    image_rgb,
                    make_overlay_pil(image_rgb, result["talc_mask"], alpha=1.0),
                    caption="Синяя зона = тальк / оталькованные участки",
                    slider_label="Прозрачность талька",
                    alpha_default=0.45,
                    height=760,
                )
            with view_tabs[4]:
                if result.get("confidence_map") is not None:
                    _show_browser_overlay(
                        image_rgb,
                        make_heatmap_pil(image_rgb, result.get("confidence_map"), alpha=1.0),
                        caption="Карта уверенности / надежности автоматического анализа",
                        slider_label="Прозрачность тепловой карты",
                        alpha_default=0.45,
                        height=760,
                    )
                else:
                    st.info("Карта уверенности была выключена в настройках запуска. Средняя уверенность в таблице посчитана облегчённо.")
            with view_tabs[5]:
                col_a, col_b = st.columns(2)
                with col_a:
                    if result.get("gradcam") is not None:
                        _show_browser_overlay(
                            image_rgb,
                            make_heatmap_pil(image_rgb, result.get("gradcam"), alpha=1.0),
                            caption="Grad-CAM: куда смотрел CNN",
                            slider_label="Прозрачность Grad-CAM",
                            alpha_default=0.45,
                            max_long_side=900,
                            height=560,
                        )
                    else:
                        st.info("Grad-CAM доступен, когда подключены веса CNN-классификатора.")
                with col_b:
                    if result.get("talc_score") is not None:
                        _show_browser_overlay(
                            image_rgb,
                            make_heatmap_pil(image_rgb, result.get("talc_score"), alpha=1.0),
                            caption="Talc-score: карта вероятных тальковых зон",
                            slider_label="Прозрачность talc-score",
                            alpha_default=0.45,
                            max_long_side=900,
                            height=560,
                        )
                    else:
                        st.info("Talc-score недоступен для текущего режима.")

            st.markdown("### Отчёт по текущему образцу")
            report_col, download_col = st.columns([1, 2])
            with report_col:
                if st.button("Сформировать PDF по образцу", key="single_pdf_report"):
                    sample_report_path = _make_single_pdf_report(payload["row"])
                    st.session_state["last_single_pdf_report_path"] = str(sample_report_path)
                    st.success(f"PDF-отчёт сформирован: {sample_report_path.name}")
            with download_col:
                single_pdf_path = st.session_state.get("last_single_pdf_report_path")
                if single_pdf_path and Path(single_pdf_path).exists():
                    st.download_button(
                        "Скачать PDF по текущему образцу",
                        data=Path(single_pdf_path).read_bytes(),
                        file_name=Path(single_pdf_path).name,
                        mime="application/pdf",
                        key="single_pdf_download",
                    )
                else:
                    st.caption("PDF появится здесь после формирования отчёта.")

            st.markdown("### GIS-экспорт по текущему образцу")
            gis_col, gis_download_col = st.columns([1, 2])
            with gis_col:
                if st.button("Сформировать GeoJSON", key="single_gis_export"):
                    try:
                        gis_path = _make_single_gis_export(payload["row"])
                        st.session_state["last_single_gis_path"] = str(gis_path)
                        st.success(f"GeoJSON сформирован: {gis_path.name}")
                    except Exception as exc:
                        st.error(f"Не удалось сформировать GeoJSON: {type(exc).__name__}: {exc}")
            with gis_download_col:
                single_gis_path = st.session_state.get("last_single_gis_path")
                if single_gis_path and Path(single_gis_path).exists():
                    st.download_button(
                        "Скачать GeoJSON по текущему образцу",
                        data=Path(single_gis_path).read_bytes(),
                        file_name=Path(single_gis_path).name,
                        mime="application/geo+json",
                        key="single_gis_download",
                    )
                else:
                    st.caption("GeoJSON появится здесь после формирования. Координаты локальные: пиксели или мкм, если указан масштаб мкм/px.")

            with st.expander("Служебные значения"):
                st.json({
                    "status": result.get("status"),
                    "runtime_mode": result.get("runtime_mode"),
                    "segmentation_source": result.get("segmentation_source"),
                    "model_pred_class": result.get("model_pred_class"),
                    "class_probs": result.get("class_probs"),
                    "talc_pixels": int(result["talc_pixels"]),
                    "valid_area_pixels": int(result["valid_area_pixels"]),
                    "talc_percent": round(float(result["talc_percent"]), 4),
                    "mean_confidence": round(float(result.get("mean_confidence", 0)), 4),
                    "sulfide_percent": round(float(result.get("sulfide_percent", 0)), 4),
                    "ordinary_share_among_sulfides": round(float(result.get("ordinary_share_among_sulfides", 0)), 4),
                    "thin_share_among_sulfides": round(float(result.get("thin_share_among_sulfides", 0)), 4),
                    "review_reason": result.get("review_reason"),
                    "adapter_warnings": result.get("adapter_warnings"),
                    "superpixel_metrics": result.get("superpixel_metrics"),
                })


with tab_batch:
    st.subheader("Пакетная обработка")
    st.write("Загрузите несколько изображений. Система проанализирует их и автоматически сформирует очередь экспертной проверки.")

    with st.expander("Метаданные партии", expanded=True):
        c1, c2, c3 = st.columns(3)
        batch_name = c1.text_input("Название партии", value="")
        deposit_name = c2.text_input("Месторождение", value="")
        ore_type = c3.text_input(
            "Тип руды / образца (необязательно)",
            value="",
            help="Это только метаданные для отчёта: например, ожидаемый тип образца, партия лаборатории или геологическая категория. Поле не влияет на прогноз модели.",
            placeholder="например: медно-никелевая / опытная партия / неизвестно",
        )
        c4, c5, c6 = st.columns(3)
        microns_per_pixel = c4.text_input("Масштаб, мкм/px", value="")
        microscope = c5.text_input("Микроскоп / камера", value="")
        operator = c6.text_input("Оператор", value="")
        batch_comment = st.text_area("Комментарий к партии", value="", height=70)

    batch_analysis_options = _analysis_options_panel("batch", expanded=True)

    batch_inputs = st.file_uploader(
        "Изображения или ZIP-архивы для пакетного анализа",
        type=["png", "jpg", "jpeg", "tif", "tiff", "zip"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if batch_inputs:
        batch_files, zip_warnings = _expand_batch_uploads(batch_inputs)
        st.caption(f"Загружено объектов: {len(batch_inputs)} · найдено изображений для анализа: {len(batch_files)}")
        if zip_warnings:
            with st.expander("Предупреждения по ZIP/файлам", expanded=False):
                for msg in zip_warnings:
                    st.warning(msg)

        if batch_files:
            with st.expander("Предпросмотр изображений перед запуском", expanded=True):
                st.caption(
                    "Проверьте, что в список попала нужная партия. "
                    "К анализу будут отправлены только изображения из таблицы ниже."
                )
                st.dataframe(_batch_preview_df(batch_files), use_container_width=True, hide_index=True)
                total_bytes = sum(_uploaded_size(f) for f in batch_files)
                st.caption(f"Оценочный суммарный размер изображений: {_human_size(total_bytes)}")

        if not batch_files:
            st.info("В загруженных файлах не найдено изображений поддерживаемых форматов: PNG/JPG/TIFF.")

        if batch_files and st.button("Запустить пакетный анализ", type="primary", key="batch_run"):
            batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
            metadata = {
                "batch_name": batch_name,
                "deposit_name": deposit_name,
                "ore_type": ore_type,
                "microns_per_pixel": microns_per_pixel,
                "microscope": microscope,
                "operator": operator,
                "batch_comment": batch_comment,
            }
            progress = st.progress(0)
            live_slot = st.empty()
            rows = []
            ok_count = 0
            error_count = 0
            review_count = 0

            for i, file in enumerate(batch_files, start=1):
                started = time.perf_counter()
                try:
                    with st.spinner(f"Анализируем {file.name} ({i}/{len(batch_files)})..."):
                        payload = _run_analysis(
                            file,
                            model,
                            runtime_info,
                            talc_threshold,
                            batch_id=batch_id,
                            metadata=metadata,
                            analysis_options=batch_analysis_options,
                        )
                    row = payload["row"]
                    rows.append(row)
                    ok_count += 1
                    if _bool(row.get("needs_review")):
                        review_count += 1
                except Exception as exc:
                    error_count += 1
                    sample_id = f"{safe_stem(file.name)}_error_{datetime.now().strftime('%H%M%S')}_{i}"
                    err = f"{type(exc).__name__}: {exc}"
                    row = STORE.save_error(
                        sample_id=sample_id,
                        original_name=file.name,
                        error_message=err,
                        batch_id=batch_id,
                        processing_time_sec=round(time.perf_counter() - started, 3),
                        metadata=metadata,
                    )
                    rows.append(row)
                    st.error(f"Не удалось обработать {file.name}: {err}")

                progress.progress(i / len(batch_files))
                live_slot.dataframe(_analysis_table(rows), use_container_width=True, hide_index=True)

            st.session_state["last_batch_id"] = batch_id
            st.success(
                f"Пакет {batch_id} завершён: успешно {ok_count}, ошибок {error_count}, "
                f"требуют проверки {review_count}."
            )

    rows = STORE.read_rows()
    if rows:
        df = _analysis_table(rows)
        st.markdown("### Результаты")
        batch_options = ["Все партии"]
        if "batch_id" in df.columns:
            batch_options += sorted([b for b in df["batch_id"].dropna().unique().tolist() if str(b).strip()], reverse=True)
        default_batch = st.session_state.get("last_batch_id")
        default_idx = batch_options.index(default_batch) if default_batch in batch_options else 0
        selected_batch = st.selectbox("Фильтр по партии", batch_options, index=default_idx, key="batch_filter")
        view_df = df.copy()
        if selected_batch != "Все партии" and "batch_id" in view_df.columns:
            view_df = view_df[view_df["batch_id"] == selected_batch]

        if not view_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Всего", len(view_df))
            c2.metric("Успешно", int((view_df.get("status", "") == "success").sum()) if "status" in view_df else "—")
            c3.metric("Ошибки", int((view_df.get("status", "") == "error").sum()) if "status" in view_df else "—")
            c4.metric("В проверку", int(view_df["needs_review"].astype(str).str.lower().isin(["true", "1", "yes", "да"]).sum()) if "needs_review" in view_df else "—")

        st.dataframe(view_df, use_container_width=True, hide_index=True)

        if not view_df.empty:
            st.markdown("### Просмотр результатов партии")
            success_view_df = view_df[view_df.get("status", "").astype(str) != "error"].copy() if "status" in view_df.columns else view_df.copy()
            if success_view_df.empty:
                st.info("В выбранной партии пока нет успешно обработанных изображений для визуального просмотра.")
            else:
                viewer_labels = _batch_viewer_labels(success_view_df)
                selected_label = st.selectbox(
                    "Выберите изображение для просмотра",
                    viewer_labels,
                    index=0,
                    key=f"batch_result_viewer_{selected_batch}",
                )
                selected_pos = viewer_labels.index(selected_label)
                selected_row = success_view_df.iloc[selected_pos].to_dict()

                metric_cols = st.columns(5)
                metric_cols[0].metric("Класс", str(selected_row.get("model_class") or selected_row.get("final_class") or "—"))
                metric_cols[1].metric("Тальк", _format_pct(selected_row.get("talc_percent_model")))
                metric_cols[2].metric("Сульфиды", _format_pct(selected_row.get("sulfide_percent")))
                metric_cols[3].metric("Уверенность", f"{float(selected_row.get('mean_confidence') or 0):.2f}")
                metric_cols[4].metric("Статус", str(selected_row.get("review_status") or "—"))

                if _bool(selected_row.get("needs_review")):
                    st.warning(f"Требует проверки: {selected_row.get('review_reason', 'причина не указана')}")

                mask_label_to_key = {
                    "Итоговая маска": "final",
                    "Superpixel-модель": "superpixel",
                    "CV/CNN-гибрид": "hybrid",
                    "Только тальк": "talc",
                }
                mask_label = st.radio(
                    "Какая сегментация показана в средней колонке",
                    list(mask_label_to_key.keys()),
                    horizontal=True,
                    key=f"batch_mask_kind_{selected_batch}",
                )

                heatmap_label_to_key = {
                    "Карта уверенности": "confidence",
                    "Уверенность superpixel": "superpixel_confidence",
                    "Grad-CAM": "gradcam",
                    "Talc-score": "talc_score",
                }
                heatmap_label = st.radio(
                    "Тепловая карта",
                    list(heatmap_label_to_key.keys()),
                    horizontal=True,
                    key=f"batch_heatmap_kind_{selected_batch}",
                )

                try:
                    preview = _load_batch_preview_payload(
                        selected_row,
                        heatmap_kind=heatmap_label_to_key[heatmap_label],
                        mask_kind=mask_label_to_key[mask_label],
                    )
                    col_img, col_mask, col_heat = st.columns(3)
                    with col_img:
                        st.caption("Исходное изображение")
                        _show_fit_image(preview["image_rgb"], caption=str(selected_row.get("original_name", "")), max_long_side=900, max_height_vh=46)
                    with col_mask:
                        st.caption("Маска")
                        if preview["mask_preview"] is not None:
                            _show_browser_overlay(
                                preview["image_rgb"],
                                preview["mask_preview"],
                                caption=preview["mask_caption"],
                                slider_label="Прозрачность маски",
                                alpha_default=0.45,
                                max_long_side=850,
                                height=500,
                            )
                        else:
                            st.info("Для выбранного образца не найден сохранённый overlay/маска.")
                    with col_heat:
                        st.caption(heatmap_label)
                        if preview["heatmap_preview"] is not None:
                            _show_browser_overlay(
                                preview["image_rgb"],
                                preview["heatmap_preview"],
                                caption=preview["heatmap_caption"],
                                slider_label="Прозрачность тепловой карты",
                                alpha_default=0.45,
                                max_long_side=850,
                                height=500,
                            )
                        else:
                            st.info("Эта карта недоступна для выбранного образца или текущего режима модели.")
                except Exception as exc:
                    st.error(f"Не удалось открыть визуальные артефакты выбранного образца: {type(exc).__name__}: {exc}")

        if not view_df.empty:
            st.markdown("### Экспорт выбранной партии")
            export_batch_id = selected_batch if selected_batch != "Все партии" else None
            pdf_col, zip_col, gis_col = st.columns(3)
            with pdf_col:
                if st.button("Сформировать PDF-отчёт", key="batch_pdf_report"):
                    report_path = _make_pdf_report(
                        rows,
                        batch_id=export_batch_id,
                        title="Отчет по партии шлифов" if export_batch_id else "Сводный отчет по анализу шлифов",
                    )
                    st.session_state["last_pdf_report_path"] = str(report_path)
                    st.success(f"PDF-отчёт сформирован: {report_path.name}")
                pdf_path_value = st.session_state.get("last_pdf_report_path")
                if pdf_path_value and Path(pdf_path_value).exists():
                    st.download_button(
                        "Скачать PDF-отчёт",
                        data=Path(pdf_path_value).read_bytes(),
                        file_name=Path(pdf_path_value).name,
                        mime="application/pdf",
                        key="batch_pdf_download",
                    )
            with zip_col:
                if st.button("Собрать ZIP выбранных outputs", key="batch_zip_build"):
                    zip_path = _make_outputs_zip(batch_id=export_batch_id)
                    st.session_state["last_batch_zip_path"] = str(zip_path)
                    st.success(f"ZIP собран: {zip_path.name}")
                zip_path_value = st.session_state.get("last_batch_zip_path")
                if zip_path_value and Path(zip_path_value).exists():
                    st.download_button(
                        "Скачать ZIP выбранных outputs",
                        data=Path(zip_path_value).read_bytes(),
                        file_name=Path(zip_path_value).name,
                        mime="application/zip",
                        key="batch_zip_download",
                    )
            with gis_col:
                if st.button("Собрать GIS GeoJSON", key="batch_gis_build"):
                    try:
                        gis_zip_path = _make_gis_export_zip(rows, batch_id=export_batch_id)
                        st.session_state["last_batch_gis_zip_path"] = str(gis_zip_path)
                        st.success(f"GIS ZIP собран: {gis_zip_path.name}")
                    except Exception as exc:
                        st.error(f"Не удалось собрать GIS ZIP: {type(exc).__name__}: {exc}")
                gis_zip_value = st.session_state.get("last_batch_gis_zip_path")
                if gis_zip_value and Path(gis_zip_value).exists():
                    st.download_button(
                        "Скачать GIS ZIP",
                        data=Path(gis_zip_value).read_bytes(),
                        file_name=Path(gis_zip_value).name,
                        mime="application/zip",
                        key="batch_gis_download",
                    )

        if "model_class" in view_df and not view_df.empty:
            with st.expander("Сводка по классам", expanded=False):
                ok_for_summary = view_df[view_df["status"].astype(str) != "error"] if "status" in view_df else view_df
                cls_summary = ok_for_summary.groupby("model_class", dropna=False).size().reset_index(name="Количество")
                st.dataframe(cls_summary, use_container_width=True, hide_index=True)

        queue_df = view_df[view_df["needs_review"].astype(str).str.lower().isin(["true", "1", "yes", "да"])] if "needs_review" in view_df else pd.DataFrame()
        st.markdown("### Очередь экспертной проверки")
        if queue_df.empty:
            st.info("Пока нет изображений, автоматически отправленных в очередь.")
        else:
            st.dataframe(queue_df, use_container_width=True, hide_index=True)
            st.caption("Во вкладке экспертной проверки можно открыть любой sample_id из этой таблицы.")
    else:
        st.info("Пока журнал пуст. Запустите анализ одного изображения или пакетную обработку.")


with tab_review:
    st.subheader("Human-in-the-loop: экспертная проверка и ручная доразметка")
    rows = STORE.read_rows()
    if not rows:
        st.info("Сначала запустите анализ хотя бы одного изображения.")
    else:
        df_full = pd.DataFrame(rows)
        default_id = st.session_state.get("last_sample_id")
        sample_ids = df_full["sample_id"].tolist()
        default_index = sample_ids.index(default_id) if default_id in sample_ids else 0

        only_queue = st.checkbox("Показывать только очередь проверки", value=False)
        display_df = df_full.copy()
        if only_queue and "needs_review" in display_df.columns:
            display_df = display_df[display_df["needs_review"].astype(str).str.lower().isin(["true", "1", "yes", "да"])]
        if display_df.empty:
            st.info("Очередь проверки пуста.")
        else:
            selected_id = st.selectbox(
                "Выберите изображение",
                display_df["sample_id"].tolist(),
                index=0 if only_queue else default_index,
                key="review_sample_select",
            )
            row = df_full[df_full["sample_id"] == selected_id].iloc[0].to_dict()

            expected_paths = STORE.result_paths(selected_id)
            image_path = _resolve_existing_artifact(row.get("image_path", ""), expected_paths["image_path"])
            model_mask_path = _resolve_existing_artifact(row.get("model_mask_path", ""), expected_paths["model_mask_path"])
            class_mask_path = _resolve_existing_artifact(row.get("class_mask_path", ""), expected_paths.get("class_mask_path"))

            if image_path is None or model_mask_path is None:
                st.error(
                    "Не удалось найти исходное изображение или маску модели. "
                    "Проверь, что папка outputs лежит рядом с app.py, либо перезапусти анализ этого изображения."
                )
                st.stop()

            image_rgb = read_image_any(image_path)
            original_shape = image_rgb.shape[:2]
            model_mask = _resize_mask_to_shape(load_mask(model_mask_path), original_shape).astype(bool)
            if class_mask_path is not None:
                model_class_mask = _resize_mask_to_shape(load_class_mask(class_mask_path), original_shape).astype(np.uint8)
            else:
                # Fallback для старых результатов: строим class_mask только из талька.
                model_class_mask = np.zeros(original_shape, dtype=np.uint8)
                model_class_mask[model_mask] = 3

            existing_expert = _resolve_existing_artifact(
                row.get("expert_mask_path", ""),
                expected_paths["expert_mask_path"],
            )
            existing_expert_class = _resolve_existing_artifact(
                row.get("expert_class_mask_path", ""),
                expected_paths.get("expert_class_mask_path"),
            )
            has_expert_mask = bool(existing_expert is not None and existing_expert.is_file())
            has_expert_class_mask = bool(existing_expert_class is not None and existing_expert_class.is_file())
            expert_base_mask = _resize_mask_to_shape(load_mask(existing_expert), original_shape).astype(bool) if has_expert_mask else model_mask
            expert_base_class_mask = (
                _resize_mask_to_shape(load_class_mask(existing_expert_class), original_shape).astype(np.uint8)
                if has_expert_class_mask else model_class_mask
            )

            st.markdown("### Быстрая экспертная проверка и редактор маски")
            st.caption(
                "В редакторе правится только мультиклассовая маска. "
                "Выберите цвет кисти: обычные сульфиды, тонкие сульфиды, тальк; ластик стирает в фон. "
                "Изменение полей не перезапускает Streamlit."
            )

            preview_key = f"frontend_preview_mask_{selected_id}"
            preview_meta_key = f"frontend_preview_meta_{selected_id}"
            review_payload_key = f"frontend_review_payload_{selected_id}"
            processed_key = f"processed_editor_submission_{selected_id}"

            # Для редактора передаём более крупную версию, чем раньше: теперь есть zoom/pan,
            # поэтому эксперт может приблизить участок без постоянных Streamlit-rerun. Размер preview увеличен, чтобы zoom 500% был полезнее.
            disp_img, disp_model_mask, _ = resize_for_display(image_rgb, model_mask, max_side=2400)
            _, disp_expert_mask, _ = resize_for_display(image_rgb, expert_base_mask, max_side=2400)
            _, disp_model_class_mask, _ = resize_for_display(image_rgb, model_class_mask, max_side=2400)
            _, disp_expert_class_mask, _ = resize_for_display(image_rgb, expert_base_class_mask, max_side=2400)

            # Safety net: all preview masks must have exactly the same H×W as disp_img.
            # Otherwise overlay indexing fails after changing preview size.
            target_editor_shape = disp_img.shape[:2]
            disp_model_mask = _resize_mask_to_shape(disp_model_mask, target_editor_shape)
            disp_expert_mask = _resize_mask_to_shape(disp_expert_mask, target_editor_shape)
            disp_model_class_mask = _resize_mask_to_shape(disp_model_class_mask, target_editor_shape)
            disp_expert_class_mask = _resize_mask_to_shape(disp_expert_class_mask, target_editor_shape)

            editor_model_background = make_overlay_pil(disp_img, disp_model_mask, alpha=0.30).convert("RGB")
            editor_expert_background = make_overlay_pil(disp_img, disp_expert_mask, alpha=0.30).convert("RGB") if has_expert_mask else None
            editor_model_class_background = make_class_overlay_pil(disp_img, disp_model_class_mask, alpha=0.35).convert("RGB")
            editor_expert_class_background = make_class_overlay_pil(disp_img, disp_expert_class_mask, alpha=0.35).convert("RGB") if has_expert_class_mask else None

            default_review_payload = {
                "class_decision": "Подтверждаю класс модели",
                "model_class": row.get("model_class", ""),
                "expert_class": row.get("model_class", ""),
                "talc_percent_model": row.get("talc_percent_model", ""),
                "mask_quality": "Маска корректная",
                "expert_comment": "",
                "mean_confidence": float(row.get("mean_confidence", 0) or 0),
                "needs_review": _bool(row.get("needs_review")),
                "review_reason": row.get("review_reason", ""),
                "final_class": row.get("final_class", row.get("model_class", "")),
            }
            review_context = st.session_state.get(review_payload_key, default_review_payload)
            review_context = {**default_review_payload, **dict(review_context or {})}

            editor_key = f"frontend_mask_editor_{selected_id}"
            component_requested_save = False

            with st.container(border=True):
                editor_value = frontend_mask_editor(
                    editor_model_background,
                    original_image=disp_img,
                    expert_background_image=editor_expert_background,
                    model_class_background_image=editor_model_class_background,
                    expert_class_background_image=editor_expert_class_background,
                    has_expert_mask=has_expert_mask,
                    has_expert_class_mask=has_expert_class_mask,
                    initial_base_choice="expert" if (has_expert_class_mask or has_expert_mask) else "model",
                    review_context=review_context,
                    ore_classes=ORE_CLASSES,
                    key=editor_key,
                    storage_key=editor_key,
                )

            final_expert_mask = st.session_state.get(preview_key, expert_base_mask)
            final_class_preview_key = f"frontend_preview_class_mask_{selected_id}"
            final_expert_class_mask = st.session_state.get(final_class_preview_key, expert_base_class_mask)

            if editor_value:
                submission_id = str(editor_value.get("submitted_at", ""))
                if submission_id and st.session_state.get(processed_key) != submission_id:
                    base_choice = str(editor_value.get("base_choice") or "model")
                    edit_layer = decode_editor_image(editor_value)
                    if edit_layer is not None:
                        # Редактор теперь всегда работает только с мультиклассовой class_mask.
                        base_class_mask = expert_base_class_mask if base_choice == "expert" and has_expert_class_mask else model_class_mask
                        final_expert_class_mask = apply_canvas_edits(
                            base_mask=base_class_mask,
                            canvas_image=edit_layer,
                            original_shape=image_rgb.shape[:2],
                            edit_mode="multiclass",
                        ).astype(np.uint8)
                        final_expert_mask = final_expert_class_mask == 3
                        st.session_state[final_class_preview_key] = final_expert_class_mask
                        st.session_state[preview_key] = final_expert_mask
                        st.session_state[preview_meta_key] = datetime.now().strftime("%H:%M:%S")

                    review_context = {
                        **review_context,
                        "class_decision": editor_value.get("class_decision", review_context.get("class_decision", "Подтверждаю класс модели")),
                        "expert_class": editor_value.get("expert_class", review_context.get("expert_class", row.get("model_class", ""))),
                        "mask_quality": editor_value.get("mask_quality", review_context.get("mask_quality", "Маска корректная")),
                        "expert_comment": editor_value.get("expert_comment", review_context.get("expert_comment", "")),
                        "base_choice": base_choice,
                    }
                    st.session_state[review_payload_key] = review_context
                    st.session_state[processed_key] = submission_id
                    component_requested_save = editor_value.get("action") == "save"

            if preview_key in st.session_state:
                preview_time = st.session_state.get(preview_meta_key, "")
                with st.expander(f"Предпросмотр финальной экспертной маски{f' · обновлено {preview_time}' if preview_time else ''}", expanded=False):
                    col_prev_class, col_prev_talc = st.columns(2)
                    with col_prev_class:
                        _show_fit_image(
                            make_class_overlay_pil(image_rgb, final_expert_class_mask, alpha=0.45),
                            caption="Экспертная мультиклассовая маска: фон / обычные / тонкие / тальк",
                        )
                    with col_prev_talc:
                        _show_fit_image(
                            make_overlay_pil(image_rgb, final_expert_mask, alpha=0.45),
                            caption="Бинарная маска талька, автоматически синхронизирована с классом 3",
                        )

            st.caption(
                "Для сохранения нажмите внутри браузерного редактора «Передать и сохранить». "
                "Кнопка «Применить правку» только пересчитает предпросмотр экспертной маски."
            )

            if component_requested_save:
                class_decision = review_context.get("class_decision", "Подтверждаю класс модели")
                expert_class = review_context.get("expert_class", row.get("model_class", ""))
                mask_quality = review_context.get("mask_quality", "Маска корректная")
                expert_comment = review_context.get("expert_comment", "")
                review_status = (
                    "подтверждено" if class_decision == "Подтверждаю класс модели" and mask_quality == "Маска корректная"
                    else "исправлено экспертом" if class_decision == "Класс неверный" or mask_quality != "Маска корректная"
                    else "требует проверки"
                )

                payload = {
                    "class_decision": class_decision,
                    "model_class": row.get("model_class", ""),
                    "expert_class": expert_class,
                    "talc_percent_model": row.get("talc_percent_model", ""),
                    "mask_quality": mask_quality,
                    "expert_comment": expert_comment,
                    "review_status": review_status,
                }
                saved_row = STORE.save_expert_review(
                    sample_id=selected_id,
                    image_rgb=image_rgb,
                    model_mask=model_mask,
                    expert_mask=final_expert_mask,
                    review_payload=payload,
                    model_class_mask=model_class_mask,
                    expert_class_mask=final_expert_class_mask,
                )
                st.success(f"Экспертная оценка сохранена. Финальный класс: {saved_row.get('final_class')}.")
                st.rerun()


with tab_logs:
    st.subheader("Журнал, файлы и экспорт")
    rows = STORE.read_rows()
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_bytes = STORE.csv_path.read_bytes() if STORE.csv_path.exists() else b""
        st.download_button(
            "Скачать CSV с результатами и экспертными оценками",
            data=csv_bytes,
            file_name="analysis_results.csv",
            mime="text/csv",
        )

        log_pdf_col, log_zip_col, log_gis_col = st.columns(3)
        with log_pdf_col:
            if st.button("Сформировать общий PDF-отчёт"):
                report_path = _make_pdf_report(rows, batch_id=None, title="Сводный отчет по анализу шлифов")
                st.session_state["last_logs_pdf_report_path"] = str(report_path)
                st.success(f"PDF-отчёт сформирован: {report_path.name}")
            log_pdf_path = st.session_state.get("last_logs_pdf_report_path")
            if log_pdf_path and Path(log_pdf_path).exists():
                st.download_button(
                    "Скачать общий PDF-отчёт",
                    data=Path(log_pdf_path).read_bytes(),
                    file_name=Path(log_pdf_path).name,
                    mime="application/pdf",
                    key="logs_pdf_download",
                )
        with log_zip_col:
            if st.button("Собрать ZIP со всеми outputs"):
                zip_path = _make_outputs_zip()
                st.session_state["last_logs_zip_path"] = str(zip_path)
                st.success(f"Архив собран: {zip_path.name}")
            log_zip_path = st.session_state.get("last_logs_zip_path")
            if log_zip_path and Path(log_zip_path).exists():
                st.download_button(
                    "Скачать ZIP outputs",
                    data=Path(log_zip_path).read_bytes(),
                    file_name=Path(log_zip_path).name,
                    mime="application/zip",
                    key="logs_zip_download",
                )
        with log_gis_col:
            if st.button("Собрать общий GIS ZIP"):
                try:
                    gis_zip_path = _make_gis_export_zip(rows, batch_id=None)
                    st.session_state["last_logs_gis_zip_path"] = str(gis_zip_path)
                    st.success(f"GIS ZIP собран: {gis_zip_path.name}")
                except Exception as exc:
                    st.error(f"Не удалось собрать GIS ZIP: {type(exc).__name__}: {exc}")
            log_gis_path = st.session_state.get("last_logs_gis_zip_path")
            if log_gis_path and Path(log_gis_path).exists():
                st.download_button(
                    "Скачать общий GIS ZIP",
                    data=Path(log_gis_path).read_bytes(),
                    file_name=Path(log_gis_path).name,
                    mime="application/zip",
                    key="logs_gis_download",
                )
    else:
        st.info("Журнал пока пуст.")

    with st.expander("Структура outputs"):
        st.code(
            """
outputs/
├── images/           # сохранённые входные изображения
├── masks/            # бинарные маски талька
├── class_masks/      # мультиклассовые маски 0/1/2/3
├── expert_masks/     # исправленные экспертные маски
├── overlays/         # overlay, class_overlay и карты расхождений
├── confidence_maps/  # карты уверенности
├── gradcams/         # Grad-CAM от CNN-классификатора
├── talc_scores/      # talc-score карты
├── reports/          # PDF-отчёты
├── gis/              # GeoJSON + GIS ZIP в локальных координатах изображения
├── reviews/          # CSV + JSON экспертных решений
└── active_learning/  # пары image + expert_mask для будущего дообучения
""".strip()
        )


with tab_help:
    st.subheader("Инструкция по работе с сайтом")
    st.markdown(
        """
### 1. Анализ одного шлифа
1. Откройте вкладку **«Анализ одного шлифа»**.
2. Загрузите изображение в формате PNG, JPG, JPEG, TIF или TIFF.
3. В блоке **«Что считать в этом запуске»** оставьте полный режим или выключите тяжёлые карты.
4. Нажмите **«Запустить анализ»**.
5. Проверьте итоговый класс, таблицу метрик и цветовую маску.
6. При необходимости скачайте **PDF по текущему образцу**.

### 2. Пакетная обработка
1. Откройте вкладку **«Пакетная обработка»**.
2. Заполните метаданные партии: месторождение, масштаб, микроскоп, оператор и комментарий.
3. Загрузите несколько изображений или **ZIP-архив** с папками изображений.
4. В блоке **«Что считать в этом запуске»** выберите, нужны ли superpixel, talc-score, Grad-CAM, карты уверенности и overlay.
5. Проверьте блок **«Предпросмотр изображений перед запуском»**.
6. Нажмите **«Запустить пакетный анализ»**.
7. После обработки скачайте CSV, ZIP outputs или PDF-отчёт по партии.

### 3. Как читать цвета маски
- 🟢 **Зелёный** — обычные срастания / сульфидные включения.
- 🔴 **Красный** — тонкие срастания / зоны замещения.
- 🔵 **Синий** — тальк / оталькованные участки.

### 4. Экспертная проверка
Во вкладке **«Экспертная проверка и доразметка»** можно открыть спорный образец, поправить маску кистью или ластиком, подтвердить или изменить класс и сохранить экспертную оценку. Исправления сохраняются в `outputs/active_learning` и могут быть использованы ML-командой для будущего дообучения.

### 5. Экспорт
- **CSV** — таблица всех запусков и экспертных решений.
- **PDF** — отчёт для одного образца, партии или всего журнала.
- **ZIP outputs** — исходники, маски, overlay, карты уверенности, Grad-CAM и active learning файлы.
        """
    )

    with st.expander("Что делать, если результат выглядит странно?", expanded=False):
        st.markdown(
            """
- Проверьте, не загружена ли слишком тёмная, пересвеченная или обрезанная картинка.
- Посмотрите вкладку **«Уверенность»** и, если доступно, **Grad-CAM / talc-score**.
- Отправьте образец в экспертную проверку и исправьте маску.
- Проверьте правило: если доля талька выше порога, по ТЗ класс должен быть **оталькованная руда**.
            """
        )
