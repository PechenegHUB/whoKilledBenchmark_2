from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

from PIL import Image as PILImage, ImageOps

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


CLASS_ORDER = ["рядовая руда", "труднообогатимая руда", "оталькованная руда"]
METRIC_LABELS = {
    "talc_percent_model": "Тальк, %",
    "sulfide_percent": "Сульфиды, %",
    "ordinary_share_among_sulfides": "Обычные среди сульфидов, %",
    "thin_share_among_sulfides": "Тонкие среди сульфидов, %",
    "mean_confidence": "Средняя уверенность",
    "processing_time_sec": "Время, с",
}


def _register_font() -> tuple[str, str]:
    """Return (regular_font_name, bold_font_name). Tries Cyrillic-capable fonts first."""
    candidates_regular = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    candidates_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    regular_path = next((p for p in candidates_regular if Path(p).exists()), None)
    bold_path = next((p for p in candidates_bold if Path(p).exists()), None)
    if regular_path:
        pdfmetrics.registerFont(TTFont("AppSans", regular_path))
        if bold_path:
            pdfmetrics.registerFont(TTFont("AppSans-Bold", bold_path))
        else:
            pdfmetrics.registerFont(TTFont("AppSans-Bold", regular_path))
        return "AppSans", "AppSans-Bold"
    # Last fallback: ASCII fonts. The PDF will still be created, but Cyrillic may render badly.
    return "Helvetica", "Helvetica-Bold"


def _styles():
    regular, bold = _register_font()
    base = getSampleStyleSheet()
    return {
        "normal": ParagraphStyle(
            "normal_ru",
            parent=base["Normal"],
            fontName=regular,
            fontSize=9.2,
            leading=12,
            textColor=colors.HexColor("#202020"),
        ),
        "small": ParagraphStyle(
            "small_ru",
            parent=base["Normal"],
            fontName=regular,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#404040"),
        ),
        "title": ParagraphStyle(
            "title_ru",
            parent=base["Title"],
            fontName=bold,
            fontSize=18,
            leading=22,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#16324F"),
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2_ru",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=12.5,
            leading=15,
            textColor=colors.HexColor("#16324F"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "center": ParagraphStyle(
            "center_ru",
            parent=base["Normal"],
            fontName=regular,
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
        ),
    }, regular, bold


def _as_text(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value)
    if text.strip() == "" or text.strip().lower() in {"nan", "none"}:
        return default
    return text


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        return f"{_as_float(value):.{digits}f}{suffix}"
    except Exception:
        return _as_text(value)


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "да"}


def _metadata_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        raw = row.get("metadata_json")
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {}


def _basename_any(path_value: Any) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    if "\\" in raw:
        return PureWindowsPath(raw).name
    return Path(raw).name


def _resolve_artifact(path_value: Any, project_root: str | Path, output_root: str | Path) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    project_root = Path(project_root)
    output_root = Path(output_root)
    raw_path = Path(raw).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([project_root / raw_path, output_root / raw_path])
    basename = _basename_any(raw)
    if basename:
        # Common artifact folders.
        for folder in ["images", "masks", "class_masks", "expert_masks", "overlays", "confidence_maps", "gradcams", "talc_scores"]:
            candidates.append(output_root / folder / basename)
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _img_flowable(path: Path | None, max_w: float, max_h: float) -> RLImage | Paragraph:
    styles, _, _ = _styles()
    if path is None or not path.exists():
        return Paragraph("Изображение не найдено", styles["small"])
    try:
        img = PILImage.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((int(max_w * 2.3), int(max_h * 2.3)), PILImage.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        buf.seek(0)
        w, h = img.size
        ratio = min(max_w / max(w, 1), max_h / max(h, 1))
        return RLImage(buf, width=w * ratio, height=h * ratio)
    except Exception as exc:
        return Paragraph(f"Не удалось открыть изображение: {type(exc).__name__}", styles["small"])


def _make_table(data: list[list[Any]], col_widths: list[float] | None = None, font: str = "AppSans", bold_font: str = "AppSans-Bold") -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9F1FA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324F")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C7D2DD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _footer(canvas, doc):
    canvas.saveState()
    font, _ = _register_font()
    canvas.setFont(font, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawRightString(A4[0] - 1.5 * cm, 0.9 * cm, f"Страница {doc.page}")
    canvas.drawString(1.5 * cm, 0.9 * cm, "Скажи мне, кто твой шлиф - автоматический отчет")
    canvas.restoreState()


def generate_pdf_report(
    rows: Iterable[dict[str, Any]],
    output_root: str | Path,
    project_root: str | Path,
    report_path: str | Path,
    *,
    title: str = "Отчет по анализу шлифов",
    batch_id: str | None = None,
    max_samples: int = 30,
) -> Path:
    """Create a compact PDF report for one image, one batch, or all logged rows."""
    rows = [dict(r) for r in rows]
    if batch_id:
        rows = [r for r in rows if str(r.get("batch_id", "")) == str(batch_id)]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    styles, regular, bold = _styles()
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.35 * cm,
        leftMargin=1.35 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
    )
    story: list[Any] = []

    story.append(Paragraph(title, styles["title"]))
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = _metadata_from_rows(rows)
    subtitle = f"Сформировано: {created_at}"
    if batch_id:
        subtitle += f" | batch_id: {batch_id}"
    story.append(Paragraph(subtitle, styles["normal"]))
    story.append(Spacer(1, 8))

    if meta:
        meta_rows = [["Параметр", "Значение"]]
        labels = {
            "batch_name": "Название партии",
            "deposit_name": "Месторождение",
            "ore_type": "Тип руды / партия",
            "microns_per_pixel": "Масштаб, мкм/px",
            "microscope": "Микроскоп / камера",
            "operator": "Оператор",
            "batch_comment": "Комментарий",
        }
        for key, label in labels.items():
            if _as_text(meta.get(key), ""):
                meta_rows.append([label, _as_text(meta.get(key))])
        if len(meta_rows) > 1:
            story.append(Paragraph("Метаданные партии", styles["h2"]))
            story.append(_make_table(meta_rows, [5 * cm, 11.5 * cm], regular, bold))
            story.append(Spacer(1, 8))

    total = len(rows)
    success = sum(1 for r in rows if str(r.get("status", "success")) == "success")
    errors = sum(1 for r in rows if str(r.get("status", "")) == "error")
    review = sum(1 for r in rows if _is_true(r.get("needs_review")))
    avg_talc = sum(_as_float(r.get("talc_percent_model")) for r in rows if str(r.get("status", "success")) != "error") / max(success, 1)
    avg_conf = sum(_as_float(r.get("mean_confidence")) for r in rows if str(r.get("status", "success")) != "error") / max(success, 1)
    summary_rows = [
        ["Всего файлов", str(total), "Успешно", str(success)],
        ["Ошибки", str(errors), "Требуют проверки", str(review)],
        ["Средняя доля талька", f"{avg_talc:.2f}%", "Средняя уверенность", f"{avg_conf:.2f}"],
    ]
    story.append(Paragraph("Сводка", styles["h2"]))
    story.append(_make_table([["Метрика", "Значение", "Метрика", "Значение"], *summary_rows], [4.2 * cm, 3.5 * cm, 4.2 * cm, 4.6 * cm], regular, bold))
    story.append(Spacer(1, 8))

    class_counter = Counter(_as_text(r.get("final_class") or r.get("model_class")) for r in rows if str(r.get("status", "success")) != "error")
    class_rows = [["Класс", "Количество"]]
    for cls in CLASS_ORDER:
        class_rows.append([cls, str(class_counter.get(cls, 0))])
    for cls, count in class_counter.items():
        if cls not in CLASS_ORDER and cls != "-":
            class_rows.append([cls, str(count)])
    story.append(Paragraph("Распределение по классам", styles["h2"]))
    story.append(_make_table(class_rows, [10 * cm, 4 * cm], regular, bold))
    story.append(Spacer(1, 8))

    table_rows = [["Файл", "Класс", "Тальк", "Сульфиды", "Увер.", "Статус"]]
    for r in rows:
        status = _as_text(r.get("status", "success"))
        cls = _as_text(r.get("final_class") or r.get("model_class"))
        table_rows.append([
            _as_text(r.get("original_name")),
            cls,
            _fmt(r.get("talc_percent_model"), suffix="%") if status != "error" else "-",
            _fmt(r.get("sulfide_percent"), suffix="%") if status != "error" else "-",
            _fmt(r.get("mean_confidence"), digits=2) if status != "error" else "-",
            _as_text(r.get("review_status") or status),
        ])
    story.append(Paragraph("Таблица результатов", styles["h2"]))
    story.append(_make_table(table_rows, [4.0 * cm, 3.3 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 3.0 * cm], regular, bold))

    successful_rows = [r for r in rows if str(r.get("status", "success")) != "error"]
    if successful_rows:
        story.append(PageBreak())
        story.append(Paragraph("Визуальные результаты", styles["title"]))
        story.append(Paragraph(
            "Цвета маски: зеленый - обычные срастания, красный - тонкие срастания, синий - тальк.",
            styles["normal"],
        ))
        story.append(Spacer(1, 8))

    for idx, r in enumerate(successful_rows[:max_samples], start=1):
        img_path = _resolve_artifact(r.get("image_path"), project_root, output_root)
        overlay_path = _resolve_artifact(r.get("class_overlay_path") or r.get("overlay_path"), project_root, output_root)
        conf_path = _resolve_artifact(r.get("confidence_map_path"), project_root, output_root)

        metrics = [
            ["Метрика", "Значение"],
            ["Итоговый класс", _as_text(r.get("final_class") or r.get("model_class"))],
            ["Доля талька", _fmt(r.get("talc_percent_model"), suffix="%")],
            ["Сульфиды", _fmt(r.get("sulfide_percent"), suffix="%")],
            ["Обычные среди сульфидов", _fmt(r.get("ordinary_share_among_sulfides"), suffix="%")],
            ["Тонкие среди сульфидов", _fmt(r.get("thin_share_among_sulfides"), suffix="%")],
            ["Уверенность", _fmt(r.get("mean_confidence"), digits=2)],
            ["Проверка", _as_text(r.get("review_reason") if _is_true(r.get("needs_review")) else "не требуется")],
        ]
        header = Paragraph(f"{idx}. {_as_text(r.get('original_name'))}", styles["h2"])
        img_table = Table(
            [[
                _img_flowable(img_path, 7.6 * cm, 5.6 * cm),
                _img_flowable(overlay_path, 7.6 * cm, 5.6 * cm),
            ], [
                Paragraph("Исходное изображение", styles["center"]),
                Paragraph("Итоговая маска", styles["center"]),
            ]],
            colWidths=[8.0 * cm, 8.0 * cm],
        )
        img_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#D7DEE8")),
        ]))
        block = [header, img_table, Spacer(1, 5), _make_table(metrics, [6.8 * cm, 9.7 * cm], regular, bold)]
        if conf_path is not None:
            block.extend([Spacer(1, 5), Paragraph("Карта уверенности сохранена в outputs/confidence_maps.", styles["small"])])
        story.append(KeepTogether(block))
        story.append(Spacer(1, 12))

    if len(successful_rows) > max_samples:
        story.append(Paragraph(f"В отчет включены первые {max_samples} изображений из {len(successful_rows)}. Полная таблица доступна в CSV.", styles["small"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return report_path
