from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.image_io import save_mask, save_rgb, save_class_mask, save_float_map
from src.visualization import make_overlay_pil, make_diff_overlay_pil, make_class_overlay_pil, make_superpixel_overlay_pil


def _output_option(options: dict[str, Any] | None, key: str, default: bool = True) -> bool:
    if not isinstance(options, dict):
        return bool(default)
    return bool(options.get(key, default))


class OutputStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.class_masks = self.root / "class_masks"
        self.hybrid_class_masks = self.root / "hybrid_class_masks"
        self.superpixel_masks = self.root / "superpixel_masks"
        self.expert_masks = self.root / "expert_masks"
        self.expert_class_masks = self.root / "expert_class_masks"
        self.overlays = self.root / "overlays"
        self.superpixel_overlays = self.root / "superpixel_overlays"
        self.confidence_maps = self.root / "confidence_maps"
        self.superpixel_confidence_maps = self.root / "superpixel_confidence_maps"
        self.gradcams = self.root / "gradcams"
        self.talc_scores = self.root / "talc_scores"
        self.reports = self.root / "reports"
        self.reviews = self.root / "reviews"
        self.active_learning = self.root / "active_learning"
        for p in [
            self.images,
            self.masks,
            self.class_masks,
            self.hybrid_class_masks,
            self.superpixel_masks,
            self.expert_masks,
            self.expert_class_masks,
            self.overlays,
            self.superpixel_overlays,
            self.confidence_maps,
            self.superpixel_confidence_maps,
            self.gradcams,
            self.talc_scores,
            self.reports,
            self.reviews,
            self.active_learning,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def result_paths(self, sample_id: str) -> dict[str, Path]:
        return {
            "image_path": self.images / f"{sample_id}.png",
            "model_mask_path": self.masks / f"{sample_id}_model_mask.png",
            "class_mask_path": self.class_masks / f"{sample_id}_class_mask.png",
            "hybrid_class_mask_path": self.hybrid_class_masks / f"{sample_id}_hybrid_class_mask.png",
            "superpixel_mask_path": self.superpixel_masks / f"{sample_id}_superpixel_native_mask.png",
            "expert_mask_path": self.expert_masks / f"{sample_id}_expert_mask.png",
            "expert_class_mask_path": self.expert_class_masks / f"{sample_id}_expert_class_mask.png",
            "overlay_path": self.overlays / f"{sample_id}_overlay.png",
            "class_overlay_path": self.overlays / f"{sample_id}_class_overlay.png",
            "hybrid_class_overlay_path": self.overlays / f"{sample_id}_hybrid_class_overlay.png",
            "superpixel_overlay_path": self.superpixel_overlays / f"{sample_id}_superpixel_overlay.png",
            "diff_overlay_path": self.overlays / f"{sample_id}_diff.png",
            "confidence_map_path": self.confidence_maps / f"{sample_id}_confidence.png",
            "superpixel_confidence_path": self.superpixel_confidence_maps / f"{sample_id}_superpixel_confidence.png",
            "gradcam_path": self.gradcams / f"{sample_id}_gradcam.png",
            "talc_score_path": self.talc_scores / f"{sample_id}_talc_score.png",
            "review_json_path": self.reviews / f"{sample_id}_review.json",
        }

    def save_analysis(
        self,
        sample_id: str,
        original_name: str,
        image_rgb: np.ndarray,
        result: dict[str, Any],
        batch_id: str = "",
        metadata: dict[str, Any] | None = None,
        output_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        paths = self.result_paths(sample_id)
        metadata = metadata or {}
        save_visual_overlays = _output_option(output_options, "save_visual_overlays", True)
        save_segmentation_comparison = _output_option(output_options, "save_segmentation_comparison", True)
        save_confidence_map = _output_option(output_options, "compute_confidence_map", True)
        save_gradcam = _output_option(output_options, "compute_gradcam", True)
        save_talc_score = _output_option(output_options, "compute_detailed_talc_map", True)

        save_rgb(paths["image_path"], image_rgb)
        save_mask(paths["model_mask_path"], result["talc_mask"])
        overlay_path = ""
        if save_visual_overlays:
            make_overlay_pil(image_rgb, result["talc_mask"], alpha=0.45).save(paths["overlay_path"])
            overlay_path = self.portable_path(paths["overlay_path"])

        class_mask = result.get("class_mask")
        class_mask_path = ""
        class_overlay_path = ""
        if class_mask is not None:
            save_class_mask(paths["class_mask_path"], class_mask)
            class_mask_path = self.portable_path(paths["class_mask_path"])
            if save_visual_overlays:
                make_class_overlay_pil(image_rgb, class_mask, alpha=0.45).save(paths["class_overlay_path"])
                class_overlay_path = self.portable_path(paths["class_overlay_path"])

        hybrid_class_mask_path = ""
        hybrid_class_overlay_path = ""
        if save_segmentation_comparison and result.get("hybrid_class_mask") is not None:
            hybrid_mask = result["hybrid_class_mask"]
            save_class_mask(paths["hybrid_class_mask_path"], hybrid_mask)
            hybrid_class_mask_path = self.portable_path(paths["hybrid_class_mask_path"])
            if save_visual_overlays:
                make_class_overlay_pil(image_rgb, hybrid_mask, alpha=0.45).save(paths["hybrid_class_overlay_path"])
                hybrid_class_overlay_path = self.portable_path(paths["hybrid_class_overlay_path"])

        superpixel_mask_path = ""
        superpixel_overlay_path = ""
        if save_segmentation_comparison and result.get("superpixel_native_mask") is not None:
            native_mask = result["superpixel_native_mask"]
            save_class_mask(paths["superpixel_mask_path"], native_mask)
            superpixel_mask_path = self.portable_path(paths["superpixel_mask_path"])
            if save_visual_overlays:
                make_superpixel_overlay_pil(image_rgb, native_mask, alpha=0.45).save(paths["superpixel_overlay_path"])
                superpixel_overlay_path = self.portable_path(paths["superpixel_overlay_path"])

        confidence_map_path = ""
        if save_confidence_map and result.get("confidence_map") is not None:
            save_float_map(paths["confidence_map_path"], result["confidence_map"])
            confidence_map_path = self.portable_path(paths["confidence_map_path"])

        superpixel_confidence_path = ""
        if save_confidence_map and save_segmentation_comparison and result.get("superpixel_confidence_map") is not None:
            save_float_map(paths["superpixel_confidence_path"], result["superpixel_confidence_map"])
            superpixel_confidence_path = self.portable_path(paths["superpixel_confidence_path"])

        gradcam_path = ""
        if save_gradcam and result.get("gradcam") is not None:
            save_float_map(paths["gradcam_path"], result["gradcam"])
            gradcam_path = self.portable_path(paths["gradcam_path"])

        talc_score_path = ""
        if save_talc_score and result.get("talc_score") is not None:
            save_float_map(paths["talc_score_path"], result["talc_score"])
            talc_score_path = self.portable_path(paths["talc_score_path"])

        class_probs = result.get("class_probs") or {}
        row = {
            "sample_id": sample_id,
            "batch_id": batch_id,
            "original_name": original_name,
            "status": result.get("status", "success"),
            "error_message": "",
            "processing_time_sec": result.get("processing_time_sec", ""),
            "image_path": self.portable_path(paths["image_path"]),
            "model_mask_path": self.portable_path(paths["model_mask_path"]),
            "class_mask_path": class_mask_path,
            "hybrid_class_mask_path": hybrid_class_mask_path,
            "superpixel_mask_path": superpixel_mask_path,
            "overlay_path": overlay_path,
            "class_overlay_path": class_overlay_path,
            "hybrid_class_overlay_path": hybrid_class_overlay_path,
            "superpixel_overlay_path": superpixel_overlay_path,
            "confidence_map_path": confidence_map_path,
            "superpixel_confidence_path": superpixel_confidence_path,
            "gradcam_path": gradcam_path,
            "talc_score_path": talc_score_path,
            "model_class": result["ore_class"],
            "raw_model_class": result.get("model_pred_class", ""),
            "expert_class": "",
            "final_class": result["ore_class"],
            "final_source": "model",
            "p_row": class_probs.get("row", ""),
            "p_difficult": class_probs.get("difficult", ""),
            "p_talc": class_probs.get("talc", ""),
            "talc_percent_model": float(result["talc_percent"]),
            "talc_percent_expert": "",
            "sulfide_percent": float(result.get("sulfide_percent", 0.0)),
            "ordinary_percent": float(result.get("ordinary_percent", 0.0)),
            "thin_percent": float(result.get("thin_percent", 0.0)),
            "ordinary_share_among_sulfides": float(result.get("ordinary_share_among_sulfides", 0.0)),
            "thin_share_among_sulfides": float(result.get("thin_share_among_sulfides", 0.0)),
            "mean_confidence": float(result.get("mean_confidence", 0.0)),
            "needs_review": bool(result.get("needs_review", False)),
            "review_reason": result.get("review_reason", ""),
            "review_status": "ожидает проверки" if result.get("needs_review", False) else "не проверено",
            "mask_quality": "",
            "expert_comment": "",
            "runtime_mode": result.get("runtime_mode", "unknown"),
            "segmentation_source": result.get("segmentation_source", ""),
            "superpixel_talc_percent": float((result.get("superpixel_metrics") or {}).get("superpixel_talc_percent", 0.0)),
            "superpixel_sulfide_percent": float((result.get("superpixel_metrics") or {}).get("superpixel_sulfide_percent", 0.0)),
            "superpixel_thin_share_in_sulfides": float((result.get("superpixel_metrics") or {}).get("superpixel_thin_share_in_sulfides", 0.0)),
            "adapter_warnings": "; ".join(result.get("adapter_warnings", []) or []),
            "metadata_json": json.dumps(metadata, ensure_ascii=False) if metadata else "",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.upsert_row(row)

        self._write_analysis_json(paths["review_json_path"], sample_id, row, result, metadata)
        return row

    def save_error(
        self,
        sample_id: str,
        original_name: str,
        error_message: str,
        batch_id: str = "",
        processing_time_sec: float | str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "sample_id": sample_id,
            "batch_id": batch_id,
            "original_name": original_name,
            "status": "error",
            "error_message": error_message,
            "processing_time_sec": processing_time_sec,
            "needs_review": True,
            "review_reason": "ошибка обработки",
            "review_status": "ошибка",
            "final_source": "error",
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False) if metadata else "",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.upsert_row(row)
        return row

    def _write_analysis_json(self, path: Path, sample_id: str, row: dict[str, Any], result: dict[str, Any], metadata: dict[str, Any]) -> None:
        payload = {
            "sample_id": sample_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "metadata": metadata,
            "row": {k: _json_safe(v) for k, v in row.items()},
            "metrics": {
                "talc_percent": result.get("talc_percent"),
                "sulfide_percent": result.get("sulfide_percent"),
                "ordinary_percent": result.get("ordinary_percent"),
                "thin_percent": result.get("thin_percent"),
                "ordinary_share_among_sulfides": result.get("ordinary_share_among_sulfides"),
                "thin_share_among_sulfides": result.get("thin_share_among_sulfides"),
                "mean_confidence": result.get("mean_confidence"),
            },
            "class_probs": result.get("class_probs", {}),
            "conclusion": result.get("conclusion", ""),
            "adapter_warnings": result.get("adapter_warnings", []),
            "segmentation_source": result.get("segmentation_source", ""),
            "superpixel_metrics": result.get("superpixel_metrics", {}),
            "superpixel_ratios": result.get("superpixel_ratios", {}),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def csv_path(self) -> Path:
        return self.reviews / "analysis_results.csv"

    def portable_path(self, path: str | Path) -> str:
        """Store paths relative to the project folder when possible."""
        path = Path(path)
        try:
            return str(path.relative_to(self.root.parent))
        except ValueError:
            return str(path)

    def read_rows(self) -> list[dict[str, Any]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def upsert_row(self, row: dict[str, Any]) -> None:
        rows = self.read_rows()
        sample_id = row["sample_id"]
        updated = False
        for i, old in enumerate(rows):
            if old.get("sample_id") == sample_id:
                merged = {**old, **{k: _stringify(v) for k, v in row.items()}}
                rows[i] = merged
                updated = True
                break
        if not updated:
            rows.append({k: _stringify(v) for k, v in row.items()})
        self.write_rows(rows)

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id", "batch_id", "original_name", "status", "error_message", "processing_time_sec",
            "image_path", "model_mask_path", "class_mask_path", "hybrid_class_mask_path", "superpixel_mask_path",
            "expert_mask_path", "expert_class_mask_path",
            "overlay_path", "class_overlay_path", "hybrid_class_overlay_path", "superpixel_overlay_path", "diff_overlay_path",
            "confidence_map_path", "superpixel_confidence_path", "gradcam_path", "talc_score_path",
            "model_class", "raw_model_class", "expert_class", "final_class", "final_source",
            "p_row", "p_difficult", "p_talc",
            "talc_percent_model", "talc_percent_expert", "sulfide_percent", "ordinary_percent", "thin_percent",
            "ordinary_share_among_sulfides", "thin_share_among_sulfides", "mean_confidence",
            "needs_review", "review_reason", "review_status", "mask_quality", "expert_comment",
            "runtime_mode", "segmentation_source",
            "superpixel_talc_percent", "superpixel_sulfide_percent", "superpixel_thin_share_in_sulfides",
            "adapter_warnings", "metadata_json", "timestamp",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def save_expert_review(
        self,
        sample_id: str,
        image_rgb: np.ndarray,
        model_mask: np.ndarray,
        expert_mask: np.ndarray,
        review_payload: dict[str, Any],
        model_class_mask: np.ndarray | None = None,
        expert_class_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        paths = self.result_paths(sample_id)
        if expert_class_mask is not None:
            # Всегда синхронизируем бинарную маску талька с классом 3.
            expert_mask = np.asarray(expert_class_mask, dtype=np.uint8) == 3

        save_mask(paths["expert_mask_path"], expert_mask)
        make_diff_overlay_pil(image_rgb, model_mask, expert_mask, alpha=0.55).save(paths["diff_overlay_path"])

        expert_class_mask_path = ""
        if expert_class_mask is not None:
            save_class_mask(paths["expert_class_mask_path"], expert_class_mask)
            expert_class_mask_path = self.portable_path(paths["expert_class_mask_path"])

        valid_area = image_rgb.shape[0] * image_rgb.shape[1]
        expert_percent = float(expert_mask.sum() / max(valid_area, 1) * 100.0)
        final_class = review_payload.get("expert_class") or review_payload.get("model_class")
        final_source = "expert" if review_payload.get("class_decision") == "Класс неверный" else "model_confirmed"
        if review_payload.get("class_decision") == "Не уверен / нужна проверка":
            final_source = "needs_review"
            final_class = review_payload.get("model_class")

        row = {
            "sample_id": sample_id,
            "expert_mask_path": self.portable_path(paths["expert_mask_path"]),
            "expert_class_mask_path": expert_class_mask_path,
            "diff_overlay_path": self.portable_path(paths["diff_overlay_path"]),
            "expert_class": review_payload.get("expert_class", ""),
            "final_class": final_class,
            "final_source": final_source,
            "talc_percent_expert": expert_percent,
            "review_status": review_payload.get("review_status", "проверено"),
            "mask_quality": review_payload.get("mask_quality", ""),
            "expert_comment": review_payload.get("expert_comment", ""),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.upsert_row(row)

        full_payload = {
            "sample_id": sample_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model": {
                "class": review_payload.get("model_class", ""),
                "talc_percent": review_payload.get("talc_percent_model", ""),
                "mask_path": self.portable_path(paths["model_mask_path"]),
                "class_mask_path": self.portable_path(paths["class_mask_path"]),
            },
            "expert": {
                "class_decision": review_payload.get("class_decision", ""),
                "expert_class": review_payload.get("expert_class", ""),
                "mask_quality": review_payload.get("mask_quality", ""),
                "talc_percent": expert_percent,
                "mask_path": self.portable_path(paths["expert_mask_path"]),
                "class_mask_path": expert_class_mask_path,
                "comment": review_payload.get("expert_comment", ""),
            },
            "final": {"class": final_class, "source": final_source},
        }
        paths["review_json_path"].write_text(json.dumps(full_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if final_source in {"expert", "needs_review"} or review_payload.get("mask_quality") in {"Частично корректная", "Маска неверная"}:
            al_dir = self.active_learning / sample_id
            al_dir.mkdir(parents=True, exist_ok=True)
            save_rgb(al_dir / "image.png", image_rgb)
            save_mask(al_dir / "expert_talc_mask.png", expert_mask)
            if paths["class_mask_path"].exists():
                # class_mask сохраняем рядом, чтобы ML-команда могла использовать мультиклассовую основу.
                (al_dir / "model_class_mask.png").write_bytes(paths["class_mask_path"].read_bytes())
            if expert_class_mask is not None:
                save_class_mask(al_dir / "expert_class_mask.png", expert_class_mask)
            (al_dir / "review.json").write_text(json.dumps(full_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return row


def _json_safe(v: Any):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    return v


def _stringify(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, np.floating):
        return f"{float(v):.6f}"
    if isinstance(v, np.integer):
        return str(int(v))
    if v is None:
        return ""
    return str(v)
