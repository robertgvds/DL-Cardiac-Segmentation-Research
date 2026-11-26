"""Avaliação volumétrica 3D completa para segmentação cardíaca e de fibrose em LGE-CMR.

Calcula por paciente, na resolução física original (`pixdim`):
  - Dice 3D por classe: `VE` (1), `Mio_Total` (2∪3), `Mio_Saudavel` (2),
    `Fibrose` (3 — `NaN` para casos saudáveis sem fibrose no GT) e `VD` (4).
  - HD95 em milímetros (distância de Hausdorff no percentil 95 usando `distance_transform_edt`).
  - Volume de fibrose (GT e predito em mL), erro absoluto (mL) e relativo (%).
  - Contagem e volume (mL) de falsos positivos de fibrose nos casos saudáveis.
  - Exporta `metrics_per_patient.csv`, `summary.json` e `dice_boxplot.png`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

LOGGER = logging.getLogger(__name__)

# Convenção de rótulos CMR-MULTI / Dataset501
LABEL_LV = 1
LABEL_MYO_HEALTHY = 2
LABEL_SCAR = 3
LABEL_RV = 4


@dataclass(frozen=True)
class PatientMetrics3D:
    """Métricas 3D consolidadas de um único paciente na resolução original."""

    patient_id: str
    has_scar: bool
    dice_lv: float
    dice_myo_total: float
    dice_myo_healthy: float
    dice_scar: float
    dice_rv: float
    hd95_lv_mm: float
    hd95_myo_total_mm: float
    hd95_scar_mm: float
    hd95_rv_mm: float
    gt_scar_volume_ml: float
    pred_scar_volume_ml: float
    scar_volume_abs_error_ml: float
    scar_volume_rel_error_pct: float
    healthy_fp_scar_voxels: int
    healthy_fp_scar_ml: float


class BinarySurfaceMetrics:
    """Utilitários matemáticos puros para Dice 3D e HD95 em milímetros."""

    @staticmethod
    def compute_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """Calcula coeficiente Dice 3D entre duas máscaras booleanas.

        Retorna `NaN` quando o ground truth está vazio para evitar inflar
        artificialmente a média de fibrose em pacientes saudáveis.
        """
        gt_sum = int(np.count_nonzero(gt_mask))
        pred_sum = int(np.count_nonzero(pred_mask))
        if gt_sum == 0:
            return float("nan")
        if pred_sum == 0:
            return 0.0
        intersection = int(np.count_nonzero(gt_mask & pred_mask))
        return (2.0 * intersection) / float(gt_sum + pred_sum)

    @staticmethod
    def compute_hd95_mm(
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        spacing_mm: Tuple[float, float, float],
    ) -> float:
        """Calcula distância de Hausdorff no percentil 95 (HD95) em milímetros.

        Usa erosão binária para extrair voxels de superfície e transformada de
        distância euclidiana com espaçamento anisotrópico real (`sampling=spacing_mm`).
        """
        if not np.any(gt_mask) or not np.any(pred_mask):
            return float("nan")

        # Superfície exata de cada estrutura 3D
        gt_surface = gt_mask ^ binary_erosion(gt_mask)
        pred_surface = pred_mask ^ binary_erosion(pred_mask)
        if not np.any(gt_surface):
            gt_surface = gt_mask
        if not np.any(pred_surface):
            pred_surface = pred_mask

        dt_pred = distance_transform_edt(~pred_surface, sampling=spacing_mm)
        dt_gt = distance_transform_edt(~gt_surface, sampling=spacing_mm)

        dist_gt_to_pred = dt_pred[gt_surface]
        dist_pred_to_gt = dt_gt[pred_surface]
        all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])
        return float(np.percentile(all_distances, 95))


class CardiacMetrics3DEvaluator:
    """Avalia um diretório de predições NIfTI contra máscaras de referência."""

    def __init__(self, gt_dir: Path, pred_dir: Path, output_dir: Path) -> None:
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.output_dir = output_dir

    @staticmethod
    def _strip_nifti_suffix(filename: str) -> str:
        """Extrai o identificador do paciente removendo `.nii.gz` ou `.nii`."""
        if filename.endswith(".nii.gz"):
            return filename[:-7]
        if filename.endswith(".nii"):
            return filename[:-4]
        return Path(filename).stem

    @staticmethod
    def _index_nifti_files(directory: Path) -> Dict[str, Path]:
        """Mapeia `{patient_id: path}` aceitando `.nii` e `.nii.gz`."""
        result: Dict[str, Path] = {}
        for p in sorted(directory.iterdir()):
            if p.is_file() and not p.name.startswith(".") and (p.name.endswith(".nii.gz") or p.name.endswith(".nii")):
                result[CardiacMetrics3DEvaluator._strip_nifti_suffix(p.name)] = p
        return result

    def evaluate_patient(self, patient_id: str, gt_path: Path, pred_path: Path) -> PatientMetrics3D:
        """Calcula todas as métricas 3D para um único paciente."""
        gt_nii = nib.load(str(gt_path))
        pred_nii = nib.load(str(pred_path))

        gt = np.rint(np.squeeze(gt_nii.get_fdata())).astype(np.int16)
        pred = np.rint(np.squeeze(pred_nii.get_fdata())).astype(np.int16)
        if gt.shape != pred.shape:
            raise ValueError(f"Shapes divergentes em {patient_id}: GT={gt.shape} vs Pred={pred.shape}")

        zooms_raw = gt_nii.header.get_zooms()[:3]
        spacing_mm = (float(zooms_raw[0]), float(zooms_raw[1]), float(zooms_raw[2]))
        voxel_vol_ml = (spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0

        gt_lv, pr_lv = (gt == LABEL_LV), (pred == LABEL_LV)
        gt_myo_h, pr_myo_h = (gt == LABEL_MYO_HEALTHY), (pred == LABEL_MYO_HEALTHY)
        gt_scar, pr_scar = (gt == LABEL_SCAR), (pred == LABEL_SCAR)
        gt_myo_tot = gt_myo_h | gt_scar
        pr_myo_tot = pr_myo_h | pr_scar
        gt_rv, pr_rv = (gt == LABEL_RV), (pred == LABEL_RV)

        has_scar = bool(np.any(gt_scar))
        gt_scar_ml = float(np.count_nonzero(gt_scar)) * voxel_vol_ml
        pred_scar_voxels = int(np.count_nonzero(pr_scar))
        pred_scar_ml = float(pred_scar_voxels) * voxel_vol_ml
        abs_err_ml = abs(pred_scar_ml - gt_scar_ml)
        rel_err_pct = (abs_err_ml / gt_scar_ml * 100.0) if gt_scar_ml > 0.0 else float("nan")

        healthy_fp_voxels = 0 if has_scar else pred_scar_voxels
        healthy_fp_ml = 0.0 if has_scar else pred_scar_ml

        return PatientMetrics3D(
            patient_id=patient_id,
            has_scar=has_scar,
            dice_lv=BinarySurfaceMetrics.compute_dice(gt_lv, pr_lv),
            dice_myo_total=BinarySurfaceMetrics.compute_dice(gt_myo_tot, pr_myo_tot),
            dice_myo_healthy=BinarySurfaceMetrics.compute_dice(gt_myo_h, pr_myo_h),
            dice_scar=BinarySurfaceMetrics.compute_dice(gt_scar, pr_scar),
            dice_rv=BinarySurfaceMetrics.compute_dice(gt_rv, pr_rv),
            hd95_lv_mm=BinarySurfaceMetrics.compute_hd95_mm(gt_lv, pr_lv, spacing_mm),
            hd95_myo_total_mm=BinarySurfaceMetrics.compute_hd95_mm(gt_myo_tot, pr_myo_tot, spacing_mm),
            hd95_scar_mm=BinarySurfaceMetrics.compute_hd95_mm(gt_scar, pr_scar, spacing_mm),
            hd95_rv_mm=BinarySurfaceMetrics.compute_hd95_mm(gt_rv, pr_rv, spacing_mm),
            gt_scar_volume_ml=gt_scar_ml,
            pred_scar_volume_ml=pred_scar_ml,
            scar_volume_abs_error_ml=abs_err_ml,
            scar_volume_rel_error_pct=rel_err_pct,
            healthy_fp_scar_voxels=healthy_fp_voxels,
            healthy_fp_scar_ml=healthy_fp_ml,
        )

    @staticmethod
    def _summarize_column(values: Sequence[float]) -> Dict[str, Optional[float]]:
        """Calcula estatísticas descritivas ignorando `NaN`."""
        valid = [v for v in values if not math.isnan(v)]
        if not valid:
            return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
        arr = np.asarray(valid, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    def _export_csv(self, results: Sequence[PatientMetrics3D], csv_path: Path) -> None:
        """Exporta tabela por paciente com ponto decimal e separador vírgula."""
        fieldnames = list(asdict(results[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                raw = asdict(row)
                formatted = {
                    k: ("" if isinstance(v, float) and math.isnan(v) else (f"{v:.6f}" if isinstance(v, float) else v))
                    for k, v in raw.items()
                }
                writer.writerow(formatted)

    def _export_boxplot(self, results: Sequence[PatientMetrics3D], plot_path: Path) -> None:
        """Gera boxplot comparativo do Dice 3D por estrutura cardíaca."""
        series_spec = [
            ("VE", [r.dice_lv for r in results]),
            ("Mio Total", [r.dice_myo_total for r in results]),
            ("Mio Saudável", [r.dice_myo_healthy for r in results]),
            ("Fibrose\n(GT+)", [r.dice_scar for r in results if not math.isnan(r.dice_scar)]),
            ("VD", [r.dice_rv for r in results]),
        ]
        labels = [name for name, vals in series_spec if vals]
        data = [vals for _, vals in series_spec if vals]

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for idx, vals in enumerate(data, start=1):
            ax.scatter([idx] * len(vals), vals, color="#1f2937", s=26, zorder=3, alpha=0.8)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Coeficiente Dice 3D")
        ax.set_title(f"Avaliação Volumétrica 3D — N={len(results)} pacientes")
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        fig.tight_layout()
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)

    def run(self) -> Dict[str, object]:
        """Executa avaliação completa, exporta CSV, resumo JSON e boxplot PNG."""
        gt_map = self._index_nifti_files(self.gt_dir)
        pred_map = self._index_nifti_files(self.pred_dir)
        common_ids = sorted(set(gt_map).intersection(pred_map))
        if not common_ids:
            raise FileNotFoundError(
                f"Nenhum paciente em comum entre GT ({self.gt_dir}) e Pred ({self.pred_dir})."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = [self.evaluate_patient(cid, gt_map[cid], pred_map[cid]) for cid in common_ids]

        csv_path = self.output_dir / "metrics_per_patient.csv"
        self._export_csv(results, csv_path)

        metric_keys = [
            "dice_lv",
            "dice_myo_total",
            "dice_myo_healthy",
            "dice_scar",
            "dice_rv",
            "hd95_lv_mm",
            "hd95_myo_total_mm",
            "hd95_scar_mm",
            "hd95_rv_mm",
            "scar_volume_abs_error_ml",
            "scar_volume_rel_error_pct",
        ]
        summary: Dict[str, object] = {
            "num_patients": len(results),
            "num_with_scar": sum(1 for r in results if r.has_scar),
            "num_healthy": sum(1 for r in results if not r.has_scar),
            "metrics": {
                key: self._summarize_column([float(getattr(r, key)) for r in results])
                for key in metric_keys
            },
            "healthy_false_positives": [
                {
                    "patient_id": r.patient_id,
                    "fp_voxels": r.healthy_fp_scar_voxels,
                    "fp_volume_ml": round(r.healthy_fp_scar_ml, 4),
                }
                for r in results
                if not r.has_scar
            ],
        }
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        boxplot_path = self.output_dir / "dice_boxplot.png"
        self._export_boxplot(results, boxplot_path)

        LOGGER.info("Avaliação 3D concluída para %d pacientes. Saídas em %s", len(results), self.output_dir)
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(description="Calcula métricas volumétricas 3D (Dice, HD95, volume de fibrose).")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Diretório contendo máscaras Ground Truth.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Diretório contendo máscaras preditas.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório de saída (CSV, JSON, boxplot).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI do módulo de avaliação."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    evaluator = CardiacMetrics3DEvaluator(
        gt_dir=args.gt_dir.resolve(),
        pred_dir=args.pred_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    summary = evaluator.run()
    LOGGER.info("Resumo:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
