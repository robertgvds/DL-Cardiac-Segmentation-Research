"""Avaliação volumétrica 3D completa (Clínica + Challenge CMR-MULTI MICCAI 2026).

Suporta integralmente os modos `anatomy_4c` (`0=Bg, 1=VE, 2=MioTotal, 3=VD`) e
`full_5c` (`0=Bg, 1=VE, 2=MioSaudavel, 3=Scar, 4=VD`), calculando na resolução
física original (`pixdim`):
  - Dice 3D padrão GT+ (`NaN` quando a classe está ausente no GT).
  - Board DSC (convenção estrita CMR-MULTI MICCAI 2026: `both empty = 1.0`,
    `one empty = 0.0`, penalizando falsos positivos de fibrose em saudáveis).
  - Distâncias de superfície 3D em milímetros: **HD95**, **HD máximo** (oficial
    CMR-MULTI) e **ASD / ASSD** (Average Symmetric Surface Distance).
  - Quantificação clínica e de challenge de fibrose: volume (mL), massa (g,
    densidade miocárdica `1.05 g/mL`), fração fibrótica (% do miocárdio),
    erro absoluto (mL), erro relativo (%), **RAE** e **vPCC** de massa fibrótica.
  - Scores compostos oficiais: `cmr_multi_task2_score` (5 classes) e
    `cmr_multi_anat_geom_score` (4 classes).
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
from scipy.stats import pearsonr

from src.data.cardiac_dataset import TaskMode

LOGGER = logging.getLogger(__name__)

# Densidade física padrão do tecido miocárdico utilizada no CMR-MULTI (MICCAI 2026)
MYOCARDIAL_DENSITY_G_PER_ML = 1.05


@dataclass(frozen=True)
class PatientMetrics3D:
    """Métricas 3D consolidadas de um único paciente na resolução física original."""

    patient_id: str
    task_mode: str
    has_scar: bool
    dice_lv: float
    dice_myo_total: float
    dice_myo_healthy: float
    dice_scar: float
    dice_rv: float
    board_dice_scar: float
    board_dice_mean: float
    hd95_lv_mm: float
    hd95_myo_total_mm: float
    hd95_scar_mm: float
    hd95_rv_mm: float
    hd_lv_mm: float
    hd_myo_total_mm: float
    hd_scar_mm: float
    hd_rv_mm: float
    asd_lv_mm: float
    asd_myo_total_mm: float
    asd_scar_mm: float
    asd_rv_mm: float
    gt_myo_volume_ml: float
    pred_myo_volume_ml: float
    myo_volume_abs_error_ml: float
    gt_scar_volume_ml: float
    pred_scar_volume_ml: float
    gt_scar_mass_g: float
    pred_scar_mass_g: float
    gt_scar_pct_myo: float
    pred_scar_pct_myo: float
    scar_volume_abs_error_ml: float
    scar_volume_rel_error_pct: float
    scar_mass_rae: float
    healthy_fp_scar_voxels: int
    healthy_fp_scar_ml: float


class BinarySurfaceMetrics:
    """Utilitários matemáticos puros para Dice, Board Dice, HD95, HD máximo e ASD em mm."""

    @staticmethod
    def compute_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """Calcula Dice 3D retornando `NaN` se GT vazio (evita inflar média de fibrose)."""
        gt_sum = int(np.count_nonzero(gt_mask))
        pred_sum = int(np.count_nonzero(pred_mask))
        if gt_sum == 0:
            return float("nan")
        if pred_sum == 0:
            return 0.0
        intersection = int(np.count_nonzero(gt_mask & pred_mask))
        return (2.0 * intersection) / float(gt_sum + pred_sum)

    @staticmethod
    def compute_board_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """Calcula Dice na convenção estrita CMR-MULTI (`both empty=1.0`, `one empty=0.0`)."""
        gt_sum = int(np.count_nonzero(gt_mask))
        pred_sum = int(np.count_nonzero(pred_mask))
        if gt_sum == 0 and pred_sum == 0:
            return 1.0
        if gt_sum == 0 or pred_sum == 0:
            return 0.0
        intersection = int(np.count_nonzero(gt_mask & pred_mask))
        return (2.0 * intersection) / float(gt_sum + pred_sum)

    @staticmethod
    def compute_surface_distances_mm(
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        spacing_mm: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Calcula `(HD95, HD_max, ASD)` em milímetros usando espaçamento anisotrópico real.

        Retorna `(NaN, NaN, NaN)` caso uma das máscaras esteja vazia.
        """
        if not np.any(gt_mask) or not np.any(pred_mask):
            return float("nan"), float("nan"), float("nan")

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

        hd95 = float(np.percentile(all_distances, 95))
        hd_max = float(np.max(all_distances))
        asd = float(np.mean(all_distances))
        return hd95, hd_max, asd


class CardiacMetrics3DEvaluator:
    """Avalia um diretório de predições NIfTI contra máscaras Ground Truth padronizadas."""

    def __init__(
        self,
        gt_dir: Path,
        pred_dir: Path,
        output_dir: Path,
        task_mode: TaskMode = TaskMode.FULL_5C,
        cohort_name: str = "CMR-MULTI",
    ) -> None:
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.output_dir = output_dir
        self.task_mode = task_mode
        self.cohort_name = cohort_name

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
            if p.is_file() and not p.name.startswith(".") and (
                p.name.endswith(".nii.gz") or p.name.endswith(".nii")
            ):
                result[CardiacMetrics3DEvaluator._strip_nifti_suffix(p.name)] = p
        return result

    def evaluate_patient(self, patient_id: str, gt_path: Path, pred_path: Path) -> PatientMetrics3D:
        """Calcula todas as métricas 3D clínicas e do challenge para um único paciente."""
        gt_nii = nib.load(str(gt_path))
        pred_nii = nib.load(str(pred_path))

        gt = np.rint(np.squeeze(gt_nii.get_fdata())).astype(np.int16)
        pred = np.rint(np.squeeze(pred_nii.get_fdata())).astype(np.int16)
        if gt.shape != pred.shape:
            raise ValueError(f"Shapes divergentes em {patient_id}: GT={gt.shape} vs Pred={pred.shape}")

        zooms_raw = gt_nii.header.get_zooms()[:3]
        spacing_mm = (float(zooms_raw[0]), float(zooms_raw[1]), float(zooms_raw[2]))
        voxel_vol_ml = (spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0

        gt_lv, pr_lv = (gt == 1), (pred == 1)

        if self.task_mode is TaskMode.ANATOMY_4C:
            # 4 classes: 0=Bg, 1=VE, 2=Miocárdio Total (Mio+Scar), 3=VD
            gt_myo_tot, pr_myo_tot = (gt == 2), (pred == 2)
            gt_rv, pr_rv = (gt == 3), (pred == 3)
            gt_myo_h = np.zeros_like(gt_lv, dtype=bool)
            pr_myo_h = np.zeros_like(pr_lv, dtype=bool)
            gt_scar = np.zeros_like(gt_lv, dtype=bool)
            pr_scar = np.zeros_like(pr_lv, dtype=bool)
        else:
            # 5 classes: 0=Bg, 1=VE, 2=Mio Saudável, 3=Scar, 4=VD
            gt_myo_h, pr_myo_h = (gt == 2), (pred == 2)
            gt_scar, pr_scar = (gt == 3), (pred == 3)
            gt_myo_tot = gt_myo_h | gt_scar
            pr_myo_tot = pr_myo_h | pr_scar
            gt_rv, pr_rv = (gt == 4), (pred == 4)

        dice_lv = BinarySurfaceMetrics.compute_dice(gt_lv, pr_lv)
        dice_myo_tot = BinarySurfaceMetrics.compute_dice(gt_myo_tot, pr_myo_tot)
        dice_rv = BinarySurfaceMetrics.compute_dice(gt_rv, pr_rv)

        hd95_lv, hd_lv, asd_lv = BinarySurfaceMetrics.compute_surface_distances_mm(gt_lv, pr_lv, spacing_mm)
        hd95_myo, hd_myo, asd_myo = BinarySurfaceMetrics.compute_surface_distances_mm(
            gt_myo_tot, pr_myo_tot, spacing_mm
        )
        hd95_rv, hd_rv, asd_rv = BinarySurfaceMetrics.compute_surface_distances_mm(gt_rv, pr_rv, spacing_mm)

        gt_myo_ml = float(np.count_nonzero(gt_myo_tot)) * voxel_vol_ml
        pred_myo_ml = float(np.count_nonzero(pr_myo_tot)) * voxel_vol_ml
        myo_abs_err_ml = abs(pred_myo_ml - gt_myo_ml)

        if self.task_mode is TaskMode.ANATOMY_4C:
            has_scar = False
            dice_myo_h = float("nan")
            dice_scar = float("nan")
            board_dice_scar = float("nan")
            board_dice_mean = float(
                np.mean(
                    [
                        BinarySurfaceMetrics.compute_board_dice(gt_lv, pr_lv),
                        BinarySurfaceMetrics.compute_board_dice(gt_myo_tot, pr_myo_tot),
                        BinarySurfaceMetrics.compute_board_dice(gt_rv, pr_rv),
                    ]
                )
            )
            hd95_scar = hd_scar = asd_scar = float("nan")
            gt_scar_ml = pred_scar_ml = gt_scar_mass_g = pred_scar_mass_g = float("nan")
            gt_scar_pct = pred_scar_pct = abs_err_ml = rel_err_pct = scar_rae = float("nan")
            healthy_fp_voxels = 0
            healthy_fp_ml = 0.0
        else:
            has_scar = bool(np.any(gt_scar))
            dice_myo_h = BinarySurfaceMetrics.compute_dice(gt_myo_h, pr_myo_h)
            dice_scar = BinarySurfaceMetrics.compute_dice(gt_scar, pr_scar)
            board_dice_scar = BinarySurfaceMetrics.compute_board_dice(gt_scar, pr_scar)
            board_dice_mean = float(
                np.mean(
                    [
                        BinarySurfaceMetrics.compute_board_dice(gt_lv, pr_lv),
                        BinarySurfaceMetrics.compute_board_dice(gt_myo_h, pr_myo_h),
                        board_dice_scar,
                        BinarySurfaceMetrics.compute_board_dice(gt_rv, pr_rv),
                    ]
                )
            )
            hd95_scar, hd_scar, asd_scar = BinarySurfaceMetrics.compute_surface_distances_mm(
                gt_scar, pr_scar, spacing_mm
            )
            gt_scar_ml = float(np.count_nonzero(gt_scar)) * voxel_vol_ml
            pred_scar_voxels = int(np.count_nonzero(pr_scar))
            pred_scar_ml = float(pred_scar_voxels) * voxel_vol_ml
            gt_scar_mass_g = gt_scar_ml * MYOCARDIAL_DENSITY_G_PER_ML
            pred_scar_mass_g = pred_scar_ml * MYOCARDIAL_DENSITY_G_PER_ML
            gt_scar_pct = (gt_scar_ml / gt_myo_ml * 100.0) if gt_myo_ml > 0.0 else float("nan")
            pred_scar_pct = (pred_scar_ml / pred_myo_ml * 100.0) if pred_myo_ml > 0.0 else float("nan")
            abs_err_ml = abs(pred_scar_ml - gt_scar_ml)
            scar_rae = (abs(pred_scar_mass_g - gt_scar_mass_g) / gt_scar_mass_g) if gt_scar_mass_g > 0.0 else float("nan")
            rel_err_pct = (scar_rae * 100.0) if not math.isnan(scar_rae) else float("nan")
            healthy_fp_voxels = 0 if has_scar else pred_scar_voxels
            healthy_fp_ml = 0.0 if has_scar else pred_scar_ml

        return PatientMetrics3D(
            patient_id=patient_id,
            task_mode=self.task_mode.value,
            has_scar=has_scar,
            dice_lv=dice_lv,
            dice_myo_total=dice_myo_tot,
            dice_myo_healthy=dice_myo_h,
            dice_scar=dice_scar,
            dice_rv=dice_rv,
            board_dice_scar=board_dice_scar,
            board_dice_mean=board_dice_mean,
            hd95_lv_mm=hd95_lv,
            hd95_myo_total_mm=hd95_myo,
            hd95_scar_mm=hd95_scar,
            hd95_rv_mm=hd95_rv,
            hd_lv_mm=hd_lv,
            hd_myo_total_mm=hd_myo,
            hd_scar_mm=hd_scar,
            hd_rv_mm=hd_rv,
            asd_lv_mm=asd_lv,
            asd_myo_total_mm=asd_myo,
            asd_scar_mm=asd_scar,
            asd_rv_mm=asd_rv,
            gt_myo_volume_ml=gt_myo_ml,
            pred_myo_volume_ml=pred_myo_ml,
            myo_volume_abs_error_ml=myo_abs_err_ml,
            gt_scar_volume_ml=gt_scar_ml,
            pred_scar_volume_ml=pred_scar_ml,
            gt_scar_mass_g=gt_scar_mass_g,
            pred_scar_mass_g=pred_scar_mass_g,
            gt_scar_pct_myo=gt_scar_pct,
            pred_scar_pct_myo=pred_scar_pct,
            scar_volume_abs_error_ml=abs_err_ml,
            scar_volume_rel_error_pct=rel_err_pct,
            scar_mass_rae=scar_rae,
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

    def _compute_challenge_aggregates(self, results: Sequence[PatientMetrics3D]) -> Dict[str, Optional[float]]:
        """Calcula métricas agregadas oficiais do challenge CMR-MULTI (MICCAI 2026)."""
        dsc_board = float(np.mean([r.board_dice_mean for r in results]))

        if self.task_mode is TaskMode.ANATOMY_4C:
            hd_vals = [
                v
                for r in results
                for v in (r.hd_lv_mm, r.hd_myo_total_mm, r.hd_rv_mm)
                if not math.isnan(v)
            ]
            asd_vals = [
                v
                for r in results
                for v in (r.asd_lv_mm, r.asd_myo_total_mm, r.asd_rv_mm)
                if not math.isnan(v)
            ]
            hd_mean = float(np.mean(hd_vals)) if hd_vals else None
            asd_mean = float(np.mean(asd_vals)) if asd_vals else None
            geom_score = (
                round(0.4 * dsc_board + 0.3 / (1.0 + hd_mean) + 0.3 / (1.0 + asd_mean), 4)
                if hd_mean is not None and asd_mean is not None
                else None
            )
            return {
                "cmr_multi_anat_dsc": round(dsc_board, 4),
                "cmr_multi_anat_hd_mm": round(hd_mean, 4) if hd_mean is not None else None,
                "cmr_multi_anat_asd_mm": round(asd_mean, 4) if asd_mean is not None else None,
                "cmr_multi_anat_geom_score": geom_score,
            }

        # Task 2 (LGE 5 classes): DSC + vPCC de massa fibrótica + RAE de massa fibrótica
        gt_masses = np.asarray([r.gt_scar_mass_g for r in results], dtype=np.float64)
        pred_masses = np.asarray([r.pred_scar_mass_g for r in results], dtype=np.float64)
        vpcc: Optional[float] = None
        if gt_masses.size >= 2 and np.std(gt_masses) > 1e-8 and np.std(pred_masses) > 1e-8:
            r_val, _ = pearsonr(gt_masses, pred_masses)
            if not math.isnan(float(r_val)):
                vpcc = float(r_val)

        rae_vals = [r.scar_mass_rae for r in results if not math.isnan(r.scar_mass_rae)]
        rae_mean = float(np.mean(rae_vals)) if rae_vals else None

        task2_score: Optional[float] = None
        if vpcc is not None and rae_mean is not None:
            task2_score = round(
                0.7 * dsc_board + 0.3 * (0.5 * max(0.0, vpcc) + 0.5 / (1.0 + rae_mean)),
                4,
            )

        return {
            "cmr_multi_dsc_board": round(dsc_board, 4),
            "cmr_multi_scar_mass_vpcc": round(vpcc, 4) if vpcc is not None else None,
            "cmr_multi_scar_mass_rae": round(rae_mean, 4) if rae_mean is not None else None,
            "cmr_multi_task2_score": task2_score,
        }

    @staticmethod
    def _export_csv(results: Sequence[PatientMetrics3D], csv_path: Path) -> None:
        """Exporta tabela por paciente com ponto decimal e separador vírgula."""
        fieldnames = list(asdict(results[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                raw = asdict(row)
                formatted = {
                    k: (
                        ""
                        if isinstance(v, float) and math.isnan(v)
                        else (f"{v:.6f}" if isinstance(v, float) else v)
                    )
                    for k, v in raw.items()
                }
                writer.writerow(formatted)

    def _export_boxplot(self, results: Sequence[PatientMetrics3D], plot_path: Path) -> None:
        """Gera boxplot comparativo do Dice 3D por estrutura cardíaca."""
        if self.task_mode is TaskMode.ANATOMY_4C:
            series_spec = [
                ("VE", [r.dice_lv for r in results if not math.isnan(r.dice_lv)]),
                ("Mio Total\n(Mio+Scar)", [r.dice_myo_total for r in results if not math.isnan(r.dice_myo_total)]),
                ("VD", [r.dice_rv for r in results if not math.isnan(r.dice_rv)]),
            ]
        else:
            series_spec = [
                ("VE", [r.dice_lv for r in results if not math.isnan(r.dice_lv)]),
                ("Mio Total", [r.dice_myo_total for r in results if not math.isnan(r.dice_myo_total)]),
                ("Mio Saudável", [r.dice_myo_healthy for r in results if not math.isnan(r.dice_myo_healthy)]),
                ("Fibrose\n(GT+)", [r.dice_scar for r in results if not math.isnan(r.dice_scar)]),
                ("VD", [r.dice_rv for r in results if not math.isnan(r.dice_rv)]),
            ]

        labels = [name for name, vals in series_spec if vals]
        data = [vals for _, vals in series_spec if vals]
        if not data:
            return

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for idx, vals in enumerate(data, start=1):
            ax.scatter([idx] * len(vals), vals, color="#1f2937", s=26, zorder=3, alpha=0.85)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Coeficiente Dice 3D")
        ax.set_title(
            f"Avaliação 3D — {self.cohort_name} ({self.task_mode.value}, N={len(results)})"
        )
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
            "board_dice_scar",
            "board_dice_mean",
            "hd95_lv_mm",
            "hd95_myo_total_mm",
            "hd95_scar_mm",
            "hd95_rv_mm",
            "hd_lv_mm",
            "hd_myo_total_mm",
            "hd_scar_mm",
            "hd_rv_mm",
            "asd_lv_mm",
            "asd_myo_total_mm",
            "asd_scar_mm",
            "asd_rv_mm",
            "myo_volume_abs_error_ml",
            "scar_volume_abs_error_ml",
            "scar_volume_rel_error_pct",
            "scar_mass_rae",
        ]

        summary: Dict[str, object] = {
            "cohort": self.cohort_name,
            "task_mode": self.task_mode.value,
            "num_patients": len(results),
            "num_with_scar": sum(1 for r in results if r.has_scar),
            "num_healthy": sum(1 for r in results if not r.has_scar),
            "cmr_multi_miccai2026": self._compute_challenge_aggregates(results),
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
                if self.task_mode is TaskMode.FULL_5C and not r.has_scar
            ],
        }
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        boxplot_path = self.output_dir / "dice_boxplot.png"
        self._export_boxplot(results, boxplot_path)

        LOGGER.info(
            "Avaliação 3D (%s | %s) concluída para %d pacientes. Saídas em %s",
            self.cohort_name,
            self.task_mode.value,
            len(results),
            self.output_dir,
        )
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(
        description="Calcula métricas volumétricas 3D clínicas e do challenge CMR-MULTI MICCAI 2026."
    )
    parser.add_argument("--gt-dir", type=Path, required=True, help="Diretório com máscaras Ground Truth.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Diretório com máscaras preditas.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório de saída (CSV, JSON, boxplot).")
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=[m.value for m in TaskMode],
        default=TaskMode.FULL_5C.value,
        help="Esquema de rótulos: `anatomy_4c` (4 classes) ou `full_5c` (5 classes).",
    )
    parser.add_argument("--cohort-name", type=str, default="CMR-MULTI", help="Nome da coorte para títulos/resumo.")
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
        task_mode=TaskMode(args.task_mode),
        cohort_name=args.cohort_name,
    )
    summary = evaluator.run()
    LOGGER.info("Resumo:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
