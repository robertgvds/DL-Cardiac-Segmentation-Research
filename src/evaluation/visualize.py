"""Visualização comparativa `LGE | Ground Truth | Predição` (4 ou 5 classes) em LGE-CMR.

Seleciona automaticamente a fatia Z de maior área de fibrose no Ground Truth
(em 5 classes) ou de maior área de miocárdio (em 4 classes / casos saudáveis)
e gera figuras individuais + painel consolidado `overview_panel.png`. Também
suporta visualização de casos sem GT (ex.: Paciente 20 do HU-UFJF).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from src.data.cardiac_dataset import TaskMode

LOGGER = logging.getLogger(__name__)

# Paletas RGBA padronizadas por modo de tarefa
PALETTES_RGBA: Dict[TaskMode, Dict[int, Tuple[float, float, float, float]]] = {
    TaskMode.ANATOMY_4C: {
        1: (0.18, 0.55, 0.96, 0.55),  # VE (Azul)
        2: (0.16, 0.78, 0.45, 0.55),  # Miocárdio Total (Verde)
        3: (0.96, 0.78, 0.16, 0.55),  # VD (Amarelo)
    },
    TaskMode.FULL_5C: {
        1: (0.18, 0.55, 0.96, 0.55),  # VE (Azul)
        2: (0.16, 0.78, 0.45, 0.55),  # Miocárdio Saudável (Verde)
        3: (0.94, 0.22, 0.22, 0.85),  # Fibrose / Scar (Vermelho)
        4: (0.96, 0.78, 0.16, 0.55),  # VD (Amarelo)
    },
}

LABELS_PT: Dict[TaskMode, Dict[int, str]] = {
    TaskMode.ANATOMY_4C: {
        1: "VE",
        2: "Mio Total (Mio+Scar)",
        3: "VD",
    },
    TaskMode.FULL_5C: {
        1: "VE",
        2: "Mio Saudável",
        3: "Fibrose (Scar)",
        4: "VD",
    },
}


class SegmentationPanelVisualizer:
    """Gera painéis qualitativos comparando LGE original, GT e Predição."""

    def __init__(
        self,
        image_dir: Path,
        gt_dir: Optional[Path],
        pred_dir: Path,
        output_dir: Path,
        task_mode: TaskMode = TaskMode.FULL_5C,
        model_label: str = "Predição",
    ) -> None:
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.output_dir = output_dir
        self.task_mode = task_mode
        self.model_label = model_label
        self.palette = PALETTES_RGBA[task_mode]
        self.class_labels = LABELS_PT[task_mode]

    @staticmethod
    def _normalize_case_id(filename: str) -> str:
        """Remove `.nii.gz`, `.nii` e eventual sufixo `_0000` da nnU-Net v2."""
        stem = (
            filename[:-7]
            if filename.endswith(".nii.gz")
            else (filename[:-4] if filename.endswith(".nii") else Path(filename).stem)
        )
        if stem.endswith("_0000"):
            stem = stem[:-5]
        return stem

    @staticmethod
    def _index_volumes(directory: Path) -> Dict[str, Path]:
        """Indexa volumes NIfTI por identificador do paciente."""
        mapping: Dict[str, Path] = {}
        for p in sorted(directory.iterdir()):
            if p.is_file() and not p.name.startswith(".") and (
                p.name.endswith(".nii.gz") or p.name.endswith(".nii")
            ):
                mapping[SegmentationPanelVisualizer._normalize_case_id(p.name)] = p
        return mapping

    def _select_representative_slice(self, ref_vol: np.ndarray) -> Tuple[int, str]:
        """Escolhe fatia Z mais informativa (maior scar em 5c ou maior miocárdio)."""
        if self.task_mode is TaskMode.FULL_5C:
            scar_counts = [int(np.count_nonzero(ref_vol[:, :, z] == 3)) for z in range(ref_vol.shape[2])]
            if max(scar_counts) > 0:
                return int(np.argmax(scar_counts)), "max_scar"
        myo_counts = [int(np.count_nonzero(ref_vol[:, :, z] == 2)) for z in range(ref_vol.shape[2])]
        if max(myo_counts) > 0:
            return int(np.argmax(myo_counts)), "max_myo"
        return ref_vol.shape[2] // 2, "mid_slice"

    @staticmethod
    def _normalize_lge_slice(slc: np.ndarray) -> np.ndarray:
        """Normaliza contraste por percentis (1–99%) apenas para visualização nítida."""
        p1, p99 = np.percentile(slc, (1.0, 99.0))
        return np.clip((slc - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    def _build_rgba_overlay(self, mask_2d: np.ndarray) -> np.ndarray:
        """Converte máscara inteira 2D `(H, W)` em camada RGBA transparente."""
        rgba = np.zeros((*mask_2d.shape, 4), dtype=np.float32)
        for cls_id, color in self.palette.items():
            rgba[mask_2d == cls_id] = color
        return rgba

    def _legend_handles(self) -> List[mpatches.Patch]:
        """Cria legenda padronizada conforme o modo de tarefa (4 ou 5 classes)."""
        return [
            mpatches.Patch(color=self.palette[cid][:3], label=self.class_labels[cid])
            for cid in sorted(self.palette.keys())
        ]

    def _render_labeled_row(
        self,
        axes_row: Sequence[plt.Axes],
        patient_id: str,
        img_2d: np.ndarray,
        gt_2d: np.ndarray,
        pred_2d: np.ndarray,
        slice_idx: int,
        reason: str,
    ) -> None:
        """Desenha trio `LGE | GT | Predição` em uma linha de subplots."""
        img_disp = self._normalize_lge_slice(img_2d)

        axes_row[0].imshow(img_disp, cmap="gray")
        axes_row[0].set_title(f"{patient_id} (z={slice_idx}, {reason})\nLGE Original", fontsize=9)

        axes_row[1].imshow(img_disp, cmap="gray")
        axes_row[1].imshow(self._build_rgba_overlay(gt_2d))
        axes_row[1].set_title(f"Ground Truth ({self.task_mode.num_classes}c)", fontsize=9)

        axes_row[2].imshow(img_disp, cmap="gray")
        axes_row[2].imshow(self._build_rgba_overlay(pred_2d))
        axes_row[2].set_title(self.model_label, fontsize=9)

        for ax in axes_row:
            ax.axis("off")

    def render_unlabeled_case(self, patient_id: str, image_path: Path, pred_path: Path) -> Path:
        """Gera figura `LGE | Predição` para casos clínicos sem GT (ex.: HU-UFJF patient_20)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        img_vol = np.squeeze(nib.load(str(image_path)).get_fdata().astype(np.float32))
        pred_vol = np.rint(np.squeeze(nib.load(str(pred_path)).get_fdata())).astype(np.int16)
        z_idx, reason = self._select_representative_slice(pred_vol)
        img_disp = self._normalize_lge_slice(img_vol[:, :, z_idx])

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
        axes[0].imshow(img_disp, cmap="gray")
        axes[0].set_title(f"{patient_id} (z={z_idx}, {reason})\nLGE Original (Sem GT)", fontsize=9)
        axes[1].imshow(img_disp, cmap="gray")
        axes[1].imshow(self._build_rgba_overlay(pred_vol[:, :, z_idx]))
        axes[1].set_title(self.model_label, fontsize=9)
        for ax in axes:
            ax.axis("off")

        fig.legend(handles=self._legend_handles(), loc="lower center", ncol=len(self.palette), frameon=False)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        out_path = self.output_dir / f"panel_{patient_id}_unlabeled.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        return out_path

    def run(self) -> List[Path]:
        """Gera painéis individuais e painel consolidado para todos os casos rotulados em comum."""
        if self.gt_dir is None:
            raise ValueError("`gt_dir` é obrigatório para `run()`; use `render_unlabeled_case()` para casos sem GT.")

        img_map = self._index_volumes(self.image_dir)
        gt_map = self._index_volumes(self.gt_dir)
        pred_map = self._index_volumes(self.pred_dir)

        common_ids = sorted(set(img_map) & set(gt_map) & set(pred_map))
        if not common_ids:
            raise FileNotFoundError("Nenhum paciente em comum entre image_dir, gt_dir e pred_dir.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []
        cached_rows: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int, str]] = []

        for cid in common_ids:
            img_vol = np.squeeze(nib.load(str(img_map[cid])).get_fdata().astype(np.float32))
            gt_vol = np.rint(np.squeeze(nib.load(str(gt_map[cid])).get_fdata())).astype(np.int16)
            pred_vol = np.rint(np.squeeze(nib.load(str(pred_map[cid])).get_fdata())).astype(np.int16)

            z_idx, reason = self._select_representative_slice(gt_vol)
            row_data = (cid, img_vol[:, :, z_idx], gt_vol[:, :, z_idx], pred_vol[:, :, z_idx], z_idx, reason)
            cached_rows.append(row_data)

            fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
            self._render_labeled_row(axes, *row_data)
            fig.legend(handles=self._legend_handles(), loc="lower center", ncol=len(self.palette), frameon=False)
            fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
            out_file = self.output_dir / f"panel_{cid}.png"
            fig.savefig(str(out_file), dpi=150)
            plt.close(fig)
            saved_paths.append(out_file)

        n_rows = len(cached_rows)
        fig_all, axes_all = plt.subplots(n_rows, 3, figsize=(11.0, 3.3 * n_rows), squeeze=False)
        for row_idx, row_data in enumerate(cached_rows):
            self._render_labeled_row(axes_all[row_idx], *row_data)
        fig_all.legend(handles=self._legend_handles(), loc="lower center", ncol=len(self.palette), frameon=False)
        fig_all.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
        overview_path = self.output_dir / "overview_panel.png"
        fig_all.savefig(str(overview_path), dpi=140)
        plt.close(fig_all)
        saved_paths.append(overview_path)

        LOGGER.info("Geradas %d figuras comparativas em %s", len(saved_paths), self.output_dir)
        return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(description="Gera figuras comparativas LGE | GT | Pred (4 ou 5 classes).")
    parser.add_argument("--image-dir", type=Path, required=True, help="Diretório com imagens LGE.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Diretório com máscaras Ground Truth.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Diretório com máscaras preditas.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório de saída das figuras PNG.")
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=[m.value for m in TaskMode],
        default=TaskMode.FULL_5C.value,
        help="Esquema de rótulos: `anatomy_4c` ou `full_5c`.",
    )
    parser.add_argument("--model-label", type=str, default="Predição", help="Título da coluna de predição.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI do visualizador."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    visualizer = SegmentationPanelVisualizer(
        image_dir=args.image_dir.resolve(),
        gt_dir=args.gt_dir.resolve(),
        pred_dir=args.pred_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        task_mode=TaskMode(args.task_mode),
        model_label=args.model_label,
    )
    visualizer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
