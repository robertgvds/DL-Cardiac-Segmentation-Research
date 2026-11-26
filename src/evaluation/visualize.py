"""Visualização comparativa `LGE | Ground Truth | Predição` (5 classes) em LGE-CMR.

Seleciona automaticamente a fatia Z de maior área de fibrose no Ground Truth
(ou a fatia de maior área de miocárdio em casos saudáveis) e gera figuras
individuais por paciente + painel consolidado `overview_test_panel.png`.
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

LOGGER = logging.getLogger(__name__)

# Paleta RGBA padronizada para as 5 classes CMR-MULTI / Dataset501
CLASS_COLORS_RGBA: Dict[int, Tuple[float, float, float, float]] = {
    1: (0.18, 0.55, 0.96, 0.55),  # VE (Azul)
    2: (0.16, 0.78, 0.45, 0.55),  # Miocárdio Saudável (Verde)
    3: (0.94, 0.22, 0.22, 0.85),  # Fibrose / Scar (Vermelho)
    4: (0.96, 0.78, 0.16, 0.55),  # VD (Amarelo)
}
CLASS_NAMES: Dict[int, str] = {
    1: "VE",
    2: "Mio Saudável",
    3: "Fibrose (Scar)",
    4: "VD",
}


class SegmentationPanelVisualizer:
    """Gera painéis qualitativos comparando LGE original, GT e Predição."""

    def __init__(self, image_dir: Path, gt_dir: Path, pred_dir: Path, output_dir: Path) -> None:
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.output_dir = output_dir

    @staticmethod
    def _normalize_case_id(filename: str) -> str:
        """Remove `.nii.gz`, `.nii` e sufixo de canal `_0000` da nnU-Net v2."""
        stem = filename[:-7] if filename.endswith(".nii.gz") else (filename[:-4] if filename.endswith(".nii") else Path(filename).stem)
        if stem.endswith("_0000"):
            stem = stem[:-5]
        return stem

    @staticmethod
    def _index_volumes(directory: Path) -> Dict[str, Path]:
        """Indexa volumes NIfTI por identificador do paciente."""
        mapping: Dict[str, Path] = {}
        for p in sorted(directory.iterdir()):
            if p.is_file() and not p.name.startswith(".") and (p.name.endswith(".nii.gz") or p.name.endswith(".nii")):
                mapping[SegmentationPanelVisualizer._normalize_case_id(p.name)] = p
        return mapping

    @staticmethod
    def _select_representative_slice(gt_vol: np.ndarray) -> Tuple[int, str]:
        """Escolhe a fatia Z com maior área de fibrose ou, na ausência, maior miocárdio."""
        scar_counts = [int(np.count_nonzero(gt_vol[:, :, z] == 3)) for z in range(gt_vol.shape[2])]
        if max(scar_counts) > 0:
            return int(np.argmax(scar_counts)), "max_scar"
        myo_counts = [int(np.count_nonzero(gt_vol[:, :, z] == 2)) for z in range(gt_vol.shape[2])]
        if max(myo_counts) > 0:
            return int(np.argmax(myo_counts)), "max_myo"
        return gt_vol.shape[2] // 2, "mid_slice"

    @staticmethod
    def _normalize_lge_slice(slc: np.ndarray) -> np.ndarray:
        """Normaliza contraste por percentis (1–99%) apenas para visualização clínica nítida."""
        p1, p99 = np.percentile(slc, (1.0, 99.0))
        return np.clip((slc - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    @staticmethod
    def _build_rgba_overlay(mask_2d: np.ndarray) -> np.ndarray:
        """Converte máscara inteira 2D `(H, W)` em camada RGBA transparente."""
        rgba = np.zeros((*mask_2d.shape, 4), dtype=np.float32)
        for cls_id, color in CLASS_COLORS_RGBA.items():
            rgba[mask_2d == cls_id] = color
        return rgba

    @staticmethod
    def _legend_handles() -> List[mpatches.Patch]:
        """Cria legenda padronizada das 4 estruturas anatômicas/patológicas."""
        return [
            mpatches.Patch(color=CLASS_COLORS_RGBA[cid][:3], label=CLASS_NAMES[cid])
            for cid in (1, 2, 3, 4)
        ]

    def _render_row(
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
        axes_row[1].set_title("Ground Truth (5 classes)", fontsize=9)

        axes_row[2].imshow(img_disp, cmap="gray")
        axes_row[2].imshow(self._build_rgba_overlay(pred_2d))
        axes_row[2].set_title("Predição nnU-Net v2", fontsize=9)

        for ax in axes_row:
            ax.axis("off")

    def run(self) -> List[Path]:
        """Gera painéis individuais e o painel consolidado para todos os pacientes em comum."""
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
            self._render_row(axes, *row_data)
            fig.legend(handles=self._legend_handles(), loc="lower center", ncol=4, frameon=False)
            fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
            out_file = self.output_dir / f"panel_{cid}.png"
            fig.savefig(str(out_file), dpi=150)
            plt.close(fig)
            saved_paths.append(out_file)

        # Painel consolidado com todos os casos avaliados
        n_rows = len(cached_rows)
        fig_all, axes_all = plt.subplots(n_rows, 3, figsize=(11.0, 3.5 * n_rows), squeeze=False)
        for row_idx, row_data in enumerate(cached_rows):
            self._render_row(axes_all[row_idx], *row_data)
        fig_all.legend(handles=self._legend_handles(), loc="lower center", ncol=4, frameon=False)
        fig_all.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
        overview_path = self.output_dir / "overview_test_panel.png"
        fig_all.savefig(str(overview_path), dpi=140)
        plt.close(fig_all)
        saved_paths.append(overview_path)

        LOGGER.info("Geradas %d figuras comparativas em %s", len(saved_paths), self.output_dir)
        return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(description="Gera figuras comparativas LGE | GT | Pred (5 classes).")
    parser.add_argument("--image-dir", type=Path, required=True, help="Diretório com imagens LGE (`*_0000.nii.gz`).")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Diretório com máscaras Ground Truth.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Diretório com máscaras preditas.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório de saída das figuras PNG.")
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
    )
    visualizer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
