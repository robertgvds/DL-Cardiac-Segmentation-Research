"""Pós-processamento anatômico 3D para máscaras NIfTI preditas em LGE-CMR.

Remove falsos positivos torácicos distantes (crítico em exames de tórax inteiro
como os da coorte HU-UFJF) preservando o maior componente conexo cardíaco 3D com
ponte leve no plano XY (2 px) para unir VE+Miocárdio+VD através de eventuais
lacunas finas do septo. Em 5 classes, remove também micro-ruídos isolados de
fibrose desconectados do miocárdio/VE.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label

from src.data.cardiac_dataset import NiftiVolumeIO, TaskMode

LOGGER = logging.getLogger(__name__)


class CardiacMaskPostProcessor:
    """Aplica filtragem anatômica 3D em volumes ou diretórios de predições NIfTI."""

    def __init__(self, task_mode: TaskMode, min_isolated_scar_voxels: int = 10) -> None:
        self.task_mode = task_mode
        self.min_isolated_scar_voxels = int(min_isolated_scar_voxels)

    def apply(self, pred_mask_3d: np.ndarray) -> np.ndarray:
        """Executa LCC 3D cardíaco e limpeza de micro-componentes isolados de fibrose."""
        cleaned = np.rint(pred_mask_3d).astype(np.int16, copy=True)
        fg = cleaned > 0
        if not np.any(fg):
            return cleaned

        struct_26 = np.ones((3, 3, 3), dtype=np.int32)
        xy_struct = np.zeros((3, 3, 3), dtype=bool)
        xy_struct[:, :, 1] = True
        fg_grouped = binary_dilation(fg, structure=xy_struct, iterations=2)
        labeled_fg, num_fg = cc_label(fg_grouped, structure=struct_26)
        if num_fg > 1:
            best_group = max(
                range(1, num_fg + 1),
                key=lambda gid: int(np.count_nonzero(fg & (labeled_fg == gid))),
            )
            cleaned[labeled_fg != best_group] = 0

        if self.task_mode is TaskMode.FULL_5C:
            scar_mask = cleaned == 3
            if np.any(scar_mask):
                anchor_region = binary_dilation((cleaned == 1) | (cleaned == 2), structure=struct_26)
                labeled_scar, num_scar = cc_label(scar_mask, structure=struct_26)
                for comp_id in range(1, num_scar + 1):
                    comp = labeled_scar == comp_id
                    voxels = int(np.count_nonzero(comp))
                    touches_myo = bool(np.any(comp & anchor_region))
                    if voxels < self.min_isolated_scar_voxels and not touches_myo:
                        cleaned[comp] = 0
                    elif voxels < 6 and touches_myo and np.count_nonzero(scar_mask) < self.min_isolated_scar_voxels:
                        # Remove resíduo minúsculo isolado em casos saudáveis (ex.: <=5 voxels)
                        cleaned[comp] = 2

        return cleaned

    def process_directory(self, pred_dir: Path, output_dir: Path, ref_image_dir: Optional[Path] = None) -> int:
        """Pós-processa todos os NIfTIs em `pred_dir` salvando em `int16` com cabeçalho fiel."""
        output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for pred_path in sorted(pred_dir.iterdir()):
            if not NiftiVolumeIO.is_nifti_file(pred_path):
                continue
            case_id = NiftiVolumeIO.strip_suffix(pred_path.name)
            pred_arr, pred_nii = NiftiVolumeIO.load_volume(pred_path, is_mask=True)
            cleaned = self.apply(pred_arr)

            ref_nii = pred_nii
            if ref_image_dir is not None:
                cand = NiftiVolumeIO.find_nifti_by_stem(ref_image_dir, f"{case_id}_0000")
                if cand is None:
                    cand = NiftiVolumeIO.find_nifti_by_stem(ref_image_dir, case_id)
                if cand is not None:
                    ref_nii = nib.load(str(cand))

            NiftiVolumeIO.save_mask_like(
                cleaned,
                ref_nii.affine.copy(),
                ref_nii.header["pixdim"].copy(),
                output_dir / f"{case_id}.nii.gz",
                dtype=np.int16,
            )
            count += 1

        LOGGER.info("Pós-processamento 3D (%s) concluído para %d volumes em %s", self.task_mode.value, count, output_dir)
        return count


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(description="Aplica pós-processamento anatômico 3D em predições NIfTI.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Diretório com predições brutas NIfTI.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório de saída das máscaras limpas.")
    parser.add_argument("--ref-image-dir", type=Path, default=None, help="Diretório de imagens de referência (affine/pixdim).")
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=[m.value for m in TaskMode],
        required=True,
        help="Esquema de rótulos: `anatomy_4c` ou `full_5c`.",
    )
    parser.add_argument("--min-scar-voxels", type=int, default=10, help="Limiar mínimo de voxels para micro-ruído de fibrose.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI do pós-processador."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    processor = CardiacMaskPostProcessor(
        task_mode=TaskMode(args.task_mode),
        min_isolated_scar_voxels=args.min_scar_voxels,
    )
    processor.process_directory(
        pred_dir=args.pred_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        ref_image_dir=args.ref_image_dir.resolve() if args.ref_image_dir is not None else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
