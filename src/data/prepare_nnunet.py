"""Conversor CMR-MULTI e HU-UFJF para datasets oficiais da nnU-Net v2 (4 ou 5 classes).

Garante:
  - Compatibilidade transparente com `.nii` (descompactado pelo Kaggle) e `.nii.gz`.
  - Preservação integral de matriz `affine` e espaçamento físico `pixdim`.
  - Suporte a dois modos de segmentação:
      * `anatomy_4c` -> `Dataset500_CMRMULTI_LGE_4C` (`0=Bg, 1=LV, 2=MyoTotal, 3=RV`)
      * `full_5c`    -> `Dataset501_CMRMULTI_LGE`    (`0=Bg, 1=LV, 2=Myo, 3=Scar, 4=RV`)
  - Separação rígida sem data leakage dos 7 casos de teste fixos (`SEED=42`) em
    `imagesTs/` + `labelsTs/`, mantendo os 40 restantes em `imagesTr/` + `labelsTr/`.
  - Conversão opcional da coorte clínica privada `robertgvds/hu-ufjf-lge-sax`
    (`imagesHu/` com os 10 rotulados + `patient_20` e `labelsHu/` padronizados
    para o mesmo esquema de 4 ou 5 classes).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

from src.data.cardiac_dataset import (
    CmrMultiCohortLoader,
    HuUfjfCohortLoader,
    LabelStandardizer,
    NiftiVolumeIO,
    TaskMode,
)
from src.data.splits import BenchmarkSplitConfig, StratifiedSplitGenerator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NnUNetDatasetConfig:
    """Especificação imutável de um dataset da nnU-Net v2 por modo de tarefa."""

    task_mode: TaskMode
    dataset_id: int
    dataset_name: str
    labels: Dict[str, int]
    file_ending: str = ".nii.gz"
    channel_names: Dict[str, str] = field(default_factory=lambda: {"0": "LGE_MRI"})

    @classmethod
    def from_task_mode(cls, task_mode: TaskMode) -> "NnUNetDatasetConfig":
        """Constrói configuração oficial (`Dataset500` para 4c ou `Dataset501` para 5c)."""
        if task_mode is TaskMode.ANATOMY_4C:
            return cls(
                task_mode=task_mode,
                dataset_id=500,
                dataset_name="Dataset500_CMRMULTI_LGE_4C",
                labels={"background": 0, "LV": 1, "MyoTotal": 2, "RV": 3},
            )
        return cls(
            task_mode=task_mode,
            dataset_id=501,
            dataset_name="Dataset501_CMRMULTI_LGE",
            labels={"background": 0, "LV": 1, "Myo": 2, "Scar": 3, "RV": 4},
        )


class CmrMultiNnUNetConverter:
    """Converte `cmr-multi-lge-sax` para `nnUNet_raw/DatasetXXX_*` (4c ou 5c)."""

    def __init__(
        self,
        input_dir: Path,
        nnunet_raw_dir: Path,
        task_mode: TaskMode = TaskMode.FULL_5C,
        split_config: Optional[BenchmarkSplitConfig] = None,
    ) -> None:
        self.input_dir = input_dir
        self.nnunet_raw_dir = nnunet_raw_dir
        self.config = NnUNetDatasetConfig.from_task_mode(task_mode)
        self.split_config = split_config or BenchmarkSplitConfig()
        self.dataset_dir = self.nnunet_raw_dir / self.config.dataset_name

    @staticmethod
    def _save_clean_image(src_path: Path, dst_path: Path) -> None:
        """Exporta volume de imagem `float32` preservando `affine` e `pixdim`."""
        arr, nii = NiftiVolumeIO.load_volume(src_path, is_mask=False)
        out_nii = nib.Nifti1Image(arr.astype(np.float32), nii.affine.copy())
        out_nii.set_data_dtype(np.float32)
        out_nii.set_qform(nii.affine, code=1)
        out_nii.set_sform(nii.affine, code=1)
        out_nii.header.set_zooms(tuple(float(z) for z in nii.header.get_zooms()[:3]))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_nii, str(dst_path))

    def build_raw_dataset(self, dry_run: bool = False) -> Dict[str, object]:
        """Converte volumes para `imagesTr`, `labelsTr`, `imagesTs`, `labelsTs` e `dataset.json`."""
        loader = CmrMultiCohortLoader(self.input_dir)
        records = loader.discover_records()
        if not dry_run and len(records) != 47:
            raise ValueError(
                f"Esperados 47 volumes rotulados (40 SAX_TR + 7 SAX_VAL), encontrados {len(records)}."
            )

        scar_map = loader.build_scar_presence_map(records)
        test_set = set(self.split_config.test_cases)

        images_tr = self.dataset_dir / "imagesTr"
        labels_tr = self.dataset_dir / "labelsTr"
        images_ts = self.dataset_dir / "imagesTs"
        labels_ts = self.dataset_dir / "labelsTs"
        for folder in (images_tr, labels_tr, images_ts, labels_ts):
            folder.mkdir(parents=True, exist_ok=True)

        train_scar_map: Dict[str, bool] = {}
        test_scar_map: Dict[str, bool] = {}

        for rec in records:
            assert rec.mask_path is not None
            _, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
            std_mask = LabelStandardizer.standardize_record_mask(rec, task_mode=self.config.task_mode)
            assert std_mask is not None

            is_test = (not dry_run) and (rec.patient_id in test_set)
            if not is_test:
                self._save_clean_image(rec.image_path, images_tr / f"{rec.patient_id}_0000.nii.gz")
                NiftiVolumeIO.save_mask_like(
                    std_mask,
                    img_nii.affine.copy(),
                    img_nii.header["pixdim"].copy(),
                    labels_tr / f"{rec.patient_id}.nii.gz",
                    dtype=np.uint8,
                )
                train_scar_map[rec.patient_id] = scar_map[rec.patient_id]
            else:
                self._save_clean_image(rec.image_path, images_ts / f"{rec.patient_id}_0000.nii.gz")
                NiftiVolumeIO.save_mask_like(
                    std_mask,
                    img_nii.affine.copy(),
                    img_nii.header["pixdim"].copy(),
                    labels_ts / f"{rec.patient_id}.nii.gz",
                    dtype=np.int16,
                )
                test_scar_map[rec.patient_id] = scar_map[rec.patient_id]

        dataset_json = {
            "channel_names": self.config.channel_names,
            "labels": self.config.labels,
            "numTraining": len(train_scar_map),
            "file_ending": self.config.file_ending,
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        (self.dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

        manifest = {
            "dataset_id": self.config.dataset_id,
            "dataset_name": self.config.dataset_name,
            "task_mode": self.config.task_mode.value,
            "dry_run": dry_run,
            "num_train_pool": len(train_scar_map),
            "num_test_fixed": len(test_scar_map),
            "train_scar_map": train_scar_map,
            "test_scar_map": test_scar_map,
        }
        (self.dataset_dir / "case_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        LOGGER.info(
            "Dataset %s (%s) pronto em %s | imagesTr=%d (%d scar) | imagesTs=%d (%d scar)",
            self.config.dataset_name,
            self.config.task_mode.value,
            self.dataset_dir,
            len(train_scar_map),
            sum(train_scar_map.values()),
            len(test_scar_map),
            sum(test_scar_map.values()),
        )
        return manifest

    def inject_splits(self, nnunet_preprocessed_dir: Path, dry_run: bool = False) -> Path:
        """Gera e injeta `splits_final.json` na pasta pré-processada da nnU-Net v2."""
        manifest_path = self.dataset_dir / "case_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifesto não encontrado em {manifest_path}. Execute primeiro a conversão do dataset."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_scar_map: Dict[str, bool] = manifest["train_scar_map"]

        generator = StratifiedSplitGenerator(self.split_config)
        splits = generator.build_nnunet_splits(train_scar_map, dry_run=dry_run)
        out_path = nnunet_preprocessed_dir / self.config.dataset_name / "splits_final.json"
        return generator.save_splits_json(splits, out_path)


class HuUfjfNnUNetConverter:
    """Prepara a coorte privada HU-UFJF no formato de entrada/GT da nnU-Net v2."""

    def __init__(self, hu_input_dir: Path, hu_output_dir: Path, task_mode: TaskMode) -> None:
        self.hu_input_dir = hu_input_dir
        self.hu_output_dir = hu_output_dir
        self.task_mode = task_mode

    def build(self) -> Dict[str, object]:
        """Exporta `imagesHu/*_0000.nii.gz` (todos) e `labelsHu/*.nii.gz` (rotulados padronizados)."""
        loader = HuUfjfCohortLoader(self.hu_input_dir)
        labeled, unlabeled = loader.discover_records()

        images_hu = self.hu_output_dir / "imagesHu"
        labels_hu = self.hu_output_dir / "labelsHu"
        images_hu.mkdir(parents=True, exist_ok=True)
        labels_hu.mkdir(parents=True, exist_ok=True)

        labeled_ids: List[str] = []
        unlabeled_ids: List[str] = []

        for rec in labeled:
            CmrMultiNnUNetConverter._save_clean_image(rec.image_path, images_hu / f"{rec.patient_id}_0000.nii.gz")
            _, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
            std_mask = LabelStandardizer.standardize_record_mask(rec, task_mode=self.task_mode)
            assert std_mask is not None
            NiftiVolumeIO.save_mask_like(
                std_mask,
                img_nii.affine.copy(),
                img_nii.header["pixdim"].copy(),
                labels_hu / f"{rec.patient_id}.nii.gz",
                dtype=np.int16,
            )
            labeled_ids.append(rec.patient_id)

        for rec in unlabeled:
            CmrMultiNnUNetConverter._save_clean_image(rec.image_path, images_hu / f"{rec.patient_id}_0000.nii.gz")
            unlabeled_ids.append(rec.patient_id)

        manifest = {
            "task_mode": self.task_mode.value,
            "labeled_ids": labeled_ids,
            "unlabeled_ids": unlabeled_ids,
        }
        (self.hu_output_dir / "hu_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.info(
            "HU-UFJF preparado para nnU-Net v2 (%s) em %s: %d rotulados | %d sem máscara",
            self.task_mode.value,
            self.hu_output_dir,
            len(labeled_ids),
            len(unlabeled_ids),
        )
        return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(
        description="Converte CMR-MULTI e HU-UFJF para datasets nnU-Net v2 (4c ou 5c) e injeta splits SEED=42."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Diretório raiz CMR-MULTI (ou data_sample/).")
    parser.add_argument("--nnunet-raw", type=Path, required=True, help="Diretório `nnUNet_raw` de destino.")
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=[m.value for m in TaskMode],
        default=TaskMode.FULL_5C.value,
        help="`anatomy_4c` (Dataset500, 4 classes) ou `full_5c` (Dataset501, 5 classes).",
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=None,
        help="Diretório `nnUNet_preprocessed` onde `splits_final.json` será salvo (opcional).",
    )
    parser.add_argument(
        "--hu-input-dir",
        type=Path,
        default=None,
        help="Diretório opcional da coorte clínica HU-UFJF para preparar inferência/avaliação externa.",
    )
    parser.add_argument(
        "--hu-output-dir",
        type=Path,
        default=None,
        help="Diretório de saída para `imagesHu/` e `labelsHu/` padronizados.",
    )
    parser.add_argument(
        "--inject-splits-only",
        action="store_true",
        help="Apenas gera/injeta `splits_final.json` usando `case_manifest.json` existente.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Permite validação rápida em subconjunto mínimo (ex.: `data_sample/`).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI do conversor nnU-Net v2."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    task_mode = TaskMode(args.task_mode)

    converter = CmrMultiNnUNetConverter(
        input_dir=args.input_dir.resolve(),
        nnunet_raw_dir=args.nnunet_raw.resolve(),
        task_mode=task_mode,
    )

    if not args.inject_splits_only:
        converter.build_raw_dataset(dry_run=args.dry_run)

    if args.nnunet_preprocessed is not None:
        converter.inject_splits(args.nnunet_preprocessed.resolve(), dry_run=args.dry_run)

    if args.hu_input_dir is not None and args.hu_output_dir is not None and not args.inject_splits_only:
        HuUfjfNnUNetConverter(
            hu_input_dir=args.hu_input_dir.resolve(),
            hu_output_dir=args.hu_output_dir.resolve(),
            task_mode=task_mode,
        ).build()

    return 0


if __name__ == "__main__":
    sys.exit(main())
