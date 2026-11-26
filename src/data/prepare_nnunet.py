"""Conversor CMR-MULTI LGE SAX para o formato oficial `Dataset501_CMRMULTI_LGE` da nnU-Net v2.

Garante:
  - Compatibilidade transparente com `.nii` (descompactado pelo Kaggle) e `.nii.gz`.
  - Preservação integral de matriz `affine` e espaçamento físico `pixdim`.
  - Separação rígida sem data leakage dos 7 casos de teste fixos (`SEED=42`) em
    `imagesTs/` + `labelsTs/`, mantendo os 40 restantes em `imagesTr/` + `labelsTr/`.
  - Geração de `dataset.json` (padrão nnU-Net v2) e injeção de `splits_final.json`
    em `nnUNet_preprocessed/` com Fold 0 espelhando exatamente os baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import nibabel as nib
import numpy as np

from src.data.splits import BenchmarkSplitConfig, StratifiedSplitGenerator

LOGGER = logging.getLogger(__name__)

EXPECTED_LABELS: Set[int] = {0, 1, 2, 3, 4}
SCAR_LABEL_ID = 3


@dataclass(frozen=True)
class Dataset501Config:
    """Especificação imutável do `Dataset501_CMRMULTI_LGE` na nnU-Net v2."""

    dataset_id: int = 501
    dataset_name: str = "Dataset501_CMRMULTI_LGE"
    file_ending: str = ".nii.gz"
    channel_names: Dict[str, str] = field(default_factory=lambda: {"0": "LGE_MRI"})
    labels: Dict[str, int] = field(
        default_factory=lambda: {
            "background": 0,
            "LV": 1,
            "Myo": 2,
            "Scar": 3,
            "RV": 4,
        }
    )


@dataclass(frozen=True)
class CmrMultiVolumeRecord:
    """Metadados validados de um volume rotulado do CMR-MULTI."""

    case_id: str
    image_path: Path
    mask_path: Path
    has_scar: bool
    split_role: str  # "train_pool" ou "test_fixed"


class CmrMultiNnUNetConverter:
    """Converte o dataset `cmr-multi-lge-sax` para a árvore `nnUNet_raw/Dataset501_CMRMULTI_LGE`."""

    def __init__(
        self,
        input_dir: Path,
        nnunet_raw_dir: Path,
        config: Optional[Dataset501Config] = None,
        split_config: Optional[BenchmarkSplitConfig] = None,
    ) -> None:
        self.input_dir = input_dir
        self.nnunet_raw_dir = nnunet_raw_dir
        self.config = config or Dataset501Config()
        self.split_config = split_config or BenchmarkSplitConfig()
        self.dataset_dir = self.nnunet_raw_dir / self.config.dataset_name

    @staticmethod
    def _strip_nifti_suffix(filename: str) -> str:
        """Remove extensões `.nii.gz` ou `.nii` preservando o identificador do caso."""
        if filename.endswith(".nii.gz"):
            return filename[:-7]
        if filename.endswith(".nii"):
            return filename[:-4]
        return Path(filename).stem

    @staticmethod
    def _is_nifti_file(path: Path) -> bool:
        """Verifica se o caminho é um arquivo NIfTI real (ignora ocultos e diretórios)."""
        return (
            path.is_file()
            and not path.name.startswith(".")
            and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))
        )

    def discover_labeled_pairs(self, dry_run: bool = False) -> List[CmrMultiVolumeRecord]:
        """Localiza todos os pares `(imagem, máscara)` em `SAX_TR` e `SAX_VAL`.

        Args:
            dry_run: Se True, relaxa a exigência de 47 volumes para permitir
                execução em `data_sample/` (2 volumes).

        Returns:
            Lista ordenada de registros validados com indicação de presença de fibrose.
        """
        pair_map: Dict[str, Tuple[Path, Path]] = {}

        for subdir in sorted(self.input_dir.rglob("*")):
            if not subdir.is_dir():
                continue
            if subdir.name.upper() not in ("SAX_TR", "SAX_VAL"):
                continue
            img_dir = subdir / "image"
            msk_dir = subdir / "anno"
            if not (img_dir.is_dir() and msk_dir.is_dir()):
                continue

            # Indexa máscaras por case_id (aceitando .nii e .nii.gz de forma independente)
            mask_by_id: Dict[str, Path] = {
                self._strip_nifti_suffix(p.name): p
                for p in sorted(msk_dir.iterdir())
                if self._is_nifti_file(p)
            }
            for img_path in sorted(img_dir.iterdir()):
                if not self._is_nifti_file(img_path):
                    continue
                case_id = self._strip_nifti_suffix(img_path.name)
                if case_id in mask_by_id:
                    pair_map[case_id] = (img_path, mask_by_id[case_id])

        if not pair_map:
            raise FileNotFoundError(
                f"Nenhum par (SAX_TR/SAX_VAL image+anno) encontrado em {self.input_dir}"
            )

        if not dry_run and len(pair_map) != 47:
            raise ValueError(
                f"Esperados 47 volumes rotulados (40 SAX_TR + 7 SAX_VAL), encontrados {len(pair_map)}."
            )

        test_set = set(self.split_config.test_cases)
        records: List[CmrMultiVolumeRecord] = []
        for case_id in sorted(pair_map):
            img_path, msk_path = pair_map[case_id]
            mask_data = np.rint(np.squeeze(nib.load(str(msk_path)).get_fdata())).astype(np.int16)
            uniq = {int(v) for v in np.unique(mask_data)}
            if not uniq.issubset(EXPECTED_LABELS):
                raise ValueError(f"Rótulos fora de {{0..4}} em {case_id}: {sorted(uniq)}")

            role = "test_fixed" if (not dry_run and case_id in test_set) else "train_pool"
            records.append(
                CmrMultiVolumeRecord(
                    case_id=case_id,
                    image_path=img_path,
                    mask_path=msk_path,
                    has_scar=(SCAR_LABEL_ID in uniq),
                    split_role=role,
                )
            )

        return records

    @staticmethod
    def _export_clean_nifti(src_path: Path, dst_path: Path, is_mask: bool, mask_dtype: type = np.uint8) -> None:
        """Carrega volume NIfTI, remove eixo singleton e salva preservando affine e pixdim."""
        nii = nib.load(str(src_path))
        arr = np.squeeze(nii.get_fdata())
        if arr.ndim != 3:
            raise ValueError(f"Volume esperado 3D após squeeze em {src_path}, obtido shape {arr.shape}")

        affine = nii.affine.copy()
        zooms = tuple(float(z) for z in nii.header.get_zooms()[:3])
        if is_mask:
            out_arr = np.rint(arr).astype(mask_dtype)
            out_nii = nib.Nifti1Image(out_arr, affine)
            out_nii.set_data_dtype(mask_dtype)
        else:
            out_arr = arr.astype(np.float32)
            out_nii = nib.Nifti1Image(out_arr, affine)
            out_nii.set_data_dtype(np.float32)

        out_nii.set_qform(affine, code=1)
        out_nii.set_sform(affine, code=1)
        out_nii.header.set_zooms(zooms)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_nii, str(dst_path))

    def build_raw_dataset(self, dry_run: bool = False) -> Dict[str, object]:
        """Converte volumes para `imagesTr`, `labelsTr`, `imagesTs`, `labelsTs` e `dataset.json`."""
        records = self.discover_labeled_pairs(dry_run=dry_run)

        images_tr = self.dataset_dir / "imagesTr"
        labels_tr = self.dataset_dir / "labelsTr"
        images_ts = self.dataset_dir / "imagesTs"
        labels_ts = self.dataset_dir / "labelsTs"
        for folder in (images_tr, labels_tr, images_ts, labels_ts):
            folder.mkdir(parents=True, exist_ok=True)

        train_scar_map: Dict[str, bool] = {}
        test_scar_map: Dict[str, bool] = {}

        for rec in records:
            if rec.split_role == "train_pool":
                self._export_clean_nifti(
                    rec.image_path, images_tr / f"{rec.case_id}_0000.nii.gz", is_mask=False
                )
                # nnUNetv2 exige rótulos inteiros consecutivos; uint8 reduz espaço e passa integrity check
                self._export_clean_nifti(
                    rec.mask_path, labels_tr / f"{rec.case_id}.nii.gz", is_mask=True, mask_dtype=np.uint8
                )
                train_scar_map[rec.case_id] = rec.has_scar
            else:
                self._export_clean_nifti(
                    rec.image_path, images_ts / f"{rec.case_id}_0000.nii.gz", is_mask=False
                )
                # Salva GT de teste em int16 para compatibilidade direta com 3D Slicer e métricas 3D
                self._export_clean_nifti(
                    rec.mask_path, labels_ts / f"{rec.case_id}.nii.gz", is_mask=True, mask_dtype=np.int16
                )
                test_scar_map[rec.case_id] = rec.has_scar

        dataset_json = {
            "channel_names": self.config.channel_names,
            "labels": self.config.labels,
            "numTraining": len(train_scar_map),
            "file_ending": self.config.file_ending,
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        (self.dataset_dir / "dataset.json").write_text(
            json.dumps(dataset_json, indent=2), encoding="utf-8"
        )

        manifest = {
            "dataset_name": self.config.dataset_name,
            "dry_run": dry_run,
            "num_train_pool": len(train_scar_map),
            "num_test_fixed": len(test_scar_map),
            "train_scar_map": train_scar_map,
            "test_scar_map": test_scar_map,
        }
        (self.dataset_dir / "case_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        LOGGER.info(
            "Dataset %s pronto em %s | imagesTr=%d (%d com fibrose) | imagesTs=%d (%d com fibrose)",
            self.config.dataset_name,
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(
        description="Converte CMR-MULTI LGE SAX para Dataset501 da nnU-Net v2 e injeta splits estratificados."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Diretório raiz contendo `SAX_TR/` e `SAX_VAL/` (ex.: /kaggle/input/cmr-multi-lge-sax ou data_sample/).",
    )
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        required=True,
        help="Diretório `nnUNet_raw` de destino.",
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=None,
        help="Diretório `nnUNet_preprocessed` onde `splits_final.json` será salvo (opcional).",
    )
    parser.add_argument(
        "--inject-splits-only",
        action="store_true",
        help="Apenas gera/injeta `splits_final.json` usando o `case_manifest.json` já existente.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Permite executar validação rápida em subconjunto mínimo (ex.: `data_sample/` com 2 volumes).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI do módulo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    converter = CmrMultiNnUNetConverter(
        input_dir=args.input_dir.resolve(),
        nnunet_raw_dir=args.nnunet_raw.resolve(),
    )

    if not args.inject_splits_only:
        converter.build_raw_dataset(dry_run=args.dry_run)

    if args.nnunet_preprocessed is not None:
        converter.inject_splits(args.nnunet_preprocessed.resolve(), dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
