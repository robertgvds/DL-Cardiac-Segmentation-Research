"""Empacotamento, validação e upload automatizado dos datasets privados no Kaggle.

Este módulo prepara em diretório temporário de staging, valida a integridade
física/anatômica (affine, pixdim, dtype e rótulos) e publica como Private Datasets
na conta Kaggle configurada:
  1. robertgvds/cmr-multi-lge-sax
  2. robertgvds/emidec-lge-sax
  3. robertgvds/hu-ufjf-lge-sax (NIfTIs puros + MATs anonimizados + manifesto)
  4. robertgvds/hu-ufjf-dicom-archive (DICOMs anonimizados estruturados em nuvem)

Também gera localmente a amostra leve `data_sample/` (2 volumes do CMR-MULTI)
para testes rápidos em CPU (--dry-run).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import nibabel as nib
import numpy as np
import pydicom
import scipy.io as sio
import SimpleITK as sitk
from skimage.draw import polygon

LOGGER = logging.getLogger("upload_kaggle_datasets")

# Constantes de configuração e convenção de rótulos
KAGGLE_USERNAME = "robertgvds"
EXPECTED_CMR_LABELS: Set[int] = {0, 1, 2, 3, 4}
EXPECTED_EMIDEC_LABELS: Set[int] = {0, 1, 2, 3, 4}

# Rótulos anatômicos originais (4 classes) e padrão unificado CMR-MULTI (5 classes)
LABEL_BG = 0
LABEL_LV = 1
LABEL_MYO = 2
LABEL_RV_4CLASS = 3
LABEL_SCAR_5CLASS = 3
LABEL_RV_5CLASS = 4
LABEL_SCAR_BIN = 1

# Extensões não-DICOM descartadas durante a limpeza do arquivo HU-UFJF
IGNORED_DICOM_EXTENSIONS: Set[str] = {
    ".exe", ".dll", ".inf", ".txt", ".html", ".htm", ".js", ".css",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".xml", ".cab", ".bin",
    ".avi", ".mp4", ".mov", ".pdf", ".cache", ".etag", ".ds_store",
}

# Tags DICOM identificáveis (LGPD / sigilo hospitalar HU-UFJF)
PII_DICOM_TAGS: Tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
    "AccessionNumber",
    "StudyID",
)

# Campos identificáveis no struct `info` dos arquivos .mat do Medviso Segment
PII_MAT_INFO_FIELDS: Tuple[str, ...] = (
    "Name",
    "ID",
    "BirthDate",
    "AcquisitionDate",
    "Institution",
    "AccessionNumber",
    "StudyUID",
    "StudyID",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Especificação imutável de um dataset privado no Kaggle."""

    key: str
    slug: str
    title: str
    subtitle: str
    description: str
    license_name: str = "other"
    is_private: bool = True

    @property
    def kaggle_id(self) -> str:
        """Retorna o identificador completo `<usuario>/<slug>` no Kaggle."""
        return f"{KAGGLE_USERNAME}/{self.slug}"


@dataclass(frozen=True)
class HuMatSource:
    """Referência de localização para um arquivo `.mat` oficial de paciente HU-UFJF."""

    patient_id: int
    source_type: str  # "file" ou "zip"
    source_path: Path
    zip_member: Optional[str] = None
    version_note: str = ""


@dataclass
class PackagingReport:
    """Resumo estruturado da preparação e validação de um dataset."""

    spec: DatasetSpec
    staging_dir: Path
    file_count: int
    total_bytes: int
    summary: Dict[str, Any] = field(default_factory=dict)


class NiftiGeometryBuilder:
    """Constrói volumes NIfTI com geometria física RAS fiel ao espaço DICOM.

    Garante que a conversão de `.mat` (Medviso Segment) e DICOM para `.nii.gz`
    preserve integralmente `affine`, `pixdim`, orientação anatômica real e
    tipos de dados (`float32` para imagens e `int16` para máscaras).
    """

    @staticmethod
    def build_ras_affine_from_segment(
        setstruct: Any,
        num_slices: int,
        reverse_slices: bool = True,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Monta matriz affine 4x4 NIfTI (RAS) para array transposto `(col, row, slice)`.

        No Medviso Segment, `IM` vem como `(row, col, k)` com origem `ImagePosition`
        correspondendo à última fatia na direção da normal DICOM (`k` decrescente).
        Ao transpor `(row, col, k) -> (col, row, k)` e opcionalmente inverter o
        eixo `k` (`reverse_slices=True`), obtém-se exatamente a mesma geometria
        right-handed gerada por `SimpleITK` / `dcm2niix`.

        Args:
            setstruct: Estrutura `setstruct` extraída do arquivo `.mat`.
            num_slices: Número de fatias no eixo Z.
            reverse_slices: Se True, inverte o eixo Z para alinhar `slice_dir`
                no sentido positivo da normal DICOM (right-handed).

        Returns:
            Tupla contendo a matriz affine 4x4 (RAS) e os zooms `(dx, dy, dz)` em mm.
        """
        dx = float(getattr(setstruct, "ResolutionX", 1.0))
        dy = float(getattr(setstruct, "ResolutionY", 1.0))
        thickness = float(getattr(setstruct, "SliceThickness", 1.0))
        gap = float(getattr(setstruct, "SliceGap", 0.0))
        dz = thickness + gap

        iop_raw = getattr(setstruct, "ImageOrientation", None)
        ipp_raw = getattr(setstruct, "ImagePosition", None)

        if iop_raw is None or np.size(iop_raw) != 6 or ipp_raw is None or np.size(ipp_raw) != 3:
            # Fallback seguro preservando pelo menos o espaçamento físico real
            affine = np.diag([dx, dy, dz, 1.0]).astype(np.float64)
            return affine, (dx, dy, dz)

        iop = np.asarray(iop_raw, dtype=np.float64).ravel()
        ipp_lps = np.asarray(ipp_raw, dtype=np.float64).ravel()

        col_dir_lps = iop[0:3]
        row_dir_lps = iop[3:6]
        normal_lps = np.cross(col_dir_lps, row_dir_lps)
        norm_len = float(np.linalg.norm(normal_lps))
        if norm_len > 0:
            normal_lps = normal_lps / norm_len

        # No Segment, k=0 está em ipp_lps (topo da pilha) e caminha em -normal_lps.
        # Invertendo k (k' = nz - 1 - k), a origem passa para a base e passo = +normal_lps.
        if reverse_slices:
            origin_lps = ipp_lps - normal_lps * dz * max(num_slices - 1, 0)
            slice_dir_lps = normal_lps
        else:
            origin_lps = ipp_lps
            slice_dir_lps = -normal_lps

        affine_lps = np.eye(4, dtype=np.float64)
        affine_lps[0:3, 0] = col_dir_lps * dx
        affine_lps[0:3, 1] = row_dir_lps * dy
        affine_lps[0:3, 2] = slice_dir_lps * dz
        affine_lps[0:3, 3] = origin_lps

        # Conversão padrão DICOM LPS -> NIfTI RAS (inverte X e Y)
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine_ras = lps_to_ras @ affine_lps
        return affine_ras, (dx, dy, dz)

    @staticmethod
    def save_nifti(
        volume: np.ndarray,
        affine: np.ndarray,
        output_path: Path,
        dtype: Any,
        zooms: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Salva volume NIfTI compactado preservando affine, pixdim e dtype explícito."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(volume, dtype=dtype)
        nii = nib.Nifti1Image(arr, affine)
        nii.set_data_dtype(dtype)
        nii.set_qform(affine, code=1)
        nii.set_sform(affine, code=1)
        if zooms is not None:
            nii.header.set_zooms(zooms)
        nib.save(nii, str(output_path))


class BaseDatasetPackager(ABC):
    """Classe base abstrata para empacotadores de datasets Kaggle."""

    def __init__(self, workspace_root: Path, staging_root: Path, spec: DatasetSpec) -> None:
        self.workspace_root = workspace_root
        self.staging_root = staging_root
        self.spec = spec
        self.staging_dir = staging_root / spec.slug

    def reset_staging_dir(self) -> None:
        """Recria a pasta de staging limpa para o dataset."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def write_kaggle_metadata(self) -> Path:
        """Escreve `dataset-metadata.json` exigido pela Kaggle CLI (sempre Private)."""
        metadata = {
            "title": self.spec.title,
            "id": self.spec.kaggle_id,
            "subtitle": self.spec.subtitle,
            "description": self.spec.description,
            "isPrivate": True,
            "licenses": [{"name": self.spec.license_name}],
        }
        meta_path = self.staging_dir / "dataset-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta_path

    def compute_staging_stats(self) -> Tuple[int, int]:
        """Calcula quantidade de arquivos (excluindo metadata) e tamanho total em bytes."""
        files = [
            p for p in self.staging_dir.rglob("*")
            if p.is_file() and p.name != "dataset-metadata.json"
        ]
        total_bytes = sum(p.stat().st_size for p in files)
        return len(files), total_bytes

    @abstractmethod
    def run(self) -> PackagingReport:
        """Executa extração, limpeza, validação e gravação de metadados."""


class CmrMultiPackager(BaseDatasetPackager):
    """Empacota `robertgvds/cmr-multi-lge-sax` a partir de `data/cmr-multi/LGE_MULTI/`."""

    REQUIRED_SUBDIRS: Tuple[Tuple[str, int, bool], ...] = (
        ("SAX_TR/image", 40, False),
        ("SAX_TR/anno", 40, True),
        ("SAX_VAL/image", 7, False),
        ("SAX_VAL/anno", 7, True),
        ("SAX_TST/image", 13, False),
    )
    REQUIRED_SPREADSHEETS: Tuple[str, ...] = (
        "dataset_lge.xlsx",
        "dataset_lge_valid.xlsx",
        "dataset_lge_test.xlsx",
    )

    def run(self) -> PackagingReport:
        """Extrai apenas SAX LGE + planilhas, valida contagens e rótulos `{0..4}`."""
        LOGGER.info("Preparando dataset %s...", self.spec.kaggle_id)
        source_root = self.workspace_root / "data" / "cmr-multi" / "LGE_MULTI"
        if not source_root.exists():
            raise FileNotFoundError(f"Diretório CMR-MULTI não encontrado: {source_root}")

        self.reset_staging_dir()
        observed_labels: Set[int] = set()
        counts: Dict[str, int] = {}

        for rel_subdir, expected_count, is_mask in self.REQUIRED_SUBDIRS:
            src_dir = source_root / rel_subdir
            dst_dir = self.staging_dir / rel_subdir
            dst_dir.mkdir(parents=True, exist_ok=True)

            nii_files = sorted(
                p for p in src_dir.iterdir()
                if p.is_file() and p.name.endswith(".nii.gz") and not p.name.startswith(".")
            )
            if len(nii_files) != expected_count:
                raise ValueError(
                    f"Contagem inesperada em {rel_subdir}: esperado {expected_count}, encontrado {len(nii_files)}"
                )

            for src_file in nii_files:
                shutil.copy2(src_file, dst_dir / src_file.name)
                if is_mask:
                    mask_arr = np.rint(nib.load(str(src_file)).get_fdata()).astype(np.int16)
                    observed_labels.update(int(v) for v in np.unique(mask_arr))

            counts[rel_subdir] = len(nii_files)

        if not observed_labels.issubset(EXPECTED_CMR_LABELS) or observed_labels != EXPECTED_CMR_LABELS:
            raise ValueError(
                f"Rótulos inválidos no CMR-MULTI: encontrados {sorted(observed_labels)}, esperados {sorted(EXPECTED_CMR_LABELS)}"
            )

        for sheet_name in self.REQUIRED_SPREADSHEETS:
            src_sheet = source_root / sheet_name
            if not src_sheet.exists():
                raise FileNotFoundError(f"Planilha ausente no CMR-MULTI: {src_sheet}")
            shutil.copy2(src_sheet, self.staging_dir / sheet_name)

        self.write_kaggle_metadata()
        file_count, total_bytes = self.compute_staging_stats()
        LOGGER.info(
            "CMR-MULTI validado: %d arquivos (%.2f MB) | rótulos=%s",
            file_count,
            total_bytes / 1048576,
            sorted(observed_labels),
        )
        return PackagingReport(
            spec=self.spec,
            staging_dir=self.staging_dir,
            file_count=file_count,
            total_bytes=total_bytes,
            summary={"counts": counts, "labels": sorted(observed_labels)},
        )

    def create_local_sample(self, sample_root: Path) -> Dict[str, Any]:
        """Gera `data_sample/` com 2 volumes CMR-MULTI (1 com fibrose, 1 sem) para dry-run em CPU."""
        if sample_root.exists():
            shutil.rmtree(sample_root)

        img_out = sample_root / "SAX_TR" / "image"
        msk_out = sample_root / "SAX_TR" / "anno"
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)

        # LGE_SAX_001 (sem fibrose) e LGE_SAX_006 (com fibrose) — ambos no treino SEED=42
        selected_cases = ["LGE_SAX_001.nii.gz", "LGE_SAX_006.nii.gz"]
        sample_summary: Dict[str, Any] = {}
        for fname in selected_cases:
            src_img = self.staging_dir / "SAX_TR" / "image" / fname
            src_msk = self.staging_dir / "SAX_TR" / "anno" / fname
            shutil.copy2(src_img, img_out / fname)
            shutil.copy2(src_msk, msk_out / fname)
            uniq = sorted(int(v) for v in np.unique(np.rint(nib.load(str(src_msk)).get_fdata())))
            sample_summary[fname] = {"has_scar": 3 in uniq, "labels": uniq}

        LOGGER.info("Amostra local gerada em %s: %s", sample_root, sample_summary)
        return sample_summary


class EmidecPackager(BaseDatasetPackager):
    """Desaninha, compacta para `.nii.gz` e empacota `robertgvds/emidec-lge-sax`."""

    EXPECTED_TRAIN_CASES = 100
    EXPECTED_TEST_CASES = 50

    @staticmethod
    def _find_single_nifti(directory: Path) -> Path:
        """Localiza recursivamente o arquivo real `.nii` ou `.nii.gz` dentro de pastas aninhadas."""
        if not directory.exists():
            raise FileNotFoundError(f"Subdiretório ausente no EMIDEC: {directory}")

        candidates = [
            p for p in sorted(directory.rglob("*"))
            if p.is_file()
            and not p.name.startswith(".")
            and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Esperado exatamente 1 arquivo NIfTI em {directory}, encontrados {len(candidates)}: {candidates}"
            )
        return candidates[0]

    @staticmethod
    def _find_info_txt(case_dir: Path) -> Optional[Path]:
        """Localiza o arquivo `.txt` clínico do paciente EMIDEC."""
        txts = [
            p for p in sorted(case_dir.iterdir())
            if p.is_file() and p.suffix.lower() == ".txt" and not p.name.startswith(".")
        ]
        return txts[0] if txts else None

    @staticmethod
    def _compress_nifti(src_path: Path, dst_path: Path, is_mask: bool) -> Set[int]:
        """Converte `.nii` para `.nii.gz` preservando integralmente affine e pixdim."""
        nii = nib.load(str(src_path))
        data = nii.get_fdata()
        affine = nii.affine.copy()
        header = nii.header.copy()

        if is_mask:
            arr = np.rint(data).astype(np.int16)
            out_nii = nib.Nifti1Image(arr, affine, header)
            out_nii.set_data_dtype(np.int16)
            nib.save(out_nii, str(dst_path))
            return {int(v) for v in np.unique(arr)}

        arr = data.astype(np.float32)
        out_nii = nib.Nifti1Image(arr, affine, header)
        out_nii.set_data_dtype(np.float32)
        nib.save(out_nii, str(dst_path))
        return set()

    def run(self) -> PackagingReport:
        """Processa os 100 casos de treino e 50 casos de teste do EMIDEC."""
        LOGGER.info("Preparando dataset %s...", self.spec.kaggle_id)
        train_src = self.workspace_root / "data" / "emidec-dataset-1.0.1"
        test_src = self.workspace_root / "data" / "emidec-segmentation-testset-1.0.0"

        if not train_src.exists() or not test_src.exists():
            raise FileNotFoundError(f"Pastas EMIDEC ausentes: {train_src} / {test_src}")

        self.reset_staging_dir()
        observed_labels: Set[int] = set()

        train_cases = sorted(p for p in train_src.iterdir() if p.is_dir() and p.name.startswith("Case_"))
        test_cases = sorted(p for p in test_src.iterdir() if p.is_dir() and p.name.startswith("Case_"))

        if len(train_cases) != self.EXPECTED_TRAIN_CASES:
            raise ValueError(f"EMIDEC train: esperado {self.EXPECTED_TRAIN_CASES}, encontrado {len(train_cases)}")
        if len(test_cases) != self.EXPECTED_TEST_CASES:
            raise ValueError(f"EMIDEC test: esperado {self.EXPECTED_TEST_CASES}, encontrado {len(test_cases)}")

        for case_dir in train_cases:
            dst_case = self.staging_dir / "train" / case_dir.name
            dst_case.mkdir(parents=True, exist_ok=True)
            img_src = self._find_single_nifti(case_dir / "Images")
            msk_src = self._find_single_nifti(case_dir / "Contours")
            self._compress_nifti(img_src, dst_case / "image.nii.gz", is_mask=False)
            case_labels = self._compress_nifti(msk_src, dst_case / "mask.nii.gz", is_mask=True)
            observed_labels.update(case_labels)

            txt_src = self._find_info_txt(case_dir)
            if txt_src is not None:
                shutil.copy2(txt_src, dst_case / "info.txt")

        for case_dir in test_cases:
            dst_case = self.staging_dir / "test" / case_dir.name
            dst_case.mkdir(parents=True, exist_ok=True)
            img_src = self._find_single_nifti(case_dir / "Images")
            self._compress_nifti(img_src, dst_case / "image.nii.gz", is_mask=False)

            txt_src = self._find_info_txt(case_dir)
            if txt_src is not None:
                shutil.copy2(txt_src, dst_case / "info.txt")

        if observed_labels != EXPECTED_EMIDEC_LABELS:
            raise ValueError(
                f"Rótulos inválidos no EMIDEC: encontrados {sorted(observed_labels)}, esperados {sorted(EXPECTED_EMIDEC_LABELS)}"
            )

        self.write_kaggle_metadata()
        file_count, total_bytes = self.compute_staging_stats()
        LOGGER.info(
            "EMIDEC validado: %d train + %d test (%d arquivos, %.2f MB) | rótulos=%s",
            len(train_cases),
            len(test_cases),
            file_count,
            total_bytes / 1048576,
            sorted(observed_labels),
        )
        return PackagingReport(
            spec=self.spec,
            staging_dir=self.staging_dir,
            file_count=file_count,
            total_bytes=total_bytes,
            summary={
                "train_cases": len(train_cases),
                "test_cases": len(test_cases),
                "labels": sorted(observed_labels),
            },
        )


class HuUfjfMatPackager(BaseDatasetPackager):
    """Empacota `robertgvds/hu-ufjf-lge-sax` a partir dos arquivos puros da pasta `HU/`.

    Constrói NIfTIs com geometria RAS fiel ao DICOM original (sem rotações/flips
    artificiais), extrai máscaras anatômicas (`Endo/Epi/RVEndo`), máscaras de
    fibrose (`Scar.Result`), máscara unificada de 5 classes (compatível com
    CMR-MULTI), salva os `.mat` anonimizados e inclui o Paciente 20 (sem máscara)
    convertido diretamente da série DICOM LGE SAX.
    """

    def _resolve_mat_sources(self) -> List[HuMatSource]:
        """Mapeia a versão revisada oficial do arquivo `.mat` de cada paciente HU-UFJF."""
        hu_root = self.workspace_root / "HU"
        seg_zip = hu_root / "segmentacoes-20261126T214110Z-1-001.zip"
        p10_file = hu_root / "Patient_10_jov.mat"

        sources: List[HuMatSource] = [
            HuMatSource(1, "zip", seg_zip, "segmentacoes/Patient_1.mat", "segmentacoes_revisadas"),
            HuMatSource(2, "zip", seg_zip, "segmentacoes/Patient_2.mat", "segmentacoes_revisadas"),
            HuMatSource(3, "zip", seg_zip, "segmentacoes/Patient_3.mat", "segmentacoes_revisadas"),
            HuMatSource(5, "zip", seg_zip, "segmentacoes/Patient_5.mat", "segmentacoes_revisadas"),
            HuMatSource(6, "zip", seg_zip, "segmentacoes/Patient_6.mat", "segmentacoes_revisadas"),
            HuMatSource(7, "zip", seg_zip, "segmentacoes/Patient_7.mat", "segmentacoes_revisadas"),
            HuMatSource(
                9,
                "zip",
                seg_zip,
                "segmentacoes/segmetacoes antigas/segmentacoesPacientes/comRoi/Patient_9.mat",
                "comRoi_unica_disponivel",
            ),
            HuMatSource(10, "file", p10_file, None, "patient_10_jov"),
            HuMatSource(18, "zip", seg_zip, "segmentacoes/Patient_18.mat", "segmentacoes_revisadas"),
            HuMatSource(19, "zip", seg_zip, "segmentacoes/Patient_19.mat", "segmentacoes_revisadas"),
        ]
        return sources

    @staticmethod
    def _read_mat_bytes(source: HuMatSource) -> bytes:
        """Lê os bytes de um `.mat` a partir de arquivo solto ou membro de ZIP."""
        if source.source_type == "file":
            return source.source_path.read_bytes()
        with zipfile.ZipFile(source.source_path, "r") as zf:
            assert source.zip_member is not None
            return zf.read(source.zip_member)

    @staticmethod
    def _save_anonymized_mat(raw_bytes: bytes, output_path: Path, patient_id: int) -> None:
        """Remove campos identificáveis (LGPD) do `.mat` e salva cópia limpa para nuvem."""
        mat_dict = sio.loadmat(io.BytesIO(raw_bytes), squeeze_me=False, struct_as_record=True)
        # Limpa campos PII dentro de `info` se presentes
        if "info" in mat_dict and isinstance(mat_dict["info"], np.ndarray) and mat_dict["info"].dtype.names:
            info_arr = mat_dict["info"]
            for field_name in PII_MAT_INFO_FIELDS:
                if field_name in info_arr.dtype.names:
                    replacement = f"HU_UFJF_P{patient_id:02d}" if field_name in ("Name", "ID") else "ANONYMIZED"
                    info_arr[field_name][0, 0] = np.array([replacement], dtype="<U32")
        # Remove chaves internas do scipy antes de regravar
        clean_dict = {k: v for k, v in mat_dict.items() if not k.startswith("__")}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(str(output_path), clean_dict, do_compression=True)

    @staticmethod
    def _rasterize_polygon(
        mask_slice: np.ndarray,
        x_coords: Optional[np.ndarray],
        y_coords: Optional[np.ndarray],
        label: int,
    ) -> None:
        """Desenha polígono preenchido no plano 2D `(row, col)` usando convenção Medviso (1-indexada)."""
        if x_coords is None or y_coords is None:
            return
        valid = (~np.isnan(x_coords)) & (~np.isnan(y_coords))
        if int(np.sum(valid)) <= 3:
            return
        # Validado visualmente: no Medviso Segment X corresponde ao eixo 0 (row) e Y ao eixo 1 (col)
        rows = np.round(x_coords[valid]).astype(int) - 1
        cols = np.round(y_coords[valid]).astype(int) - 1
        rr, cc = polygon(rows, cols, shape=mask_slice.shape)
        mask_slice[rr, cc] = label

    def _extract_volumes_from_mat(self, raw_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float]]:
        """Extrai imagem, máscaras e affine RAS de um `.mat` do Medviso Segment."""
        mat = sio.loadmat(io.BytesIO(raw_bytes), squeeze_me=True, struct_as_record=False)
        setstruct = mat["setstruct"]
        if isinstance(setstruct, np.ndarray) and setstruct.size > 0:
            setstruct = setstruct.flat[0]

        raw_img = mat.get("im", None)
        if raw_img is None or getattr(raw_img, "size", 0) == 0:
            raw_img = setstruct.IM
        img_rck = np.asarray(raw_img, dtype=np.float32)
        if img_rck.ndim != 3:
            raise ValueError(f"Esperado volume 3D LGE no .mat, obtido shape {img_rck.shape}")

        num_slices = img_rck.shape[2]
        anat_rck = np.zeros(img_rck.shape, dtype=np.int16)

        for z in range(num_slices):
            # Ordem importa: primeiro Epicárdio (Mio=2), depois Endocárdio (VE=1) e VD (3)
            self._rasterize_polygon(
                anat_rck[:, :, z],
                setstruct.EpiX[:, z] if hasattr(setstruct, "EpiX") else None,
                setstruct.EpiY[:, z] if hasattr(setstruct, "EpiY") else None,
                LABEL_MYO,
            )
            self._rasterize_polygon(
                anat_rck[:, :, z],
                setstruct.EndoX[:, z] if hasattr(setstruct, "EndoX") else None,
                setstruct.EndoY[:, z] if hasattr(setstruct, "EndoY") else None,
                LABEL_LV,
            )
            self._rasterize_polygon(
                anat_rck[:, :, z],
                setstruct.RVEndoX[:, z] if hasattr(setstruct, "RVEndoX") else None,
                setstruct.RVEndoY[:, z] if hasattr(setstruct, "RVEndoY") else None,
                LABEL_RV_4CLASS,
            )

        # Extração fiel da fibrose validada no Segment (`Scar.Result`)
        scar_rck = np.zeros(img_rck.shape, dtype=np.int16)
        if hasattr(setstruct, "Scar") and hasattr(setstruct.Scar, "Result"):
            scar_res = np.asarray(setstruct.Scar.Result)
            if scar_res.shape == img_rck.shape:
                scar_rck = (scar_res > 0).astype(np.int16)

        # Máscara unificada 5 classes compatível com CMR-MULTI: 0=Bg, 1=VE, 2=Mio Saudável, 3=Scar, 4=VD
        seg5_rck = np.zeros(img_rck.shape, dtype=np.int16)
        seg5_rck[anat_rck == LABEL_LV] = LABEL_LV
        seg5_rck[anat_rck == LABEL_MYO] = LABEL_MYO
        seg5_rck[anat_rck == LABEL_RV_4CLASS] = LABEL_RV_5CLASS
        seg5_rck[scar_rck == LABEL_SCAR_BIN] = LABEL_SCAR_5CLASS

        # Transpõe (row, col, z) -> (col, row, z) e inverte z para convenção RAS right-handed
        img_ijk = np.transpose(img_rck, (1, 0, 2))[:, :, ::-1].copy()
        anat_ijk = np.transpose(anat_rck, (1, 0, 2))[:, :, ::-1].copy()
        scar_ijk = np.transpose(scar_rck, (1, 0, 2))[:, :, ::-1].copy()
        seg5_ijk = np.transpose(seg5_rck, (1, 0, 2))[:, :, ::-1].copy()

        affine_ras, zooms = NiftiGeometryBuilder.build_ras_affine_from_segment(
            setstruct, num_slices=num_slices, reverse_slices=True
        )
        return img_ijk, anat_ijk, scar_ijk, seg5_ijk, affine_ras, zooms

    def _extract_patient_20_from_dicom(self, out_dir: Path) -> Optional[Dict[str, Any]]:
        """Converte a série LGE SAX (`DE_single-shot_EC_psir_p2_MAG`) do Paciente 20 para NIfTI puro."""
        research_zip = self.workspace_root / "HU" / "IMAGENS DICOM PACIENTES PESQUISA-20261126T213956Z-1-001.zip"
        if not research_zip.exists():
            return None

        with zipfile.ZipFile(research_zip, "r") as outer:
            p20_members = [m for m in outer.namelist() if "PACIENTE 20" in m and m.endswith(".zip")]
            if not p20_members:
                return None
            p20_bytes = outer.read(p20_members[0])

        with tempfile.TemporaryDirectory(prefix="hu_p20_") as tmp_str:
            tmp_dir = Path(tmp_str)
            series_files: Dict[str, List[Path]] = defaultdict(list)
            with zipfile.ZipFile(io.BytesIO(p20_bytes), "r") as izf:
                for idx, info in enumerate(izf.infolist()):
                    if info.is_dir():
                        continue
                    ext = Path(info.filename).suffix.lower()
                    if ext in IGNORED_DICOM_EXTENSIONS:
                        continue
                    raw = izf.read(info.filename)
                    try:
                        ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
                    except Exception:
                        continue
                    sdesc = str(getattr(ds, "SeriesDescription", ""))
                    if "DE_single-shot_EC_psir_p2_MAG" in sdesc:
                        suid = str(getattr(ds, "SeriesInstanceUID", "default"))
                        dcm_out = tmp_dir / f"{suid}_{idx:04d}.dcm"
                        dcm_out.write_bytes(raw)
                        series_files[suid].append(dcm_out)

            if not series_files:
                LOGGER.warning("Série LGE SAX não encontrada nos DICOMs do Paciente 20.")
                return None

            # Escolhe a série MAG eixo curto mais completa
            best_uid = max(series_files, key=lambda uid: len(series_files[uid]))
            reader = sitk.ImageSeriesReader()
            sorted_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(tmp_dir), best_uid)
            if not sorted_names:
                sorted_names = [str(p) for p in sorted(series_files[best_uid])]
            reader.SetFileNames(sorted_names)
            sitk_img = reader.Execute()
            sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / "img.nii.gz"
            sitk.WriteImage(sitk_img, str(img_path))

            nii_check = nib.load(str(img_path))
            zooms = tuple(round(float(z), 4) for z in nii_check.header.get_zooms()[:3])
            return {
                "patient_id": "patient_20",
                "source": "DICOM_DE_single-shot_EC_psir_p2_MAG",
                "has_anat": False,
                "has_scar": False,
                "shape": str(nii_check.shape),
                "spacing_mm": str(zooms),
                "myo_volume_ml": "",
                "scar_volume_ml": "",
                "labels_5class": "[]",
            }

    def run(self) -> PackagingReport:
        """Gera NIfTIs puros, MATs anonimizados e manifesto CSV do HU-UFJF."""
        LOGGER.info("Preparando dataset %s a partir de HU/...", self.spec.kaggle_id)
        self.reset_staging_dir()

        nifti_root = self.staging_dir / "nifti"
        mat_root = self.staging_dir / "raw_mat_anonymized"
        nifti_root.mkdir(parents=True, exist_ok=True)
        mat_root.mkdir(parents=True, exist_ok=True)

        manifest_rows: List[Dict[str, Any]] = []
        for src in self._resolve_mat_sources():
            raw_bytes = self._read_mat_bytes(src)
            img_ijk, anat_ijk, scar_ijk, seg5_ijk, affine_ras, zooms = self._extract_volumes_from_mat(raw_bytes)

            p_name = f"patient_{src.patient_id}"
            p_dir = nifti_root / p_name
            NiftiGeometryBuilder.save_nifti(img_ijk, affine_ras, p_dir / "img.nii.gz", np.float32, zooms)
            NiftiGeometryBuilder.save_nifti(anat_ijk, affine_ras, p_dir / "seg_anat.nii.gz", np.int16, zooms)
            NiftiGeometryBuilder.save_nifti(scar_ijk, affine_ras, p_dir / "seg_scar.nii.gz", np.int16, zooms)
            NiftiGeometryBuilder.save_nifti(seg5_ijk, affine_ras, p_dir / "seg_5class.nii.gz", np.int16, zooms)

            # Salva o .mat original anonimizado na nuvem
            self._save_anonymized_mat(raw_bytes, mat_root / f"{p_name}.mat", src.patient_id)

            voxel_ml = (zooms[0] * zooms[1] * zooms[2]) / 1000.0
            myo_voxels = int(np.count_nonzero((seg5_ijk == LABEL_MYO) | (seg5_ijk == LABEL_SCAR_5CLASS)))
            scar_voxels = int(np.count_nonzero(scar_ijk == LABEL_SCAR_BIN))
            labels_5c = sorted(int(v) for v in np.unique(seg5_ijk))

            manifest_rows.append(
                {
                    "patient_id": p_name,
                    "source": src.version_note,
                    "has_anat": True,
                    "has_scar": scar_voxels > 0,
                    "shape": str(img_ijk.shape),
                    "spacing_mm": str(tuple(round(z, 4) for z in zooms)),
                    "myo_volume_ml": f"{myo_voxels * voxel_ml:.2f}",
                    "scar_volume_ml": f"{scar_voxels * voxel_ml:.2f}",
                    "labels_5class": str(labels_5c),
                }
            )

        # Inclui Paciente 20 (sem segmentação) extraído diretamente do DICOM
        p20_row = self._extract_patient_20_from_dicom(nifti_root / "patient_20")
        if p20_row is not None:
            manifest_rows.append(p20_row)

        manifest_path = self.staging_dir / "manifest.csv"
        fieldnames = [
            "patient_id", "source", "has_anat", "has_scar",
            "shape", "spacing_mm", "myo_volume_ml", "scar_volume_ml", "labels_5class",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        readme_text = (
            "# HU-UFJF LGE-CMR SAX Dataset (Private)\n\n"
            "- **Origem:** Coorte clínica de miocardite (HU-UFJF), convertida diretamente dos arquivos "
            "originais `.mat` (Medviso Segment) e DICOM preservando `affine` RAS e `pixdim` reais.\n"
            "- **Estrutura:**\n"
            "  - `nifti/patient_X/img.nii.gz` (`float32`)\n"
            "  - `nifti/patient_X/seg_anat.nii.gz` (`int16`: `0=Bg, 1=LV, 2=MyoTotal, 3=RV`)\n"
            "  - `nifti/patient_X/seg_scar.nii.gz` (`int16`: `0=Bg, 1=Scar` via `Scar.Result`)\n"
            "  - `nifti/patient_X/seg_5class.nii.gz` (`int16`: `0=Bg, 1=LV, 2=MyoHealthy, 3=Scar, 4=RV`)\n"
            "  - `raw_mat_anonymized/patient_X.mat` (arquivos `.mat` originais anonimizados)\n"
            "  - `manifest.csv` (dimensões, espaçamentos físicos e volumes em mL)\n"
        )
        (self.staging_dir / "README.md").write_text(readme_text, encoding="utf-8")

        self.write_kaggle_metadata()
        file_count, total_bytes = self.compute_staging_stats()
        LOGGER.info(
            "HU-UFJF validado: %d pacientes (%d rotulados com fibrose) | %d arquivos (%.2f MB)",
            len(manifest_rows),
            sum(1 for r in manifest_rows if r["has_scar"]),
            file_count,
            total_bytes / 1048576,
        )
        return PackagingReport(
            spec=self.spec,
            staging_dir=self.staging_dir,
            file_count=file_count,
            total_bytes=total_bytes,
            summary={"patients": manifest_rows},
        )


class HuUfjfDicomArchivePackager(BaseDatasetPackager):
    """Empacota `robertgvds/hu-ufjf-dicom-archive` (Private) com DICOMs 100% anonimizados.

    Reaproveita os ZIPs já anonimizados de `HU/DICOM-*.zip` para os pacientes
    `1, 2, 3, 5, 6, 7, 9, 10` (removendo `__MACOSX`/`.DS_Store`) e extrai, filtra
    e anonimiza via `pydicom` os arquivos DICOM reais de `18, 19, 20` a partir de
    `HU/IMAGENS DICOM PACIENTES PESQUISA-*.zip` (descartando visualizadores `.exe`,
    `.dll` e nomes civis).
    """

    ANONYMIZED_ZIP_MAP: Dict[int, str] = {
        1: "DICOM/Paciente 1.zip",
        2: "DICOM/Paciente 2.zip",
        3: "DICOM/Paciente 3.zip",
        5: "DICOM/Paciente 5.zip",
        6: "DICOM/Paciente 6.zip",
        7: "DICOM/Paciente 7.zip",
        9: "DICOM/Paciente  9.zip",
        10: "DICOM/Paciente 10.zip",
    }
    EXTRA_RESEARCH_PATIENTS: Tuple[int, ...] = (18, 19, 20)

    @staticmethod
    def _anonymize_dicom_bytes(raw_bytes: bytes, patient_label: str) -> Optional[bytes]:
        """Valida se os bytes são um DICOM de imagem e remove tags identificáveis."""
        try:
            ds = pydicom.dcmread(io.BytesIO(raw_bytes), force=False)
        except Exception:
            return None
        if not hasattr(ds, "Rows") or not hasattr(ds, "SOPInstanceUID"):
            return None

        for tag in PII_DICOM_TAGS:
            if hasattr(ds, tag):
                setattr(ds, tag, patient_label if tag in ("PatientName", "PatientID") else "")

        buf = io.BytesIO()
        ds.save_as(buf, enforce_file_format=False)
        return buf.getvalue()

    def _package_patient_zip_clean(
        self,
        raw_zip_bytes: bytes,
        dst_zip_path: Path,
        patient_id: int,
        force_anonymize_tags: bool,
    ) -> int:
        """Limpa lixo de SO/visualizadores e gera um ZIP estruturado e anonimizado por paciente."""
        patient_label = f"HU_UFJF_P{patient_id:02d}"
        written_count = 0
        dst_zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(raw_zip_bytes), "r") as src_zf, zipfile.ZipFile(
            dst_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as dst_zf:
            for info in sorted(src_zf.infolist(), key=lambda x: x.filename):
                if info.is_dir():
                    continue
                norm_name = info.filename.replace("\\", "/")
                base_name = os.path.basename(norm_name)
                if "__MACOSX" in norm_name or base_name.startswith(".") or base_name.lower() == "dicomdir":
                    continue
                if Path(base_name).suffix.lower() in IGNORED_DICOM_EXTENSIONS:
                    continue

                raw_file = src_zf.read(info.filename)
                if force_anonymize_tags:
                    clean_bytes = self._anonymize_dicom_bytes(raw_file, patient_label)
                    if clean_bytes is None:
                        continue
                    written_count += 1
                    dst_zf.writestr(f"patient_{patient_id}/dicom/img_{written_count:05d}.dcm", clean_bytes)
                else:
                    # Mesmo para os ZIPs já anonimizados, remove prefixos irregulares mantendo subpastas de séries
                    parts = [p for p in norm_name.split("/") if p]
                    rel_parts = parts[2:] if len(parts) > 2 else parts[-1:]
                    clean_rel = "/".join(rel_parts)
                    written_count += 1
                    dst_zf.writestr(f"patient_{patient_id}/{clean_rel}", raw_file)

        return written_count

    def run(self) -> PackagingReport:
        """Constrói o arquivo em nuvem de DICOMs anonimizados dos 11 pacientes HU-UFJF."""
        LOGGER.info("Preparando dataset %s (DICOMs anonimizados)...", self.spec.kaggle_id)
        hu_root = self.workspace_root / "HU"
        anon_outer_zip = hu_root / "DICOM-20261126T214124Z-1-001.zip"
        research_outer_zip = hu_root / "IMAGENS DICOM PACIENTES PESQUISA-20261126T213956Z-1-001.zip"

        self.reset_staging_dir()
        dicom_counts: Dict[str, int] = {}

        with zipfile.ZipFile(anon_outer_zip, "r") as outer:
            for pid, member in sorted(self.ANONYMIZED_ZIP_MAP.items()):
                inner_bytes = outer.read(member)
                dst_zip = self.staging_dir / f"patient_{pid}_dicom.zip"
                cnt = self._package_patient_zip_clean(inner_bytes, dst_zip, pid, force_anonymize_tags=False)
                dicom_counts[f"patient_{pid}"] = cnt
                LOGGER.info("  patient_%d: %d arquivos DICOM empacotados.", pid, cnt)

        with zipfile.ZipFile(research_outer_zip, "r") as outer:
            names = outer.namelist()
            for pid in self.EXTRA_RESEARCH_PATIENTS:
                matches = [m for m in names if f"PACIENTE {pid}/" in m and m.endswith(".zip")]
                if not matches:
                    LOGGER.warning("ZIP DICOM não encontrado para PACIENTE %d", pid)
                    continue
                inner_bytes = outer.read(matches[0])
                dst_zip = self.staging_dir / f"patient_{pid}_dicom.zip"
                cnt = self._package_patient_zip_clean(inner_bytes, dst_zip, pid, force_anonymize_tags=True)
                dicom_counts[f"patient_{pid}"] = cnt
                LOGGER.info("  patient_%d (anonimizado via pydicom): %d DICOMs empacotados.", pid, cnt)

        manifest_path = self.staging_dir / "dicom_archive_manifest.json"
        manifest_path.write_text(json.dumps(dicom_counts, indent=2), encoding="utf-8")

        self.write_kaggle_metadata()
        file_count, total_bytes = self.compute_staging_stats()
        LOGGER.info(
            "HU-UFJF DICOM Archive validado: %d pacientes | %d arquivos (%.2f MB)",
            len(dicom_counts),
            file_count,
            total_bytes / 1048576,
        )
        return PackagingReport(
            spec=self.spec,
            staging_dir=self.staging_dir,
            file_count=file_count,
            total_bytes=total_bytes,
            summary={"dicom_counts": dicom_counts},
        )


class KaggleDatasetPublisher:
    """Encapsula chamadas à Kaggle CLI para criação/atualização e verificação de datasets."""

    def __init__(self, kaggle_bin: str = "kaggle") -> None:
        self.kaggle_bin = kaggle_bin

    def _run_cli(self, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        """Executa comando da Kaggle CLI capturando saída e registrando logs."""
        cmd = [self.kaggle_bin, *args]
        LOGGER.info("Executando: %s", " ".join(cmd))
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if result.stdout.strip():
            LOGGER.info("STDOUT:\n%s", result.stdout.strip())
        if result.stderr.strip():
            LOGGER.warning("STDERR:\n%s", result.stderr.strip())
        if check and result.returncode != 0:
            raise RuntimeError(f"Falha na Kaggle CLI ({result.returncode}): {' '.join(cmd)}")
        return result

    def dataset_exists(self, kaggle_id: str) -> bool:
        """Verifica se o dataset já existe na conta Kaggle."""
        res = self._run_cli(["datasets", "status", kaggle_id], check=False)
        return res.returncode == 0 and "404" not in (res.stdout + res.stderr)

    def publish(self, report: PackagingReport, version_message: str) -> None:
        """Cria dataset privado (ou envia nova versão se já existir)."""
        meta_path = report.staging_dir / "dataset-metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("isPrivate", False):
            raise PermissionError(f"Bloqueado por segurança: isPrivate=False em {report.spec.kaggle_id}")

        if self.dataset_exists(report.spec.kaggle_id):
            LOGGER.info("Dataset %s já existe; enviando nova versão...", report.spec.kaggle_id)
            self._run_cli(
                [
                    "datasets",
                    "version",
                    "-p",
                    str(report.staging_dir),
                    "-m",
                    version_message,
                    "--dir-mode",
                    "zip",
                ]
            )
        else:
            LOGGER.info("Criando novo Private Dataset %s...", report.spec.kaggle_id)
            self._run_cli(
                [
                    "datasets",
                    "create",
                    "-p",
                    str(report.staging_dir),
                    "--dir-mode",
                    "zip",
                ]
            )

    def list_remote_files(self, kaggle_id: str) -> str:
        """Consulta `kaggle datasets files` para conferência pós-upload."""
        res = self._run_cli(["datasets", "files", kaggle_id], check=False)
        return res.stdout.strip()


class DatasetUploadOrchestrator:
    """Orquestrador principal do pipeline de preparação, validação e publicação."""

    SPECS: Dict[str, DatasetSpec] = {
        "cmr_multi": DatasetSpec(
            key="cmr_multi",
            slug="cmr-multi-lge-sax",
            title="CMR-MULTI LGE SAX (Clean)",
            subtitle="Volumes LGE-CMR eixo curto (SAX_TR, SAX_VAL, SAX_TST) e planilhas",
            description="Subconjunto limpo LGE SAX do CMR-MULTI (40 TR, 7 VAL, 13 TST; 5 classes: 0=Bg,1=LV,2=Myo,3=Scar,4=RV).",
        ),
        "emidec": DatasetSpec(
            key="emidec",
            slug="emidec-lge-sax",
            title="EMIDEC LGE SAX (Clean NIfTI GZ)",
            subtitle="100 casos treino + 50 casos teste EMIDEC desaninhados em .nii.gz",
            description="EMIDEC desaninhado e compactado preservando affine/pixdim (0=Bg,1=LV,2=Myo,3=Scar,4=MVO; sem VD).",
        ),
        "hu_ufjf": DatasetSpec(
            key="hu_ufjf",
            slug="hu-ufjf-lge-sax",
            title="HU-UFJF LGE SAX (Private Cohort)",
            subtitle="Coorte clínica de miocardite HU-UFJF em NIfTI RAS puro e MAT anonimizado",
            description="10 pacientes rotulados (anatomia + fibrose via Scar.Result) + 1 paciente sem máscara (P20), com geometria RAS fiel ao DICOM.",
        ),
        "hu_dicom": DatasetSpec(
            key="hu_dicom",
            slug="hu-ufjf-dicom-archive",
            title="HU-UFJF DICOM Archive (Private Anonymized)",
            subtitle="Backup estruturado e anonimizado dos exames DICOM brutos da coorte HU-UFJF",
            description="Arquivos DICOM anonimizados de 11 pacientes (P1..P20), sem nomes civis nem executáveis/visualizadores.",
        ),
    }

    def __init__(
        self,
        workspace_root: Path,
        staging_root: Path,
        sample_dir: Path,
        kaggle_bin: str,
        dry_run: bool,
    ) -> None:
        self.workspace_root = workspace_root
        self.staging_root = staging_root
        self.sample_dir = sample_dir
        self.dry_run = dry_run
        self.publisher = KaggleDatasetPublisher(kaggle_bin=kaggle_bin)

    def _build_packager(self, key: str) -> BaseDatasetPackager:
        """Instancia o empacotador correspondente à chave solicitada."""
        spec = self.SPECS[key]
        mapping = {
            "cmr_multi": CmrMultiPackager,
            "emidec": EmidecPackager,
            "hu_ufjf": HuUfjfMatPackager,
            "hu_dicom": HuUfjfDicomArchivePackager,
        }
        return mapping[key](self.workspace_root, self.staging_root, spec)

    def execute(
        self,
        selected_keys: Sequence[str],
        create_sample: bool = True,
        skip_staging: bool = False,
    ) -> List[PackagingReport]:
        """Executa o fluxo completo para os datasets selecionados."""
        self.staging_root.mkdir(parents=True, exist_ok=True)
        reports: List[PackagingReport] = []

        for key in selected_keys:
            packager = self._build_packager(key)
            if skip_staging:
                if not (packager.staging_dir / "dataset-metadata.json").exists():
                    raise FileNotFoundError(f"Staging não encontrado para --skip-staging: {packager.staging_dir}")
                file_count, total_bytes = packager.compute_staging_stats()
                report = PackagingReport(
                    spec=packager.spec,
                    staging_dir=packager.staging_dir,
                    file_count=file_count,
                    total_bytes=total_bytes,
                    summary={"reused_staging": True},
                )
            else:
                report = packager.run()
                if key == "cmr_multi" and create_sample and isinstance(packager, CmrMultiPackager):
                    packager.create_local_sample(self.sample_dir)

            reports.append(report)

            if not self.dry_run:
                self.publisher.publish(report, version_message="Initial clean structured release")

        return reports


def build_arg_parser() -> argparse.ArgumentParser:
    """Configura argumentos de linha de comando (CLI) em inglês."""
    parser = argparse.ArgumentParser(
        description="Empacota, valida e publica os datasets privados de LGE-CMR no Kaggle."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Diretório raiz contendo `data/` e `HU/`.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "dlcs_kaggle_staging",
        help="Diretório temporário de staging (fora do Git).",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path.cwd() / "data_sample",
        help="Diretório local para amostra mínima (2 volumes CMR-MULTI) para testes em CPU.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DatasetUploadOrchestrator.SPECS.keys()),
        default=["cmr_multi", "emidec", "hu_ufjf", "hu_dicom"],
        help="Lista de datasets a preparar/publicar.",
    )
    parser.add_argument(
        "--kaggle-bin",
        type=str,
        default="kaggle",
        help="Caminho ou nome do executável da Kaggle CLI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa extração e todas as validações em staging sem enviar ao Kaggle.",
    )
    parser.add_argument(
        "--skip-staging",
        action="store_true",
        help="Reutiliza o staging já validado em `--staging-dir` e executa apenas a publicação.",
    )
    return parser


def main() -> int:
    """Ponto de entrada principal da CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    orchestrator = DatasetUploadOrchestrator(
        workspace_root=args.workspace_root.resolve(),
        staging_root=args.staging_dir.resolve(),
        sample_dir=args.sample_dir.resolve(),
        kaggle_bin=args.kaggle_bin,
        dry_run=args.dry_run,
    )
    reports = orchestrator.execute(args.datasets, create_sample=True, skip_staging=args.skip_staging)

    summary_payload = [
        {
            "kaggle_id": r.spec.kaggle_id,
            "staging_dir": str(r.staging_dir),
            "file_count": r.file_count,
            "size_mb": round(r.total_bytes / 1048576, 2),
            "summary": r.summary,
        }
        for r in reports
    ]
    LOGGER.info("Resumo final:\n%s", json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
