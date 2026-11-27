"""Camada POO de dados, padronização de rótulos, pré-processamento físico e augmentation LGE-CMR.

Responsabilidades encapsuladas (SRP):
  - `TaskMode` e `LabelStandardizer`: padroniza máscaras do CMR-MULTI e da coorte
    clínica HU-UFJF para 4 classes (anatomia sem scar: `0=Bg, 1=VE, 2=MioTotal, 3=VD`)
    ou 5 classes (`0=Bg, 1=VE, 2=MioSaudavel, 3=Scar, 4=VD`).
  - `VolumePreprocessor`: reamostragem física isotrópica no plano XY (`1.0 mm`),
    normalização robusta de intensidades (compatível com PSIR negativo e MAG tórax
    inteiro) e remapeamento inverso de probabilidades para o espaço NIfTI nativo.
  - `CardiacAugmentor`: aumento de dados 2D em treino com simulação multi-resolução
    (Hoh et al., 2024), deformação elástica leve, rotação preservando quiralidade
    VE/VD (`flip=False`) e perturbações fotométricas.
  - `CmrMultiCohortLoader` e `HuUfjfCohortLoader`: leitura transparente de `.nii`
    e `.nii.gz` (contornando a descompactação automática do Kaggle).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates, zoom

LOGGER = logging.getLogger(__name__)


class TaskMode(str, Enum):
    """Modos suportados de segmentação cardíaca LGE-CMR."""

    ANATOMY_4C = "anatomy_4c"  # 0=Bg, 1=LV, 2=MyoTotal (Myo+Scar), 3=RV
    FULL_5C = "full_5c"        # 0=Bg, 1=LV, 2=MyoHealthy, 3=Scar, 4=RV

    @property
    def num_classes(self) -> int:
        """Retorna a quantidade total de classes (incluindo fundo)."""
        return 4 if self is TaskMode.ANATOMY_4C else 5

    @property
    def class_names(self) -> Dict[int, str]:
        """Retorna o dicionário canônico de nomes por índice de classe."""
        if self is TaskMode.ANATOMY_4C:
            return {0: "Background", 1: "LV", 2: "MyoTotal", 3: "RV"}
        return {0: "Background", 1: "LV", 2: "MyoHealthy", 3: "Scar", 4: "RV"}


@dataclass(frozen=True)
class VolumeRecord:
    """Metadados de um volume LGE-CMR de paciente (com ou sem máscara)."""

    patient_id: str
    cohort: str
    image_path: Path
    mask_path: Optional[Path] = None
    aux_scar_path: Optional[Path] = None
    mask_format: str = "cmr_5class"  # "cmr_5class", "hu_5class", "hu_anat_4class"


@dataclass
class PreparedVolume:
    """Volume 3D reamostrado no espaço físico de trabalho preservando metadados nativos."""

    patient_id: str
    cohort: str
    image_resampled: np.ndarray            # (H_res, W_res, Z) float32 normalizado
    mask_resampled: Optional[np.ndarray]   # (H_res, W_res, Z) int16 padronizado
    mask_native: Optional[np.ndarray]      # (H_nat, W_nat, Z) int16 padronizado
    native_shape: Tuple[int, int, int]
    native_spacing_mm: Tuple[float, float, float]
    native_affine: np.ndarray
    native_header_pixdim: np.ndarray
    target_spacing_mm: float


class NiftiVolumeIO:
    """Utilitários seguros de leitura/escrita NIfTI aceitando `.nii` e `.nii.gz`."""

    @staticmethod
    def strip_suffix(filename: str) -> str:
        """Remove `.nii.gz` ou `.nii` do nome do arquivo."""
        if filename.endswith(".nii.gz"):
            return filename[:-7]
        if filename.endswith(".nii"):
            return filename[:-4]
        return Path(filename).stem

    @staticmethod
    def is_nifti_file(path: Path) -> bool:
        """Verifica se o caminho corresponde a um arquivo NIfTI válido."""
        return path.is_file() and not path.name.startswith(".") and (
            path.name.endswith(".nii.gz") or path.name.endswith(".nii")
        )

    @staticmethod
    def find_nifti_by_stem(directory: Path, stem: str) -> Optional[Path]:
        """Localiza `<stem>.nii.gz` ou `<stem>.nii` no diretório informado."""
        for ext in (".nii.gz", ".nii"):
            candidate = directory / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def load_volume(path: Path, is_mask: bool = False) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """Carrega volume 3D removendo eventual dimensão singleton `(X, Y, Z, 1)`."""
        nii = nib.load(str(path))
        data = np.squeeze(nii.get_fdata())
        if data.ndim != 3:
            raise ValueError(f"Volume esperado em 3D em {path}, mas obtido shape {data.shape}.")
        if is_mask:
            arr = np.rint(data).astype(np.int16)
        else:
            arr = data.astype(np.float32)
        return arr, nii

    @staticmethod
    def save_mask_like(
        mask_3d: np.ndarray,
        affine: np.ndarray,
        pixdim: np.ndarray,
        output_path: Path,
        dtype: object = np.int16,
    ) -> Path:
        """Salva máscara NIfTI preservando `affine`, `pixdim` e tipo inteiro exato."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.rint(mask_3d).astype(dtype)
        out_nii = nib.Nifti1Image(arr, affine)
        out_nii.header["pixdim"] = pixdim
        out_nii.set_data_dtype(dtype)
        out_nii.set_qform(affine, code=1)
        out_nii.set_sform(affine, code=1)
        nib.save(out_nii, str(output_path))
        return output_path


class LabelStandardizer:
    """Padroniza máscaras manuais (CMR-MULTI e HU-UFJF) para o esquema da tarefa.

    Esquemas suportados:
      - `TaskMode.ANATOMY_4C`: `0=Fundo, 1=VE, 2=Miocárdio Total (Mio + Scar), 3=VD`
      - `TaskMode.FULL_5C`: `0=Fundo, 1=VE, 2=Miocárdio Saudável, 3=Scar, 4=VD`
    """

    @staticmethod
    def standardize_cmr_multi_mask(raw_mask: np.ndarray, task_mode: TaskMode) -> np.ndarray:
        """Converte máscara CMR-MULTI (`0..4`) para 4 classes (unindo Mio+Scar) ou 5 classes."""
        mask = np.rint(raw_mask).astype(np.int16)
        unique_vals = set(int(v) for v in np.unique(mask))
        if not unique_vals.issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"Rótulos inesperados na máscara CMR-MULTI: {sorted(unique_vals)}")

        if task_mode is TaskMode.FULL_5C:
            return mask

        # Une Miocárdio Saudável (2) + Fibrose (3) em Miocárdio Total (2) e remapeia VD (4 -> 3)
        out = np.zeros_like(mask, dtype=np.int16)
        out[mask == 1] = 1
        out[(mask == 2) | (mask == 3)] = 2
        out[mask == 4] = 3
        return out

    @staticmethod
    def standardize_hu_ufjf_mask(
        mask_arr: np.ndarray,
        task_mode: TaskMode,
        mask_format: str,
        aux_scar_arr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Padroniza máscaras clínicas do HU-UFJF (`seg_5class` ou `seg_anat` + `seg_scar`).

        Garante compatibilidade total com o protocolo CMR-MULTI:
          - Quando `seg_5class` está presente (`0=Bg, 1=VE, 2=MioSaudavel, 3=Scar, 4=VD`),
            aplica exatamente a mesma regra do CMR-MULTI (`2|3 -> 2` e `4 -> 3` em 4 classes).
          - Quando apenas `seg_anat` (`0=Bg, 1=VE, 2=MioTotal, 3=VD`) está disponível,
            recompõe `5 classes` usando `seg_scar` (`>0 -> 3` e `3 -> 4` para VD), ou
            retorna diretamente o formato 4 classes unindo eventuais voxels de `seg_scar`
            no miocárdio total.
        """
        mask = np.rint(mask_arr).astype(np.int16)
        if mask_format == "hu_5class":
            return LabelStandardizer.standardize_cmr_multi_mask(mask, task_mode)

        if mask_format == "hu_anat_4class":
            unique_vals = set(int(v) for v in np.unique(mask))
            if not unique_vals.issubset({0, 1, 2, 3}):
                raise ValueError(f"Rótulos inesperados em seg_anat HU-UFJF: {sorted(unique_vals)}")

            scar_bin = (
                (np.rint(aux_scar_arr).astype(np.int16) > 0)
                if aux_scar_arr is not None
                else np.zeros_like(mask, dtype=bool)
            )
            if task_mode is TaskMode.ANATOMY_4C:
                out = mask.copy()
                # Garante união explícita de fibrose ao miocárdio total quando presente
                out[scar_bin] = 2
                return out

            # Reconstrói formato unificado 5 classes a partir de seg_anat + seg_scar
            out = np.zeros_like(mask, dtype=np.int16)
            out[mask == 1] = 1
            out[mask == 2] = 2
            out[mask == 3] = 4  # VD em 5 classes = 4
            out[scar_bin] = 3   # Scar em 5 classes = 3
            return out

        raise ValueError(f"Formato de máscara HU-UFJF não suportado: '{mask_format}'")

    @classmethod
    def standardize_record_mask(cls, record: VolumeRecord, task_mode: TaskMode) -> Optional[np.ndarray]:
        """Carrega e padroniza a máscara de um `VolumeRecord` conforme a coorte e tarefa."""
        if record.mask_path is None:
            return None
        raw_mask, _ = NiftiVolumeIO.load_volume(record.mask_path, is_mask=True)
        if record.mask_format == "cmr_5class":
            return cls.standardize_cmr_multi_mask(raw_mask, task_mode)
        aux_scar = None
        if record.aux_scar_path is not None and record.aux_scar_path.is_file():
            aux_scar, _ = NiftiVolumeIO.load_volume(record.aux_scar_path, is_mask=True)
        return cls.standardize_hu_ufjf_mask(
            raw_mask,
            task_mode=task_mode,
            mask_format=record.mask_format,
            aux_scar_arr=aux_scar,
        )


class VolumePreprocessor:
    """Reamostra volumes para espaçamento físico isotrópico no plano XY e normaliza intensidades."""

    def __init__(
        self,
        target_spacing_mm: float = 1.0,
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        zscore_clip: Tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        self.target_spacing_mm = float(target_spacing_mm)
        self.clip_percentiles = clip_percentiles
        self.zscore_clip = zscore_clip

    def normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normaliza intensidades por volume com recorte percentil + Z-score robusto.

        Trata tanto imagens PSIR com valores negativos (CMR-MULTI) quanto imagens
        de magnitude de tórax completo com fundo zerado (HU-UFJF): se o volume é
        não-negativo e possui mais de 25% de zeros exatos, calcula estatísticas
        exclusivamente nos voxels de primeiro plano (`> 0`).
        """
        vol = volume.astype(np.float32, copy=False)
        zero_fraction = float(np.mean(vol == 0.0))
        if float(np.min(vol)) >= 0.0 and zero_fraction > 0.25:
            fg = vol[vol > 0.0]
            ref = fg if fg.size > 100 else vol.ravel()
        else:
            ref = vol.ravel()

        p_low, p_high = np.percentile(ref, self.clip_percentiles)
        clipped = np.clip(vol, float(p_low), float(p_high))

        ref_clipped = np.clip(ref, float(p_low), float(p_high))
        mean_val = float(np.mean(ref_clipped))
        std_val = float(np.std(ref_clipped))
        if std_val < 1e-6:
            std_val = 1.0

        norm = (clipped - mean_val) / std_val
        return np.clip(norm, self.zscore_clip[0], self.zscore_clip[1]).astype(np.float32)

    def resample_inplane(
        self,
        volume: np.ndarray,
        spacing_xy: Tuple[float, float],
        is_mask: bool = False,
    ) -> np.ndarray:
        """Reamostra apenas o plano XY para `target_spacing_mm`, preservando o eixo Z."""
        scale_x = float(spacing_xy[0]) / self.target_spacing_mm
        scale_y = float(spacing_xy[1]) / self.target_spacing_mm
        if abs(scale_x - 1.0) < 1e-3 and abs(scale_y - 1.0) < 1e-3:
            return volume.astype(np.int16 if is_mask else np.float32, copy=True)

        order = 0 if is_mask else 1
        resampled = zoom(volume, zoom=(scale_x, scale_y, 1.0), order=order)
        if is_mask:
            return np.rint(resampled).astype(np.int16)
        return resampled.astype(np.float32)

    @staticmethod
    def resample_probs_to_native(
        probs_chwd: np.ndarray,
        native_shape_hwz: Tuple[int, int, int],
    ) -> np.ndarray:
        """Reamostra mapa 4D de probabilidades `(C, H_res, W_res, Z)` para `(C, H_nat, W_nat, Z)`."""
        num_classes, h_res, w_res, z_res = probs_chwd.shape
        h_nat, w_nat, z_nat = native_shape_hwz
        if (h_res, w_res, z_res) == (h_nat, w_nat, z_nat):
            return probs_chwd

        scale_h = h_nat / float(h_res)
        scale_w = w_nat / float(w_res)
        scale_z = z_nat / float(z_res)
        out = np.empty((num_classes, h_nat, w_nat, z_nat), dtype=np.float32)
        for cls_idx in range(num_classes):
            resized = zoom(probs_chwd[cls_idx], zoom=(scale_h, scale_w, scale_z), order=1)
            # Corrige eventuais diferenças de ±1 voxel por arredondamento de zoom
            out[cls_idx] = VolumePreprocessor._fit_exact_shape_3d(resized, native_shape_hwz)
        return out

    @staticmethod
    def _fit_exact_shape_3d(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Ajusta array 3D em ±1 voxel caso o arredondamento de `scipy.ndimage.zoom` divirja."""
        if arr.shape == target_shape:
            return arr
        out = np.zeros(target_shape, dtype=arr.dtype)
        s0 = min(arr.shape[0], target_shape[0])
        s1 = min(arr.shape[1], target_shape[1])
        s2 = min(arr.shape[2], target_shape[2])
        out[:s0, :s1, :s2] = arr[:s0, :s1, :s2]
        return out

    @staticmethod
    def pad_or_center_crop_2d(
        arr: np.ndarray,
        target_size: int,
        pad_value: float = 0.0,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Aplica padding simétrico ou recorte central para `(target_size, target_size)`.

        Retorna também os deslocamentos `(pad_top, pad_left, crop_top, crop_left)` para
        reconstrução exata da geometria original após inferência.
        """
        h, w = arr.shape
        pad_top = pad_left = crop_top = crop_left = 0

        if h < target_size:
            pad_total = target_size - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            arr = np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=pad_value)
        elif h > target_size:
            crop_top = (h - target_size) // 2
            arr = arr[crop_top : crop_top + target_size, :]

        if w < target_size:
            pad_total = target_size - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            arr = np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=pad_value)
        elif w > target_size:
            crop_left = (w - target_size) // 2
            arr = arr[:, crop_left : crop_left + target_size]

        return arr, (pad_top, pad_left, crop_top, crop_left)

    def prepare_volume(self, record: VolumeRecord, task_mode: TaskMode) -> PreparedVolume:
        """Executa leitura, normalização, padronização de máscara e reamostragem física."""
        img_native, img_nii = NiftiVolumeIO.load_volume(record.image_path, is_mask=False)
        zooms_raw = img_nii.header.get_zooms()[:3]
        spacing_mm = (float(zooms_raw[0]), float(zooms_raw[1]), float(zooms_raw[2]))

        img_norm = self.normalize_intensity(img_native)
        img_res = self.resample_inplane(img_norm, spacing_xy=(spacing_mm[0], spacing_mm[1]), is_mask=False)

        mask_native = LabelStandardizer.standardize_record_mask(record, task_mode=task_mode)
        mask_res = None
        if mask_native is not None:
            if mask_native.shape != img_native.shape:
                raise ValueError(
                    f"Shape divergente entre imagem {img_native.shape} e máscara {mask_native.shape} "
                    f"em {record.patient_id}"
                )
            mask_res = self.resample_inplane(mask_native, spacing_xy=(spacing_mm[0], spacing_mm[1]), is_mask=True)
            # Garante alinhamento exato entre imagem e máscara reamostradas
            if mask_res.shape != img_res.shape:
                mask_res = self._fit_exact_shape_3d(mask_res, img_res.shape)

        return PreparedVolume(
            patient_id=record.patient_id,
            cohort=record.cohort,
            image_resampled=img_res,
            mask_resampled=mask_res,
            mask_native=mask_native,
            native_shape=img_native.shape,
            native_spacing_mm=spacing_mm,
            native_affine=img_nii.affine.copy(),
            native_header_pixdim=img_nii.header["pixdim"].copy(),
            target_spacing_mm=self.target_spacing_mm,
        )


class CardiacAugmentor:
    """Pipeline de Data Augmentation 2D orientado à física de LGE-CMR (apenas treino).

    Implementado em NumPy/SciPy para independência total de versões externas e
    reprodutibilidade determinística. Nunca aplica flip horizontal/vertical
    para preservar a quiralidade anatômica entre VE e VD.
    """

    def __init__(
        self,
        max_rotation_deg: float = 25.0,
        scale_range: Tuple[float, float] = (0.80, 1.25),
        max_shift_frac: float = 0.12,
        elastic_prob: float = 0.30,
        multires_blur_prob: float = 0.40,
        multires_sigma_range: Tuple[float, float] = (0.5, 1.6),
        intensity_prob: float = 0.50,
        noise_prob: float = 0.40,
    ) -> None:
        self.max_rotation_deg = max_rotation_deg
        self.scale_range = scale_range
        self.max_shift_frac = max_shift_frac
        self.elastic_prob = elastic_prob
        self.multires_blur_prob = multires_blur_prob
        self.multires_sigma_range = multires_sigma_range
        self.intensity_prob = intensity_prob
        self.noise_prob = noise_prob

    def __call__(
        self,
        image_2d: np.ndarray,
        mask_2d: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica transformações geométricas sincronizadas e perturbações fotométricas."""
        img = image_2d.astype(np.float32, copy=True)
        msk = mask_2d.astype(np.int16, copy=True)
        h, w = img.shape

        # 1. Transformação afim combinada (rotação + escala + translação) em passo único
        angle_rad = math.radians(float(rng.uniform(-self.max_rotation_deg, self.max_rotation_deg))) if rng.random() < 0.75 else 0.0
        scale = float(rng.uniform(*self.scale_range)) if rng.random() < 0.60 else 1.0
        shift_y = float(rng.uniform(-self.max_shift_frac, self.max_shift_frac) * h) if rng.random() < 0.60 else 0.0
        shift_x = float(rng.uniform(-self.max_shift_frac, self.max_shift_frac) * w) if rng.random() < 0.60 else 0.0

        if angle_rad != 0.0 or scale != 1.0 or shift_y != 0.0 or shift_x != 0.0:
            cos_a = math.cos(angle_rad) / scale
            sin_a = math.sin(angle_rad) / scale
            matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
            center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=np.float64)
            offset = center - matrix @ center + np.array([shift_y, shift_x], dtype=np.float64)
            img = affine_transform(img, matrix, offset=offset, order=1, mode="constant", cval=0.0).astype(np.float32)
            msk = affine_transform(msk, matrix, offset=offset, order=0, mode="constant", cval=0).astype(np.int16)

        # 2. Deformação elástica leve (inspirada em Hoh et al., 2024)
        if rng.random() < self.elastic_prob:
            alpha = float(rng.uniform(4.0, 12.0))
            sigma = float(rng.uniform(8.0, 14.0))
            dy = gaussian_filter(rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), sigma=sigma) * alpha
            dx = gaussian_filter(rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), sigma=sigma) * alpha
            grid_y, grid_x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
            coords = np.array([grid_y + dy, grid_x + dx])
            img = map_coordinates(img, coords, order=1, mode="constant", cval=0.0).astype(np.float32)
            msk = np.rint(map_coordinates(msk, coords, order=0, mode="constant", cval=0)).astype(np.int16)

        # 3. Simulação multi-resolução (desfoque gaussiano) para generalização ao HU-UFJF (Hoh et al., 2024)
        if rng.random() < self.multires_blur_prob:
            blur_sigma = float(rng.uniform(*self.multires_sigma_range))
            img = gaussian_filter(img, sigma=blur_sigma).astype(np.float32)

        # 4. Perturbação de brilho, contraste e gama apenas na imagem
        if rng.random() < self.intensity_prob:
            contrast = float(rng.uniform(0.85, 1.15))
            brightness = float(rng.uniform(-0.15, 0.15))
            img = img * contrast + brightness

        if rng.random() < 0.30:
            gamma = float(rng.uniform(0.80, 1.25))
            img_min = float(img.min())
            img_span = float(img.max() - img_min) + 1e-6
            scaled = np.clip((img - img_min) / img_span, 0.0, 1.0)
            img = (np.power(scaled, gamma) * img_span + img_min).astype(np.float32)

        # 5. Ruído gaussiano aditivo
        if rng.random() < self.noise_prob:
            noise_std = float(rng.uniform(0.01, 0.08))
            img = (img + rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)).astype(np.float32)

        return np.clip(img, -5.0, 5.0).astype(np.float32), msk


class CmrMultiCohortLoader:
    """Indexa volumes rotulados do dataset `robertgvds/cmr-multi-lge-sax` (SAX_TR + SAX_VAL)."""

    SUBSETS: Tuple[str, ...] = ("SAX_TR", "SAX_VAL")

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def discover_records(self) -> List[VolumeRecord]:
        """Localiza pares `(image, anno)` aceitando `.nii` e `.nii.gz`."""
        records: List[VolumeRecord] = []
        for subset in self.SUBSETS:
            img_dir = self.root_dir / subset / "image"
            anno_dir = self.root_dir / subset / "anno"
            if not img_dir.exists() or not anno_dir.exists():
                continue

            anno_map = {
                NiftiVolumeIO.strip_suffix(p.name): p
                for p in sorted(anno_dir.iterdir())
                if NiftiVolumeIO.is_nifti_file(p)
            }
            for img_path in sorted(img_dir.iterdir()):
                if not NiftiVolumeIO.is_nifti_file(img_path):
                    continue
                case_id = NiftiVolumeIO.strip_suffix(img_path.name)
                if case_id not in anno_map:
                    LOGGER.warning("Máscara ausente para %s em %s; ignorando.", case_id, anno_dir)
                    continue
                records.append(
                    VolumeRecord(
                        patient_id=case_id,
                        cohort="cmr_multi",
                        image_path=img_path,
                        mask_path=anno_map[case_id],
                        mask_format="cmr_5class",
                    )
                )

        if not records:
            raise FileNotFoundError(f"Nenhum par (image, anno) CMR-MULTI encontrado em {self.root_dir}")
        return sorted(records, key=lambda r: r.patient_id)

    @staticmethod
    def build_scar_presence_map(records: Sequence[VolumeRecord]) -> Dict[str, bool]:
        """Determina presença de fibrose (`label == 3`) por paciente para estratificação."""
        scar_map: Dict[str, bool] = {}
        for rec in records:
            assert rec.mask_path is not None
            mask_arr, _ = NiftiVolumeIO.load_volume(rec.mask_path, is_mask=True)
            scar_map[rec.patient_id] = bool(np.any(mask_arr == 3))
        return scar_map


class HuUfjfCohortLoader:
    """Indexa pacientes da coorte clínica privada `robertgvds/hu-ufjf-lge-sax`."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def discover_records(self) -> Tuple[List[VolumeRecord], List[VolumeRecord]]:
        """Retorna `(labeled_records, unlabeled_records)` padronizando busca `.nii`/`.nii.gz`."""
        nifti_root = self.root_dir / "nifti"
        if not nifti_root.exists():
            # Fallback caso o usuário forneça diretamente a pasta `nifti/`
            nifti_root = self.root_dir

        labeled: List[VolumeRecord] = []
        unlabeled: List[VolumeRecord] = []

        patient_dirs = sorted(
            (p for p in nifti_root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: (len(p.name), p.name),
        )
        for p_dir in patient_dirs:
            img_path = NiftiVolumeIO.find_nifti_by_stem(p_dir, "img")
            if img_path is None:
                continue

            seg5_path = NiftiVolumeIO.find_nifti_by_stem(p_dir, "seg_5class")
            seg_anat_path = NiftiVolumeIO.find_nifti_by_stem(p_dir, "seg_anat")
            seg_scar_path = NiftiVolumeIO.find_nifti_by_stem(p_dir, "seg_scar")

            if seg5_path is not None:
                labeled.append(
                    VolumeRecord(
                        patient_id=p_dir.name,
                        cohort="hu_ufjf",
                        image_path=img_path,
                        mask_path=seg5_path,
                        aux_scar_path=seg_scar_path,
                        mask_format="hu_5class",
                    )
                )
            elif seg_anat_path is not None:
                labeled.append(
                    VolumeRecord(
                        patient_id=p_dir.name,
                        cohort="hu_ufjf",
                        image_path=img_path,
                        mask_path=seg_anat_path,
                        aux_scar_path=seg_scar_path,
                        mask_format="hu_anat_4class",
                    )
                )
            else:
                unlabeled.append(
                    VolumeRecord(
                        patient_id=p_dir.name,
                        cohort="hu_ufjf",
                        image_path=img_path,
                        mask_path=None,
                        mask_format="unlabeled",
                    )
                )

        LOGGER.info(
            "Coorte HU-UFJF indexada em %s: %d pacientes rotulados | %d sem máscara",
            nifti_root,
            len(labeled),
            len(unlabeled),
        )
        return labeled, unlabeled


class CardiacSliceDataset:
    """Dataset 2D fatiado em memória compatível com `torch.utils.data.DataLoader`.

    Utiliza `PreparedVolume` já reamostrados a `1.0 mm × 1.0 mm`, mantendo
    separação estrita por paciente e evitando I/O repetido durante as épocas.
    """

    def __init__(
        self,
        volumes: Sequence[PreparedVolume],
        patch_size: int = 256,
        augmentor: Optional[CardiacAugmentor] = None,
        seed: int = 42,
        include_empty_slices: bool = True,
    ) -> None:
        self.patch_size = int(patch_size)
        self.augmentor = augmentor
        self.rng = np.random.default_rng(seed)
        self.samples: List[Tuple[np.ndarray, np.ndarray, str, int]] = []

        for vol in volumes:
            if vol.mask_resampled is None:
                continue
            num_slices = vol.image_resampled.shape[2]
            for z in range(num_slices):
                img_slc = vol.image_resampled[:, :, z]
                msk_slc = vol.mask_resampled[:, :, z]
                if not include_empty_slices and not np.any(msk_slc > 0):
                    continue
                img_pad, _ = VolumePreprocessor.pad_or_center_crop_2d(img_slc, self.patch_size, pad_value=0.0)
                msk_pad, _ = VolumePreprocessor.pad_or_center_crop_2d(msk_slc, self.patch_size, pad_value=0)
                self.samples.append((img_pad.astype(np.float32), msk_pad.astype(np.int16), vol.patient_id, z))

        if not self.samples:
            raise ValueError("Nenhuma fatia válida encontrada para construir CardiacSliceDataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[object, object]:
        import torch

        img, msk, _, _ = self.samples[index]
        if self.augmentor is not None:
            img, msk = self.augmentor(img, msk, self.rng)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img[None, :, :])).float()
        msk_tensor = torch.from_numpy(np.ascontiguousarray(msk)).long()
        return img_tensor, msk_tensor
