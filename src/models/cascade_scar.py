"""Pipeline POO da Cascata em 2 Etapas (Anatomia 4c -> ROI Miocárdio -> Especialista em Fibrose).

Implementa a Regra de Ouro da literatura LGE-CMR (2024–2026: ScarNet 2025,
MYOSAIQ MELBA 2026, Ramzan et al. ISBI 2026, Hoh et al. JCMR 2024):
  - **Etapa 1 (Anatomia — 4 classes):** Reutiliza diretamente o checkpoint já
    treinado do Exp 05A (`U-Net ResNet34 4c`) para localizar VE, Miocárdio Total
    e VD (ou treina do zero como fallback se nenhum checkpoint for informado).
  - **Etapa 2 (Especialista em Fibrose na ROI do Miocárdio):**
      * Treinada na união harmonizada **CMR-MULTI treino (35) + EMIDEC treino (100)
        = 135 volumes LGE reais**, desbloqueando os 100 pacientes do EMIDEC sem
        conflito pela ausência de VD!
      * Recorta a ROI do miocárdio em alta resolução no plano (`0.75 mm × 0.75 mm`,
        `160 × 160` px = `120 × 120 mm`).
      * Aplica **normalização robusta local relativa ao miocárdio (`mediana / MAD`)**
        — análoga ao critério clínico `n-SD` — eliminando o *domain shift* de
        intensidade entre scanners (PSIR negativo vs. MAG tórax inteiro do HU-UFJF).
      * Recebe entrada nativa 3 canais `(LGE_local, Prior_Mio, Prior_VE)` com
        perturbação morfológica aleatória dos *priors* no treino (impedindo
        dependência excessiva da borda exata da Etapa 1).
      * Calibra o limiar de probabilidade de fibrose na validação (`N=5`, `SEED=42`),
        funde na máscara final 5 classes (`int16`) restrita à zona anatômica do
        miocárdio (+2 px de tolerância) e exporta também o **mapa probabilístico
        contínuo de fibrose (`*_SCAR_PROB.nii.gz`, `float32`)**.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import (
    affine_transform,
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    label as cc_label,
    map_coordinates,
    rotate as nd_rotate,
    shift as nd_shift,
    zoom,
)

from src.data.cardiac_dataset import (
    CmrMultiCohortLoader,
    EmidecCohortLoader,
    HuUfjfCohortLoader,
    LabelStandardizer,
    NiftiVolumeIO,
    TaskMode,
    VolumePreprocessor,
    VolumeRecord,
)
from src.data.splits import BenchmarkSplitConfig, StratifiedSplitGenerator
from src.evaluation.metrics_3d import BinarySurfaceMetrics, CardiacMetrics3DEvaluator
from src.evaluation.visualize import SegmentationPanelVisualizer
from src.models.unet_resnet34 import (
    CardiacPostProcessor,
    CombinedSegmentationLoss,
    SlidingWindow2DInferer,
    UnetResNet34Config,
    UnetResNet34Pipeline,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CascadeScarConfig:
    """Hiperparâmetros imutáveis do Exp 06 (Cascata em 2 Etapas CMR-MULTI + EMIDEC)."""

    stage1_target_spacing_mm: float = 1.0
    stage1_patch_size: int = 256
    stage2_encoder_name: str = "resnet34"
    stage2_encoder_weights: str = "imagenet"
    stage2_spacing_mm: float = 0.75
    stage2_roi_size: int = 160
    stage2_epochs: int = 80
    stage2_warmup_epochs: int = 5
    stage2_batch_size: int = 32
    stage2_lr: float = 3e-4
    stage2_min_lr: float = 1e-6
    stage2_weight_decay: float = 1e-4
    stage2_ce_weight: float = 0.4
    stage2_dice_weight: float = 0.6
    stage2_class_weights: Tuple[float, float, float] = (0.4, 1.5, 4.5)
    stage2_tta_angles_deg: Tuple[float, ...] = (0.0, -10.0, 10.0)
    myo_fusion_dilation_px: int = 2
    min_scar_voxels: int = 8
    threshold_grid: Tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    seed: int = 42
    num_workers: int = 2

    def to_serializable_dict(self) -> Dict[str, object]:
        """Serializa configuração para JSON."""
        raw = asdict(self)
        raw["stage2_class_weights"] = list(self.stage2_class_weights)
        raw["stage2_tta_angles_deg"] = list(self.stage2_tta_angles_deg)
        raw["threshold_grid"] = list(self.threshold_grid)
        return raw


@dataclass
class RoiVolumePack:
    """Volume reamostrado para a Etapa 2 com metadados exatos do recorte ROI."""

    patient_id: str
    cohort: str
    img_roi_3d: np.ndarray            # (R, R, Z) float32 normalizado localmente ao miocárdio
    myo_prior_roi_3d: np.ndarray      # (R, R, Z) float32 binário {0, 1}
    lv_prior_roi_3d: np.ndarray       # (R, R, Z) float32 binário {0, 1}
    target_roi_3d: Optional[np.ndarray]  # (R, R, Z) int16 {0=Fora, 1=MioSaudavel, 2=Scar}
    resampled_shape: Tuple[int, int, int]
    crop_box_yx: Tuple[int, int, int, int]  # (y0, y1, x0, x1) no espaço reamostrado (pode ter padding)
    native_shape: Tuple[int, int, int]
    native_affine: np.ndarray
    native_header_pixdim: np.ndarray


class MyocardiumRoiExtractor:
    """Reamostra para alta resolução (`0.75 mm`), normaliza relativo ao miocárdio e recorta ROI."""

    def __init__(self, target_spacing_mm: float = 0.75, roi_size: int = 160) -> None:
        self.target_spacing_mm = float(target_spacing_mm)
        self.roi_size = int(roi_size)

    @staticmethod
    def _resample_xy(volume: np.ndarray, spacing_xy: Tuple[float, float], target_mm: float, is_mask: bool) -> np.ndarray:
        """Reamostra apenas XY preservando fatias Z originais."""
        sx = float(spacing_xy[0]) / target_mm
        sy = float(spacing_xy[1]) / target_mm
        if abs(sx - 1.0) < 1e-3 and abs(sy - 1.0) < 1e-3:
            return volume.astype(np.int16 if is_mask else np.float32, copy=True)
        out = zoom(volume, zoom=(sx, sy, 1.0), order=(0 if is_mask else 1))
        return np.rint(out).astype(np.int16) if is_mask else out.astype(np.float32)

    @staticmethod
    def normalize_relative_to_myocardium(
        img_res: np.ndarray,
        myo_mask: np.ndarray,
        lv_mask: np.ndarray,
    ) -> np.ndarray:
        """Normaliza intensidades com `mediana / MAD` do miocárdio (invariante a scanner/PSIR/MAG).

        Em LGE-CMR, a fibrose é clinicamente definida por desvio acima do miocárdio
        remoto (`n-SD`). Centralizar pela mediana e escalar pelo MAD do miocárdio
        coloca miocárdio saudável próximo de `0.0` e cicatriz entre `+2.0` e `+6.0`
        tanto no CMR-MULTI quanto no EMIDEC e no HU-UFJF.
        """
        vol = img_res.astype(np.float32, copy=False)
        if int(np.count_nonzero(myo_mask)) >= 50:
            ref_vals = vol[myo_mask > 0]
        elif int(np.count_nonzero(lv_mask)) >= 50:
            ref_vals = vol[lv_mask > 0]
        else:
            ref_vals = vol.ravel()

        p1, p995 = np.percentile(ref_vals, (1.0, 99.5))
        ref_clean = np.clip(ref_vals, float(p1), float(p995))
        med = float(np.median(ref_clean))
        mad = float(np.median(np.abs(ref_clean - med)))
        std = float(np.std(ref_clean))
        scale = max(1.4826 * mad, 0.5 * std, 1e-4)
        return np.clip((vol - med) / scale, -4.0, 8.0).astype(np.float32)

    @staticmethod
    def _extract_centered_crop_3d(
        vol: np.ndarray,
        cy: int,
        cx: int,
        roi_size: int,
        pad_value: float = 0.0,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extrai recorte 3D `(roi_size, roi_size, Z)` centrado em `(cy, cx)` com padding seguro."""
        h, w, z = vol.shape
        half = roi_size // 2
        y0 = cy - half
        x0 = cx - half
        y1 = y0 + roi_size
        x1 = x0 + roi_size

        out = np.full((roi_size, roi_size, z), pad_value, dtype=vol.dtype)
        src_y0 = max(0, y0)
        src_x0 = max(0, x0)
        src_y1 = min(h, y1)
        src_x1 = min(w, x1)

        dst_y0 = src_y0 - y0
        dst_x0 = src_x0 - x0
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1, :] = vol[src_y0:src_y1, src_x0:src_x1, :]
        return out, (y0, y1, x0, x1)

    @staticmethod
    def paste_roi_probs_to_native(
        roi_probs_chwd: np.ndarray,
        pack: RoiVolumePack,
    ) -> np.ndarray:
        """Insere probabilidades da ROI `(C, R, R, Z)` de volta no volume nativo `(C, H_nat, W_nat, Z)`."""
        num_classes = roi_probs_chwd.shape[0]
        h_res, w_res, z_res = pack.resampled_shape
        y0, y1, x0, x1 = pack.crop_box_yx

        full_probs = np.zeros((num_classes, h_res, w_res, z_res), dtype=np.float32)
        # Fora da ROI, probabilidade = 1.0 na classe 0 (fundo/fora do miocárdio)
        full_probs[0, :, :, :] = 1.0

        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = min(h_res, y1)
        dst_x1 = min(w_res, x1)

        src_y0 = dst_y0 - y0
        src_x0 = dst_x0 - x0
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        if dst_y1 > dst_y0 and dst_x1 > dst_x0:
            full_probs[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = roi_probs_chwd[:, src_y0:src_y1, src_x0:src_x1, :]

        return VolumePreprocessor.resample_probs_to_native(full_probs, pack.native_shape)

    def build_roi_pack(
        self,
        record: VolumeRecord,
        lv_prior_native: np.ndarray,
        myo_prior_native: np.ndarray,
        gt_stage2_native: Optional[np.ndarray] = None,
    ) -> RoiVolumePack:
        """Constrói `RoiVolumePack` pronto para treino ou inferência na Etapa 2."""
        img_native, img_nii = NiftiVolumeIO.load_volume(record.image_path, is_mask=False)
        zooms = tuple(float(z) for z in img_nii.header.get_zooms()[:3])

        img_res = self._resample_xy(img_native, (zooms[0], zooms[1]), self.target_spacing_mm, is_mask=False)
        lv_res = self._resample_xy(lv_prior_native, (zooms[0], zooms[1]), self.target_spacing_mm, is_mask=True)
        myo_res = self._resample_xy(myo_prior_native, (zooms[0], zooms[1]), self.target_spacing_mm, is_mask=True)

        # Ajusta possível diferença de ±1 voxel por arredondamento
        lv_res = VolumePreprocessor._fit_exact_shape_3d(lv_res, img_res.shape)
        myo_res = VolumePreprocessor._fit_exact_shape_3d(myo_res, img_res.shape)

        img_norm = self.normalize_relative_to_myocardium(img_res, myo_res, lv_res)

        anchor = (myo_res > 0) | (lv_res > 0)
        if np.any(anchor):
            ys, xs, _ = np.where(anchor)
            cy = int(round(0.5 * (int(ys.min()) + int(ys.max()))))
            cx = int(round(0.5 * (int(xs.min()) + int(xs.max()))))
        else:
            cy, cx = img_res.shape[0] // 2, img_res.shape[1] // 2

        img_roi, crop_box = self._extract_centered_crop_3d(img_norm, cy, cx, self.roi_size, pad_value=0.0)
        myo_roi, _ = self._extract_centered_crop_3d((myo_res > 0).astype(np.float32), cy, cx, self.roi_size, pad_value=0.0)
        lv_roi, _ = self._extract_centered_crop_3d((lv_res > 0).astype(np.float32), cy, cx, self.roi_size, pad_value=0.0)

        target_roi = None
        if gt_stage2_native is not None:
            gt_res = self._resample_xy(gt_stage2_native, (zooms[0], zooms[1]), self.target_spacing_mm, is_mask=True)
            gt_res = VolumePreprocessor._fit_exact_shape_3d(gt_res, img_res.shape)
            # Mapeia rótulo unificado (2=MioSaudavel, 3=Scar) para alvo 3 classes (0=Fora, 1=MioSaud, 2=Scar)
            tgt = np.zeros_like(gt_res, dtype=np.int16)
            tgt[gt_res == 2] = 1
            tgt[gt_res == 3] = 2
            target_roi, _ = self._extract_centered_crop_3d(tgt, cy, cx, self.roi_size, pad_value=0)

        return RoiVolumePack(
            patient_id=record.patient_id,
            cohort=record.cohort,
            img_roi_3d=img_roi.astype(np.float32),
            myo_prior_roi_3d=myo_roi.astype(np.float32),
            lv_prior_roi_3d=lv_roi.astype(np.float32),
            target_roi_3d=target_roi,
            resampled_shape=img_res.shape,
            crop_box_yx=crop_box,
            native_shape=img_native.shape,
            native_affine=img_nii.affine.copy(),
            native_header_pixdim=img_nii.header["pixdim"].copy(),
        )


class Stage2RoiAugmentor:
    """Augmentation 2D para a Etapa 2 com perturbação morfológica dos *priors* anatômicos."""

    def __call__(
        self,
        img_2d: np.ndarray,
        myo_prior: np.ndarray,
        lv_prior: np.ndarray,
        target_2d: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aplica transformações sincronizadas + ruído/blur no LGE + perturbação dos *priors*."""
        img = img_2d.astype(np.float32, copy=True)
        myo = myo_prior.astype(np.float32, copy=True)
        lv = lv_prior.astype(np.float32, copy=True)
        tgt = target_2d.astype(np.int16, copy=True)
        h, w = img.shape

        # 1. Transformação afim sincronizada (rotação ±30°, escala 0.85-1.20x, translação ±10%)
        angle_rad = math.radians(float(rng.uniform(-30.0, 30.0))) if rng.random() < 0.80 else 0.0
        scale = float(rng.uniform(0.85, 1.20)) if rng.random() < 0.60 else 1.0
        shift_y = float(rng.uniform(-0.10, 0.10) * h) if rng.random() < 0.60 else 0.0
        shift_x = float(rng.uniform(-0.10, 0.10) * w) if rng.random() < 0.60 else 0.0

        if angle_rad != 0.0 or scale != 1.0 or shift_y != 0.0 or shift_x != 0.0:
            cos_a = math.cos(angle_rad) / scale
            sin_a = math.sin(angle_rad) / scale
            matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
            center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=np.float64)
            offset = center - matrix @ center + np.array([shift_y, shift_x], dtype=np.float64)
            img = affine_transform(img, matrix, offset=offset, order=1, mode="constant", cval=0.0).astype(np.float32)
            myo = (affine_transform(myo, matrix, offset=offset, order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)
            lv = (affine_transform(lv, matrix, offset=offset, order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)
            tgt = affine_transform(tgt, matrix, offset=offset, order=0, mode="constant", cval=0).astype(np.int16)

        # 2. Deformação elástica leve sincronizada
        if rng.random() < 0.30:
            alpha = float(rng.uniform(3.0, 9.0))
            sigma = float(rng.uniform(7.0, 12.0))
            dy = gaussian_filter(rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), sigma=sigma) * alpha
            dx = gaussian_filter(rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), sigma=sigma) * alpha
            gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
            coords = np.array([gy + dy, gx + dx])
            img = map_coordinates(img, coords, order=1, mode="constant", cval=0.0).astype(np.float32)
            myo = (map_coordinates(myo, coords, order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)
            lv = (map_coordinates(lv, coords, order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)
            tgt = np.rint(map_coordinates(tgt, coords, order=0, mode="constant", cval=0)).astype(np.int16)

        # 3. Perturbação realista dos priors anatômicos (simula imperfeições da Etapa 1 no teste!)
        if rng.random() < 0.65:
            op = int(rng.integers(0, 3))
            iters = int(rng.integers(1, 3))
            if op == 0 and np.any(myo > 0.5):
                myo = binary_dilation(myo > 0.5, iterations=iters).astype(np.float32)
            elif op == 1 and np.any(myo > 0.5):
                eroded = binary_erosion(myo > 0.5, iterations=1)
                if np.any(eroded):
                    myo = eroded.astype(np.float32)
            # Pequeno deslocamento independente de ±2 px nos canais de prior
            dy_p = float(rng.uniform(-2.0, 2.0))
            dx_p = float(rng.uniform(-2.0, 2.0))
            myo = (nd_shift(myo, shift=(dy_p, dx_p), order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)
            lv = (nd_shift(lv, shift=(dy_p, dx_p), order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)

        # Dropout ocasional dos priors para forçar a rede a aprender textura real do LGE
        if rng.random() < 0.10:
            myo = np.zeros_like(myo)
            lv = np.zeros_like(lv)

        # 4. Simulação multi-resolução + contraste + ruído apenas no canal LGE
        if rng.random() < 0.40:
            img = gaussian_filter(img, sigma=float(rng.uniform(0.4, 1.4))).astype(np.float32)
        if rng.random() < 0.55:
            img = img * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.20, 0.20))
        if rng.random() < 0.40:
            img = (img + rng.normal(0.0, float(rng.uniform(0.02, 0.12)), size=img.shape).astype(np.float32)).astype(np.float32)

        return np.clip(img, -4.0, 8.0).astype(np.float32), myo, lv, tgt


class Stage2RoiSliceDataset:
    """Dataset 2D da Etapa 2 (`135 volumes LGE` -> fatias 3 canais na ROI do miocárdio)."""

    def __init__(
        self,
        packs: Sequence[RoiVolumePack],
        augmentor: Optional[Stage2RoiAugmentor] = None,
        seed: int = 42,
        oversample_scar_slices: int = 2,
    ) -> None:
        self.augmentor = augmentor
        self.rng = np.random.default_rng(seed)
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for pack in packs:
            if pack.target_roi_3d is None:
                continue
            for z in range(pack.img_roi_3d.shape[2]):
                myo_slc = pack.myo_prior_roi_3d[:, :, z]
                tgt_slc = pack.target_roi_3d[:, :, z]
                # Mantém apenas fatias que contêm miocárdio (foco total na anatomia do VE!)
                if not np.any(myo_slc > 0.5) and not np.any(tgt_slc > 0):
                    continue
                sample = (
                    pack.img_roi_3d[:, :, z],
                    myo_slc,
                    pack.lv_prior_roi_3d[:, :, z],
                    tgt_slc,
                )
                self.samples.append(sample)
                # Oversampling leve em fatias com fibrose positiva para balancear classes
                if oversample_scar_slices > 1 and np.any(tgt_slc == 2):
                    for _ in range(oversample_scar_slices - 1):
                        self.samples.append(sample)

        if not self.samples:
            raise ValueError("Nenhuma fatia ROI válida encontrada para a Etapa 2.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[object, object]:
        import torch

        img, myo, lv, tgt = self.samples[index]
        if self.augmentor is not None:
            img, myo, lv, tgt = self.augmentor(img, myo, lv, tgt, self.rng)
        x = np.stack([img, myo, lv], axis=0).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(tgt)).long()


class CascadeScarPipeline:
    """Orquestrador POO ponta a ponta da Cascata em 2 Etapas (CMR-MULTI + EMIDEC -> HU-UFJF)."""

    def __init__(
        self,
        config: CascadeScarConfig,
        cmr_dir: Path,
        output_dir: Path,
        emidec_dir: Optional[Path] = None,
        hu_dir: Optional[Path] = None,
        stage1_checkpoint: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.cmr_dir = cmr_dir
        self.emidec_dir = emidec_dir
        self.hu_dir = hu_dir
        self.stage1_checkpoint = stage1_checkpoint
        self.output_dir = output_dir

        self.stage1_preproc = VolumePreprocessor(target_spacing_mm=config.stage1_target_spacing_mm)
        self.stage1_postproc = CardiacPostProcessor(task_mode=TaskMode.ANATOMY_4C)
        self.stage1_inferer = SlidingWindow2DInferer(
            num_classes=4,
            patch_size=config.stage1_patch_size,
            overlap=0.5,
            tta_angles_deg=(0.0, -8.0, 8.0),
        )
        self.roi_extractor = MyocardiumRoiExtractor(
            target_spacing_mm=config.stage2_spacing_mm,
            roi_size=config.stage2_roi_size,
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _load_or_train_stage1(self, device: object, dry_run: bool = False) -> object:
        """Carrega checkpoint pré-treinado do Exp 05A (4c) ou treina Etapa 1 como fallback."""
        import segmentation_models_pytorch as smp
        import torch

        if self.stage1_checkpoint is not None and self.stage1_checkpoint.is_file():
            LOGGER.info("Carregando pesos pré-treinados da Etapa 1 (Exp 05A 4c): %s", self.stage1_checkpoint)
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=1,
                classes=4,
                decoder_use_batchnorm=True,
            ).to(device)
            ckpt = torch.load(str(self.stage1_checkpoint), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model

        LOGGER.info("Nenhum checkpoint da Etapa 1 fornecido; treinando Etapa 1 (4c) no CMR-MULTI...")
        stage1_out = self.output_dir / "_stage1_trained"
        s1_cfg = UnetResNet34Config(
            task_mode=TaskMode.ANATOMY_4C,
            epochs=2 if dry_run else 100,
            seed=self.config.seed,
        )
        s1_pipe = UnetResNet34Pipeline(config=s1_cfg, cmr_dir=self.cmr_dir, output_dir=stage1_out)
        s1_pipe.run(dry_run=dry_run)
        ckpt_path = stage1_out / "checkpoint_best.pth"
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=4,
            decoder_use_batchnorm=True,
        ).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def _predict_stage1_4c(
        self,
        stage1_model: Optional[object],
        device: Optional[object],
        record: VolumeRecord,
        mock_mode: bool = False,
    ) -> np.ndarray:
        """Executa Etapa 1 (4c) retornando máscara anatômica nativa `(0=Bg,1=VE,2=MioTotal,3=VD)`."""
        prepared = self.stage1_preproc.prepare_volume(record, task_mode=TaskMode.ANATOMY_4C)
        if mock_mode:
            if prepared.mask_native is not None:
                return self.stage1_postproc.apply(prepared.mask_native.copy())
            return np.zeros(prepared.native_shape, dtype=np.int16)
        assert stage1_model is not None and device is not None
        return self.stage1_inferer.predict_volume(
            model=stage1_model,
            device=device,
            prepared=prepared,
            post_processor=self.stage1_postproc,
            use_tta=True,
        )

    def _predict_stage2_probs_native(
        self,
        stage2_model: Optional[object],
        device: Optional[object],
        pack: RoiVolumePack,
        use_tta: bool = True,
        mock_mode: bool = False,
    ) -> np.ndarray:
        """Executa Etapa 2 na ROI com TTA rotacional e devolve probabilidades `(3, H_nat, W_nat, Z)`."""
        r = self.config.stage2_roi_size
        z_dim = pack.img_roi_3d.shape[2]
        if mock_mode:
            roi_probs = np.zeros((3, r, r, z_dim), dtype=np.float32)
            roi_probs[0] = 1.0
            if pack.target_roi_3d is not None:
                for cls_idx in (0, 1, 2):
                    roi_probs[cls_idx] = (pack.target_roi_3d == cls_idx).astype(np.float32)
            return self.roi_extractor.paste_roi_probs_to_native(roi_probs, pack)

        import torch

        assert stage2_model is not None and device is not None
        stage2_model.eval()
        angles = self.config.stage2_tta_angles_deg if use_tta else (0.0,)
        roi_probs = np.zeros((3, r, r, z_dim), dtype=np.float32)

        for z in range(z_dim):
            img_slc = pack.img_roi_3d[:, :, z]
            myo_slc = pack.myo_prior_roi_3d[:, :, z]
            lv_slc = pack.lv_prior_roi_3d[:, :, z]
            if not np.any(myo_slc > 0.5):
                roi_probs[0, :, :, z] = 1.0
                continue

            slc_sum = np.zeros((3, r, r), dtype=np.float32)
            for angle in angles:
                if abs(angle) < 1e-3:
                    x_in = np.stack([img_slc, myo_slc, lv_slc], axis=0)[None, :, :, :].astype(np.float32)
                else:
                    ri = nd_rotate(img_slc, angle=angle, reshape=False, order=1, mode="constant", cval=0.0)
                    rm = (nd_rotate(myo_slc, angle=angle, reshape=False, order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)
                    rl = (nd_rotate(lv_slc, angle=angle, reshape=False, order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)
                    x_in = np.stack([ri, rm, rl], axis=0)[None, :, :, :].astype(np.float32)

                tensor_in = torch.from_numpy(x_in).to(device)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", enabled=(str(device).startswith("cuda"))):
                        logits = stage2_model(tensor_in)
                    p = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

                if abs(angle) >= 1e-3:
                    p = np.stack(
                        [nd_rotate(p[c], angle=-angle, reshape=False, order=1, mode="constant", cval=0.0) for c in range(3)],
                        axis=0,
                    )
                slc_sum += p
            roi_probs[:, :, :, z] = slc_sum / float(len(angles))

        return self.roi_extractor.paste_roi_probs_to_native(roi_probs, pack)

    def fuse_stage1_and_stage2(
        self,
        stage1_4c_native: np.ndarray,
        stage2_probs_native: np.ndarray,
        scar_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Funde anatomia 4c da Etapa 1 com probabilidades de fibrose da Etapa 2 em máscara 5c."""
        fused = np.zeros_like(stage1_4c_native, dtype=np.int16)
        fused[stage1_4c_native == 1] = 1  # VE
        fused[stage1_4c_native == 2] = 2  # Miocárdio (inicialmente saudável)
        fused[stage1_4c_native == 3] = 4  # VD (remapeado 3 -> 4)

        scar_prob = np.clip(stage2_probs_native[2], 0.0, 1.0).astype(np.float32)
        myo_zone = stage1_4c_native == 2
        if not np.any(myo_zone):
            return fused, np.zeros_like(scar_prob, dtype=np.float32)

        # Restringe busca da fibrose à ROI do miocárdio da Etapa 1 (+ tolerância XY leve)
        xy_struct = np.zeros((3, 3, 3), dtype=bool)
        xy_struct[:, :, 1] = True
        allowed_zone = binary_dilation(myo_zone, structure=xy_struct, iterations=self.config.myo_fusion_dilation_px)
        # Nunca sobrescreve VD com fibrose do VE
        allowed_zone = allowed_zone & (stage1_4c_native != 3)

        scar_prob_masked = np.where(allowed_zone, scar_prob, 0.0).astype(np.float32)
        scar_bin = scar_prob_masked >= float(scar_threshold)

        # Limpa micro-ruído de fibrose (< min_scar_voxels) para proteger especificidade em saudáveis
        if np.any(scar_bin):
            struct_26 = np.ones((3, 3, 3), dtype=np.int32)
            labeled, n_comp = cc_label(scar_bin, structure=struct_26)
            total_scar = int(np.count_nonzero(scar_bin))
            for cid in range(1, n_comp + 1):
                comp = labeled == cid
                vox = int(np.count_nonzero(comp))
                if vox < self.config.min_scar_voxels and (total_scar < self.config.min_scar_voxels * 2 or not np.any(comp & myo_zone)):
                    scar_bin[comp] = False

        fused[scar_bin] = 3
        return fused, scar_prob_masked

    def _build_stage2_training_packs(
        self,
        cmr_train_records: Sequence[VolumeRecord],
        emidec_records: Sequence[VolumeRecord],
    ) -> List[RoiVolumePack]:
        """Constrói packs ROI de alta resolução a partir de CMR-MULTI (35) + EMIDEC (100)."""
        packs: List[RoiVolumePack] = []
        for rec in list(cmr_train_records) + list(emidec_records):
            gt_std = LabelStandardizer.standardize_record_mask(rec, task_mode=TaskMode.FULL_5C)
            assert gt_std is not None
            lv_gt = (gt_std == 1).astype(np.int16)
            myo_gt = ((gt_std == 2) | (gt_std == 3)).astype(np.int16)
            packs.append(
                self.roi_extractor.build_roi_pack(
                    record=rec,
                    lv_prior_native=lv_gt,
                    myo_prior_native=myo_gt,
                    gt_stage2_native=gt_std,
                )
            )
        LOGGER.info("Packs ROI da Etapa 2 preparados: %d volumes totais.", len(packs))
        return packs

    def _train_stage2(
        self,
        train_packs: Sequence[RoiVolumePack],
        val_records: Sequence[VolumeRecord],
        val_stage1_preds: Dict[str, np.ndarray],
        val_roi_packs: Dict[str, RoiVolumePack],
        val_gt_5c: Dict[str, np.ndarray],
        device: object,
        dry_run: bool = False,
    ) -> Tuple[object, float]:
        """Treina Etapa 2 especialista em fibrose e calibra limiar ótimo na validação."""
        import segmentation_models_pytorch as smp
        import torch
        from torch.utils.data import DataLoader

        train_ds = Stage2RoiSliceDataset(
            packs=train_packs,
            augmentor=Stage2RoiAugmentor(),
            seed=self.config.seed,
            oversample_scar_slices=2,
        )
        batch_size = min(self.config.stage2_batch_size, len(train_ds))
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if dry_run else self.config.num_workers,
            pin_memory=(str(device).startswith("cuda")),
            drop_last=False,
        )

        model = smp.Unet(
            encoder_name=self.config.stage2_encoder_name,
            encoder_weights=self.config.stage2_encoder_weights,
            in_channels=3,
            classes=3,
            decoder_use_batchnorm=True,
        ).to(device)

        criterion = CombinedSegmentationLoss(
            class_weights=self.config.stage2_class_weights,
            ce_weight=self.config.stage2_ce_weight,
            dice_weight=self.config.stage2_dice_weight,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.stage2_lr,
            weight_decay=self.config.stage2_weight_decay,
        )

        total_epochs = 1 if dry_run else self.config.stage2_epochs
        warmup = min(self.config.stage2_warmup_epochs, max(1, total_epochs // 5))

        def lr_lambda(ep_idx: int) -> float:
            if ep_idx < warmup:
                return float(ep_idx + 1) / float(warmup)
            progress = float(ep_idx - warmup) / float(max(1, total_epochs - warmup))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_r = self.config.stage2_min_lr / self.config.stage2_lr
            return min_r + (1.0 - min_r) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

        best_score = -1.0
        ckpt_path = self.output_dir / "stage2_checkpoint_best.pth"
        history: List[Dict[str, float]] = []

        for epoch in range(1, total_epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.item())
                n_batches += 1
                if dry_run:
                    break

            scheduler.step()
            avg_loss = running_loss / max(1, n_batches)

            should_eval = dry_run or epoch <= 8 or (epoch % 2 == 0) or (epoch == total_epochs)
            if should_eval:
                scar_dices: List[float] = []
                board_scars: List[float] = []
                for rec in val_records:
                    probs = self._predict_stage2_probs_native(
                        model, device, val_roi_packs[rec.patient_id], use_tta=False, mock_mode=False
                    )
                    fused, _ = self.fuse_stage1_and_stage2(val_stage1_preds[rec.patient_id], probs, scar_threshold=0.50)
                    gt = val_gt_5c[rec.patient_id]
                    d_scar = BinarySurfaceMetrics.compute_dice(gt == 3, fused == 3)
                    if not math.isnan(d_scar):
                        scar_dices.append(d_scar)
                    board_scars.append(BinarySurfaceMetrics.compute_board_dice(gt == 3, fused == 3))

                mean_scar = float(np.mean(scar_dices)) if scar_dices else 0.0
                mean_board_scar = float(np.mean(board_scars)) if board_scars else 0.0
                val_score = 0.75 * mean_scar + 0.25 * mean_board_scar
                history.append(
                    {
                        "epoch": float(epoch),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train_loss": avg_loss,
                        "val_dice_scar": mean_scar,
                        "val_board_scar": mean_board_scar,
                        "val_score": val_score,
                    }
                )
                if val_score > best_score:
                    best_score = val_score
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_val_score": best_score,
                            "config": self.config.to_serializable_dict(),
                            "model_state_dict": model.state_dict(),
                        },
                        str(ckpt_path),
                    )
                LOGGER.info(
                    "Stage2 Ep %03d/%03d | loss=%.4f | val_scar=%.4f | board_scar=%.4f | score=%.4f%s",
                    epoch,
                    total_epochs,
                    avg_loss,
                    mean_scar,
                    mean_board_scar,
                    val_score,
                    " [BEST]" if val_score >= best_score else "",
                )

        if history:
            hist_csv = self.output_dir / "stage2_training_history.csv"
            with hist_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
                writer.writeheader()
                for row in history:
                    writer.writerow({k: f"{v:.6f}" for k, v in row.items()})

            fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
            eps = [int(r["epoch"]) for r in history]
            axes[0].plot(eps, [r["train_loss"] for r in history], color="#2563eb")
            axes[0].set_title("Perda Etapa 2 (CMR-MULTI + EMIDEC)")
            axes[0].set_xlabel("Época")
            axes[0].grid(linestyle="--", alpha=0.4)
            axes[1].plot(eps, [r["val_dice_scar"] for r in history], label="Dice Scar (GT+)", color="#dc2626")
            axes[1].plot(eps, [r["val_board_scar"] for r in history], label="Board Scar", color="#0f172a")
            axes[1].set_ylim(0.0, 1.02)
            axes[1].set_title("Validação End-to-End (Stage 1 -> Stage 2)")
            axes[1].legend(frameon=False)
            axes[1].grid(linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(str(self.output_dir / "stage2_training_curves.png"), dpi=150)
            plt.close(fig)

        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            LOGGER.info("Melhor Etapa 2 restaurada (época %d, score=%.4f).", ckpt["epoch"], ckpt["best_val_score"])

        # Calibração do limiar de fibrose na validação (com TTA)
        best_thr = 0.50
        if not dry_run:
            cached_val_probs = {
                rec.patient_id: self._predict_stage2_probs_native(
                    model, device, val_roi_packs[rec.patient_id], use_tta=True, mock_mode=False
                )
                for rec in val_records
            }
            best_thr_score = -1.0
            for thr in self.config.threshold_grid:
                d_list: List[float] = []
                b_list: List[float] = []
                for rec in val_records:
                    fused, _ = self.fuse_stage1_and_stage2(
                        val_stage1_preds[rec.patient_id], cached_val_probs[rec.patient_id], scar_threshold=thr
                    )
                    gt = val_gt_5c[rec.patient_id]
                    ds = BinarySurfaceMetrics.compute_dice(gt == 3, fused == 3)
                    if not math.isnan(ds):
                        d_list.append(ds)
                    b_list.append(BinarySurfaceMetrics.compute_board_dice(gt == 3, fused == 3))
                score_t = 0.70 * (float(np.mean(d_list)) if d_list else 0.0) + 0.30 * (float(np.mean(b_list)) if b_list else 0.0)
                LOGGER.info("Calibração limiar t=%.2f -> scar=%.4f | board=%.4f | score=%.4f", thr, float(np.mean(d_list)) if d_list else 0.0, float(np.mean(b_list)) if b_list else 0.0, score_t)
                if score_t > best_thr_score:
                    best_thr_score = score_t
                    best_thr = float(thr)
            LOGGER.info("Limiar ótimo de fibrose calibrado na validação: t=%.2f", best_thr)

        return model, best_thr

    def _evaluate_cohort_end_to_end(
        self,
        cohort_key: str,
        cohort_display: str,
        labeled_records: Sequence[VolumeRecord],
        unlabeled_records: Sequence[VolumeRecord],
        stage1_model: Optional[object],
        stage2_model: Optional[object],
        device: Optional[object],
        scar_threshold: float,
        mock_mode: bool = False,
    ) -> Dict[str, object]:
        """Executa inferência end-to-end (Etapa 1 -> Etapa 2), avaliação MICCAI 2026 e exportação NIfTI."""
        cohort_out = self.output_dir / cohort_key
        images_dir = cohort_out / "_tmp_eval" / "images"
        gt_dir = cohort_out / "_tmp_eval" / "gt"
        pred_dir = cohort_out / "_tmp_eval" / "pred"
        nifti_bundle = cohort_out / f"niftis_{cohort_key}_cascade_5c"
        for d in (images_dir, gt_dir, pred_dir, nifti_bundle):
            d.mkdir(parents=True, exist_ok=True)

        model_label = "Cascata 2-Etapas (CMR+EMIDEC)"

        for rec in labeled_records:
            s1_4c = self._predict_stage1_4c(stage1_model, device, rec, mock_mode=mock_mode)
            gt_5c = LabelStandardizer.standardize_record_mask(rec, task_mode=TaskMode.FULL_5C)
            assert gt_5c is not None
            pack = self.roi_extractor.build_roi_pack(
                record=rec,
                lv_prior_native=(s1_4c == 1).astype(np.int16),
                myo_prior_native=(s1_4c == 2).astype(np.int16),
                gt_stage2_native=gt_5c if mock_mode else None,
            )
            s2_probs = self._predict_stage2_probs_native(
                stage2_model, device, pack, use_tta=True, mock_mode=mock_mode
            )
            pred_5c, scar_prob_map = self.fuse_stage1_and_stage2(s1_4c, s2_probs, scar_threshold=scar_threshold)

            img_native, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
            img_out = images_dir / f"{rec.patient_id}.nii.gz"
            gt_out = gt_dir / f"{rec.patient_id}.nii.gz"
            pred_out = pred_dir / f"{rec.patient_id}.nii.gz"

            out_img = nib.Nifti1Image(img_native.astype(np.float32), img_nii.affine)
            out_img.header["pixdim"] = img_nii.header["pixdim"]
            out_img.set_data_dtype(np.float32)
            nib.save(out_img, str(img_out))

            NiftiVolumeIO.save_mask_like(gt_5c, img_nii.affine, img_nii.header["pixdim"], gt_out, dtype=np.int16)
            NiftiVolumeIO.save_mask_like(pred_5c, img_nii.affine, img_nii.header["pixdim"], pred_out, dtype=np.int16)

            p_dir = nifti_bundle / rec.patient_id
            p_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_out, p_dir / f"{rec.patient_id}_LGE.nii.gz")
            shutil.copy2(gt_out, p_dir / f"{rec.patient_id}_GT_5c.nii.gz")
            shutil.copy2(pred_out, p_dir / f"{rec.patient_id}_PRED_cascade_5c.nii.gz")

            # Salva mapa contínuo de probabilidade de fibrose (float32) para análise de border zone
            prob_nii = nib.Nifti1Image(scar_prob_map.astype(np.float32), img_nii.affine)
            prob_nii.header["pixdim"] = img_nii.header["pixdim"]
            prob_nii.set_data_dtype(np.float32)
            nib.save(prob_nii, str(p_dir / f"{rec.patient_id}_SCAR_PROB.nii.gz"))

        evaluator = CardiacMetrics3DEvaluator(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=cohort_out,
            task_mode=TaskMode.FULL_5C,
            cohort_name=cohort_display,
        )
        summary = evaluator.run()

        visualizer = SegmentationPanelVisualizer(
            image_dir=images_dir,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=cohort_out / "figures",
            task_mode=TaskMode.FULL_5C,
            model_label=model_label,
        )
        visualizer.run()

        for rec in unlabeled_records:
            s1_4c = self._predict_stage1_4c(stage1_model, device, rec, mock_mode=mock_mode)
            pack = self.roi_extractor.build_roi_pack(
                record=rec,
                lv_prior_native=(s1_4c == 1).astype(np.int16),
                myo_prior_native=(s1_4c == 2).astype(np.int16),
                gt_stage2_native=None,
            )
            s2_probs = self._predict_stage2_probs_native(stage2_model, device, pack, use_tta=True, mock_mode=mock_mode)
            pred_5c, scar_prob_map = self.fuse_stage1_and_stage2(s1_4c, s2_probs, scar_threshold=scar_threshold)

            img_native, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
            p_dir = nifti_bundle / rec.patient_id
            p_dir.mkdir(parents=True, exist_ok=True)
            img_p = p_dir / f"{rec.patient_id}_LGE.nii.gz"
            pred_p = p_dir / f"{rec.patient_id}_PRED_cascade_5c.nii.gz"

            out_img = nib.Nifti1Image(img_native.astype(np.float32), img_nii.affine)
            out_img.header["pixdim"] = img_nii.header["pixdim"]
            out_img.set_data_dtype(np.float32)
            nib.save(out_img, str(img_p))
            NiftiVolumeIO.save_mask_like(pred_5c, img_nii.affine, img_nii.header["pixdim"], pred_p, dtype=np.int16)

            prob_nii = nib.Nifti1Image(scar_prob_map.astype(np.float32), img_nii.affine)
            prob_nii.header["pixdim"] = img_nii.header["pixdim"]
            prob_nii.set_data_dtype(np.float32)
            nib.save(prob_nii, str(p_dir / f"{rec.patient_id}_SCAR_PROB.nii.gz"))
            visualizer.render_unlabeled_case(rec.patient_id, img_p, pred_p)

        zip_base = cohort_out / f"niftis_predicoes_{cohort_key}_cascade_5c"
        shutil.make_archive(str(zip_base), "zip", str(nifti_bundle))
        shutil.rmtree(nifti_bundle, ignore_errors=True)
        shutil.rmtree(cohort_out / "_tmp_eval", ignore_errors=True)
        return summary

    def run(self, dry_run: bool = False) -> Dict[str, object]:
        """Executa o fluxo completo da Cascata em 2 Etapas (CMR-MULTI + EMIDEC -> HU-UFJF)."""
        self._seed_everything(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmr_loader = CmrMultiCohortLoader(self.cmr_dir)
        all_cmr = cmr_loader.discover_records()
        scar_map = cmr_loader.build_scar_presence_map(all_cmr)
        split_gen = StratifiedSplitGenerator(BenchmarkSplitConfig(seed=self.config.seed))
        splits = split_gen.split_train_val_test(scar_map, dry_run=dry_run)
        split_gen.save_splits_json(splits, self.output_dir / "splits_used.json")

        rec_by_id = {r.patient_id: r for r in all_cmr}
        train_cmr = [rec_by_id[cid] for cid in splits["train"]]
        val_cmr = [rec_by_id[cid] for cid in splits["val"]]
        test_cmr = [rec_by_id[cid] for cid in splits["test"]]

        emidec_records: List[VolumeRecord] = []
        if self.emidec_dir is not None and self.emidec_dir.exists():
            emidec_records = EmidecCohortLoader(self.emidec_dir).discover_train_records()

        train_packs = self._build_stage2_training_packs(train_cmr, emidec_records)

        mock_mode = False
        device = None
        stage1_model = None
        stage2_model = None
        best_thr = 0.50

        try:
            import segmentation_models_pytorch  # noqa: F401
            import torch
        except ImportError as exc:
            if not dry_run:
                raise ImportError("PyTorch e segmentation_models_pytorch são obrigatórios.") from exc
            LOGGER.warning("PyTorch/SMP ausentes em CPU local (%s); executando validação dry-run estrutural.", exc)
            ds = Stage2RoiSliceDataset(train_packs, augmentor=Stage2RoiAugmentor(), seed=self.config.seed)
            img0, myo0, lv0, tgt0 = ds.samples[0]
            ai, am, al, at = Stage2RoiAugmentor()(img0, myo0, lv0, tgt0, np.random.default_rng(self.config.seed))
            LOGGER.info(
                "Sanity check Stage2 dry-run OK: %d fatias ROI | shape=%s | alvos únicos=%s",
                len(ds),
                ai.shape,
                np.unique(at).tolist(),
            )
            mock_mode = True

        if not mock_mode:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            stage1_model = self._load_or_train_stage1(device=device, dry_run=dry_run)

            # Gera ROIs realistas da validação usando predições da Etapa 1
            val_stage1_preds: Dict[str, np.ndarray] = {}
            val_roi_packs: Dict[str, RoiVolumePack] = {}
            val_gt_5c: Dict[str, np.ndarray] = {}
            for rec in val_cmr:
                s1 = self._predict_stage1_4c(stage1_model, device, rec, mock_mode=False)
                val_stage1_preds[rec.patient_id] = s1
                val_roi_packs[rec.patient_id] = self.roi_extractor.build_roi_pack(
                    record=rec,
                    lv_prior_native=(s1 == 1).astype(np.int16),
                    myo_prior_native=(s1 == 2).astype(np.int16),
                    gt_stage2_native=None,
                )
                gt = LabelStandardizer.standardize_record_mask(rec, task_mode=TaskMode.FULL_5C)
                assert gt is not None
                val_gt_5c[rec.patient_id] = gt

            stage2_model, best_thr = self._train_stage2(
                train_packs=train_packs,
                val_records=val_cmr,
                val_stage1_preds=val_stage1_preds,
                val_roi_packs=val_roi_packs,
                val_gt_5c=val_gt_5c,
                device=device,
                dry_run=dry_run,
            )

        cfg_dump = self.config.to_serializable_dict()
        cfg_dump["calibrated_scar_threshold"] = best_thr
        cfg_dump["num_stage2_train_cmr"] = len(train_cmr)
        cfg_dump["num_stage2_train_emidec"] = len(emidec_records)
        cfg_dump["stage1_checkpoint_used"] = str(self.stage1_checkpoint) if self.stage1_checkpoint else "trained_from_scratch"
        (self.output_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2, ensure_ascii=False), encoding="utf-8")

        report: Dict[str, object] = {
            "experiment": "exp06_cascade_scar_cmr_emidec",
            "calibrated_scar_threshold": best_thr,
            "num_stage2_train_volumes": len(train_packs),
            "dry_run": dry_run,
        }
        report["cmr_multi_test"] = self._evaluate_cohort_end_to_end(
            cohort_key="cmr_multi_test",
            cohort_display="CMR-MULTI Test (Cascade)",
            labeled_records=test_cmr,
            unlabeled_records=[],
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            device=device,
            scar_threshold=best_thr,
            mock_mode=mock_mode,
        )

        if self.hu_dir is not None and self.hu_dir.exists():
            hu_labeled, hu_unlabeled = HuUfjfCohortLoader(self.hu_dir).discover_records()
            if hu_labeled:
                report["hu_ufjf_external"] = self._evaluate_cohort_end_to_end(
                    cohort_key="hu_ufjf_external",
                    cohort_display="HU-UFJF Private Cohort (Cascade)",
                    labeled_records=hu_labeled,
                    unlabeled_records=hu_unlabeled,
                    stage1_model=stage1_model,
                    stage2_model=stage2_model,
                    device=device,
                    scar_threshold=best_thr,
                    mock_mode=mock_mode,
                )

        (self.output_dir / "experiment_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("Pipeline Cascata 2-Etapas finalizado em %s", self.output_dir)
        return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(
        description="Treina e avalia Cascata em 2 Etapas (Anatomia 4c -> Especialista Fibrose CMR-MULTI + EMIDEC)."
    )
    parser.add_argument("--cmr-dir", type=Path, required=True, help="Diretório do dataset CMR-MULTI.")
    parser.add_argument("--emidec-dir", type=Path, default=None, help="Diretório opcional do dataset EMIDEC (100 casos).")
    parser.add_argument("--hu-dir", type=Path, default=None, help="Diretório opcional da coorte privada HU-UFJF.")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None, help="Checkpoint pré-treinado da Etapa 1 (Exp 05A 4c).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório de exportação.")
    parser.add_argument("--epochs", type=int, default=80, help="Épocas de treino da Etapa 2.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do mini-batch da Etapa 2.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Taxa de aprendizado inicial (AdamW).")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória oficial (SEED=42).")
    parser.add_argument("--dry-run", action="store_true", help="Validação rápida em CPU sobre data_sample/.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI da Cascata em 2 Etapas."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    config = CascadeScarConfig(
        stage2_epochs=args.epochs,
        stage2_batch_size=args.batch_size,
        stage2_lr=args.lr,
        seed=args.seed,
    )
    pipeline = CascadeScarPipeline(
        config=config,
        cmr_dir=args.cmr_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        emidec_dir=args.emidec_dir.resolve() if args.emidec_dir is not None else None,
        hu_dir=args.hu_dir.resolve() if args.hu_dir is not None else None,
        stage1_checkpoint=args.stage1_checkpoint.resolve() if args.stage1_checkpoint is not None else None,
    )
    pipeline.run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
