"""Adaptação de Domínio em CV 5-Fold estratificado na coorte clínica HU-UFJF (Exp 07).

Compara rigorosamente, sob exatamente os mesmos 5 folds estratificados por volume
de fibrose (`N=10` pacientes HU-UFJF avaliados *out-of-fold*, zero *data leakage*):
  - **Exp 07A (`finetune_hu_only`):** *Fine-tuning* puro a partir dos pesos do
    Exp 05B (`U-Net ResNet34 5c`) exclusivamente nos 8 pacientes HU-UFJF de treino
    de cada fold, com LR diferencial (`encoder=3e-5`, `decoder=1.5e-4`) e recorte
    de patches `256×256` centrados no coração com *jitter* espacial (±36 px).
  - **Exp 07B (`co_finetune_hu_cmr`):** *Co-fine-tuning* regularizado combinando
    os 8 pacientes HU-UFJF (*oversampling* 4×) + os 35 pacientes de treino do
    CMR-MULTI (`SEED=42`), além de avaliar retenção no CMR-MULTI Test (`N=7`)
    para quantificar *catastrophic forgetting*.
  - Predição por *ensemble* probabilístico dos 5 folds para `patient_20` (sem GT).
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

from src.data.cardiac_dataset import (
    CardiacAugmentor,
    CmrMultiCohortLoader,
    HuUfjfCohortLoader,
    NiftiVolumeIO,
    PreparedVolume,
    TaskMode,
    VolumePreprocessor,
    VolumeRecord,
)
from src.data.splits import BenchmarkSplitConfig, StratifiedSplitGenerator
from src.evaluation.metrics_3d import CardiacMetrics3DEvaluator
from src.evaluation.visualize import SegmentationPanelVisualizer
from src.models.unet_resnet34 import (
    CardiacPostProcessor,
    CombinedSegmentationLoss,
    SlidingWindow2DInferer,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HuDomainAdaptationConfig:
    """Hiperparâmetros imutáveis do Exp 07 (Adaptação de Domínio HU-UFJF 5-Fold)."""

    num_folds: int = 5
    target_spacing_mm: float = 1.0
    patch_size: int = 256
    heart_crop_jitter_px: int = 36
    finetune_epochs: int = 60
    co_finetune_epochs: int = 45
    warmup_epochs: int = 5
    batch_size: int = 16
    encoder_lr: float = 3e-5
    decoder_lr: float = 1.5e-4
    min_lr_ratio: float = 0.02
    weight_decay: float = 2e-4
    ce_weight: float = 0.45
    dice_weight: float = 0.55
    class_weights: Tuple[float, ...] = (0.5, 1.5, 2.0, 5.0, 1.5)
    hu_co_oversample: int = 4
    hu_scar_slice_oversample: int = 2
    tta_angles_deg: Tuple[float, ...] = (0.0, -8.0, 8.0)
    seed: int = 42
    num_workers: int = 2

    def to_serializable_dict(self) -> Dict[str, object]:
        raw = asdict(self)
        raw["class_weights"] = list(self.class_weights)
        raw["tta_angles_deg"] = list(self.tta_angles_deg)
        return raw


class HuStratifiedKFoldSplitter:
    """Divide os pacientes rotulados do HU-UFJF em K folds balanceados por volume de fibrose."""

    def __init__(self, num_folds: int = 5) -> None:
        self.num_folds = int(num_folds)

    def build_folds(self, hu_volumes: Sequence[PreparedVolume], dry_run: bool = False) -> List[Dict[str, List[str]]]:
        """Ordena por volume de fibrose (mL) e distribui em serpentina (snake-style)."""
        if len(hu_volumes) < 2:
            raise ValueError(f"São necessários ao menos 2 pacientes HU-UFJF; encontrados {len(hu_volumes)}.")

        k = min(2, len(hu_volumes)) if dry_run else min(self.num_folds, len(hu_volumes))
        scored: List[Tuple[float, str]] = []
        for vol in hu_volumes:
            assert vol.mask_native is not None
            voxel_ml = (
                vol.native_spacing_mm[0] * vol.native_spacing_mm[1] * vol.native_spacing_mm[2]
            ) / 1000.0
            scar_ml = float(np.count_nonzero(vol.mask_native == 3)) * voxel_ml
            scored.append((scar_ml, vol.patient_id))

        # Ordena decrescente por volume de fibrose e atribui em serpentina para balancear carga
        scored.sort(key=lambda item: (-item[0], item[1]))
        buckets: List[List[str]] = [[] for _ in range(k)]
        for idx, (_, pid) in enumerate(scored):
            cycle = idx // k
            pos = idx % k
            bucket_idx = pos if (cycle % 2 == 0) else (k - 1 - pos)
            buckets[bucket_idx].append(pid)

        all_ids = [pid for _, pid in scored]
        folds: List[Dict[str, List[str]]] = []
        for bucket in buckets:
            test_ids = sorted(bucket)
            test_set = set(test_ids)
            train_ids = sorted(pid for pid in all_ids if pid not in test_set)
            folds.append({"train": train_ids, "test_oof": test_ids})

        LOGGER.info("Gerados %d folds estratificados por volume de fibrose no HU-UFJF: %s", len(folds), folds)
        return folds


class HeartCenteredPatchDataset:
    """Dataset 2D com recorte `256×256` centrado no coração + *jitter* espacial (ideal para HU-UFJF)."""

    def __init__(
        self,
        volumes: Sequence[PreparedVolume],
        patch_size: int = 256,
        jitter_px: int = 36,
        augmentor: Optional[CardiacAugmentor] = None,
        seed: int = 42,
        repeat_factor: int = 1,
        scar_slice_oversample: int = 1,
    ) -> None:
        self.patch_size = int(patch_size)
        self.jitter_px = int(jitter_px)
        self.augmentor = augmentor
        self.rng = np.random.default_rng(seed)
        self.samples: List[Tuple[np.ndarray, np.ndarray, int, int, bool]] = []

        for vol in volumes:
            if vol.mask_resampled is None:
                continue
            fg_3d = vol.mask_resampled > 0
            if np.any(fg_3d):
                ys, xs, _ = np.where(fg_3d)
                vol_cy = int(round(0.5 * (int(ys.min()) + int(ys.max()))))
                vol_cx = int(round(0.5 * (int(xs.min()) + int(xs.max()))))
            else:
                vol_cy = vol.image_resampled.shape[0] // 2
                vol_cx = vol.image_resampled.shape[1] // 2

            for z in range(vol.image_resampled.shape[2]):
                img_slc = vol.image_resampled[:, :, z]
                msk_slc = vol.mask_resampled[:, :, z]
                has_fg = bool(np.any(msk_slc > 0))
                has_scar = bool(np.any(msk_slc == 3))
                entry = (img_slc, msk_slc, vol_cy, vol_cx, has_fg)

                copies = max(1, repeat_factor)
                if has_scar and scar_slice_oversample > 1:
                    copies *= scar_slice_oversample
                for _ in range(copies):
                    self.samples.append(entry)

        if not self.samples:
            raise ValueError("Nenhuma fatia válida encontrada para HeartCenteredPatchDataset.")

    @staticmethod
    def _crop_or_pad_centered(
        arr: np.ndarray,
        cy: int,
        cx: int,
        size: int,
        pad_value: float,
    ) -> np.ndarray:
        """Extrai recorte `(size, size)` centrado em `(cy, cx)` preenchendo bordas se necessário."""
        h, w = arr.shape
        half = size // 2
        y0 = cy - half
        x0 = cx - half
        y1 = y0 + size
        x1 = x0 + size

        out = np.full((size, size), pad_value, dtype=arr.dtype)
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(h, y1), min(w, x1)
        dy0, dx0 = sy0 - y0, sx0 - x0
        dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
        if sy1 > sy0 and sx1 > sx0:
            out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[object, object]:
        import torch

        img_slc, msk_slc, cy, cx, has_fg = self.samples[index]
        h, w = img_slc.shape
        if self.augmentor is not None and (h > self.patch_size or w > self.patch_size):
            if has_fg:
                cy_j = cy + int(self.rng.integers(-self.jitter_px, self.jitter_px + 1))
                cx_j = cx + int(self.rng.integers(-self.jitter_px, self.jitter_px + 1))
            else:
                cy_j = int(self.rng.integers(self.patch_size // 2, max(self.patch_size // 2 + 1, h - self.patch_size // 2)))
                cx_j = int(self.rng.integers(self.patch_size // 2, max(self.patch_size // 2 + 1, w - self.patch_size // 2)))
        else:
            cy_j, cx_j = cy, cx

        img_patch = self._crop_or_pad_centered(img_slc, cy_j, cx_j, self.patch_size, pad_value=0.0).astype(np.float32)
        msk_patch = self._crop_or_pad_centered(msk_slc, cy_j, cx_j, self.patch_size, pad_value=0).astype(np.int16)

        if self.augmentor is not None:
            img_patch, msk_patch = self.augmentor(img_patch, msk_patch, self.rng)

        return (
            torch.from_numpy(np.ascontiguousarray(img_patch[None, :, :])).float(),
            torch.from_numpy(np.ascontiguousarray(msk_patch)).long(),
        )


class HuDomainAdaptationPipeline:
    """Executa Exp 07A (Fine-Tuning Puro) e Exp 07B (Co-Fine-Tuning HU+CMR) em CV 5-Fold."""

    def __init__(
        self,
        config: HuDomainAdaptationConfig,
        cmr_dir: Path,
        hu_dir: Path,
        output_dir: Path,
        pretrained_checkpoint: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.cmr_dir = cmr_dir
        self.hu_dir = hu_dir
        self.output_dir = output_dir
        self.pretrained_checkpoint = pretrained_checkpoint

        self.preprocessor = VolumePreprocessor(target_spacing_mm=config.target_spacing_mm)
        self.post_processor = CardiacPostProcessor(task_mode=TaskMode.FULL_5C, min_isolated_scar_voxels=3)
        self.inferer = SlidingWindow2DInferer(
            num_classes=5,
            patch_size=config.patch_size,
            overlap=0.5,
            tta_angles_deg=config.tta_angles_deg,
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

    def _build_warmstarted_model(self, device: object) -> object:
        """Instancia U-Net ResNet34 (5c) e carrega pesos pré-treinados do Exp 05B."""
        import segmentation_models_pytorch as smp
        import torch

        has_ckpt = self.pretrained_checkpoint is not None and self.pretrained_checkpoint.is_file()
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None if has_ckpt else "imagenet",
            in_channels=1,
            classes=5,
            decoder_use_batchnorm=True,
        ).to(device)
        if has_ckpt:
            assert self.pretrained_checkpoint is not None
            ckpt = torch.load(str(self.pretrained_checkpoint), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        return model

    def _train_single_fold(
        self,
        train_hu_vols: Sequence[PreparedVolume],
        cmr_train_vols: Sequence[PreparedVolume],
        co_train: bool,
        fold_idx: int,
        device: object,
        dry_run: bool = False,
    ) -> Tuple[object, List[Dict[str, float]]]:
        """Treina 1 fold com LR diferencial e agenda cosseno sem vazamento para o fold de teste."""
        import torch
        from torch.utils.data import ConcatDataset, DataLoader

        self._seed_everything(self.config.seed + fold_idx * 101)
        aug = CardiacAugmentor(multires_blur_prob=0.25)

        hu_ds = HeartCenteredPatchDataset(
            volumes=train_hu_vols,
            patch_size=self.config.patch_size,
            jitter_px=self.config.heart_crop_jitter_px,
            augmentor=aug,
            seed=self.config.seed + fold_idx,
            repeat_factor=self.config.hu_co_oversample if co_train else 1,
            scar_slice_oversample=self.config.hu_scar_slice_oversample,
        )

        if co_train and cmr_train_vols:
            cmr_ds = HeartCenteredPatchDataset(
                volumes=cmr_train_vols,
                patch_size=self.config.patch_size,
                jitter_px=20,
                augmentor=CardiacAugmentor(multires_blur_prob=0.40),
                seed=self.config.seed + 500 + fold_idx,
                repeat_factor=1,
                scar_slice_oversample=1,
            )
            train_ds: object = ConcatDataset([hu_ds, cmr_ds])
            total_epochs = 1 if dry_run else self.config.co_finetune_epochs
        else:
            train_ds = hu_ds
            total_epochs = 1 if dry_run else self.config.finetune_epochs

        batch_size = min(self.config.batch_size, len(train_ds))  # type: ignore[arg-type]
        loader = DataLoader(
            train_ds,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if dry_run else self.config.num_workers,
            pin_memory=(str(device).startswith("cuda")),
            drop_last=False,
        )

        model = self._build_warmstarted_model(device)
        criterion = CombinedSegmentationLoss(
            class_weights=self.config.class_weights,
            ce_weight=self.config.ce_weight,
            dice_weight=self.config.dice_weight,
        ).to(device)

        # Taxa de aprendizado diferencial (encoder mais suave que decoder/cabeça)
        encoder_params = list(model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.config.encoder_lr},
                {"params": decoder_params, "lr": self.config.decoder_lr},
            ],
            weight_decay=self.config.weight_decay,
        )

        warmup = min(self.config.warmup_epochs, max(1, total_epochs // 5))

        def lr_lambda(ep_idx: int) -> float:
            if ep_idx < warmup:
                return float(ep_idx + 1) / float(warmup)
            prog = float(ep_idx - warmup) / float(max(1, total_epochs - warmup))
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

        history: List[Dict[str, float]] = []
        for epoch in range(1, total_epochs + 1):
            model.train()
            loss_sum = 0.0
            steps = 0
            for imgs, masks in loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.item())
                steps += 1
                if dry_run:
                    break
            scheduler.step()
            avg_loss = loss_sum / max(1, steps)
            history.append({"fold": float(fold_idx), "epoch": float(epoch), "train_loss": avg_loss})
            if epoch == 1 or epoch % 15 == 0 or epoch == total_epochs:
                LOGGER.info(
                    "[%s | Fold %d] Ep %02d/%02d | loss=%.4f | lr_dec=%.2e",
                    "Co-FT" if co_train else "FT-HU",
                    fold_idx,
                    epoch,
                    total_epochs,
                    avg_loss,
                    float(optimizer.param_groups[1]["lr"]),
                )

        model.eval()
        return model, history

    def _predict_probs_native(self, model: object, device: object, prepared: PreparedVolume) -> np.ndarray:
        """Retorna mapa 4D de probabilidades `(5, H_nat, W_nat, Z)` usando janela deslizante + TTA."""
        from scipy.ndimage import rotate as nd_rotate

        h_res, w_res, z_res = prepared.image_resampled.shape
        probs_res = np.zeros((5, h_res, w_res, z_res), dtype=np.float32)
        angles = self.config.tta_angles_deg

        for z in range(z_res):
            slc = prepared.image_resampled[:, :, z]
            slc_sum = np.zeros((5, h_res, w_res), dtype=np.float32)
            for angle in angles:
                if abs(angle) < 1e-3:
                    p_ang = self.inferer._predict_single_slice_probs(model, device, slc)
                else:
                    rot_in = nd_rotate(slc, angle=angle, reshape=False, order=1, mode="constant", cval=0.0)
                    p_rot = self.inferer._predict_single_slice_probs(model, device, rot_in)
                    p_ang = np.stack(
                        [nd_rotate(p_rot[c], angle=-angle, reshape=False, order=1, mode="constant", cval=0.0) for c in range(5)],
                        axis=0,
                    )
                slc_sum += p_ang
            probs_res[:, :, :, z] = slc_sum / float(len(angles))

        return VolumePreprocessor.resample_probs_to_native(probs_res, prepared.native_shape)

    def _run_cv_strategy(
        self,
        strategy_key: str,
        strategy_label: str,
        co_train: bool,
        folds: Sequence[Dict[str, List[str]]],
        hu_rec_by_id: Dict[str, VolumeRecord],
        hu_vol_by_id: Dict[str, PreparedVolume],
        hu_unlabeled_recs: Sequence[VolumeRecord],
        cmr_train_vols: Sequence[PreparedVolume],
        cmr_test_recs: Sequence[VolumeRecord],
        device: Optional[object],
        mock_mode: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, object]:
        """Executa CV 5-fold completo para uma estratégia, avalia OOF HU-UFJF e retenção CMR-MULTI."""
        strat_out = self.output_dir / strategy_key
        hu_out = strat_out / "hu_ufjf_oof"
        img_dir = hu_out / "_tmp" / "images"
        gt_dir = hu_out / "_tmp" / "gt"
        pred_dir = hu_out / "_tmp" / "pred"
        bundle_dir = hu_out / f"niftis_hu_ufjf_{strategy_key}"
        for d in (img_dir, gt_dir, pred_dir, bundle_dir):
            d.mkdir(parents=True, exist_ok=True)

        all_history: List[Dict[str, float]] = []
        # Acumuladores de ensemble para patient_20 (e retenção CMR-MULTI em co_train)
        unlabeled_prob_sum: Dict[str, np.ndarray] = {}
        unlabeled_prepared: Dict[str, PreparedVolume] = {
            r.patient_id: self.preprocessor.prepare_volume(r, TaskMode.FULL_5C) for r in hu_unlabeled_recs
        }
        cmr_test_prob_sum: Dict[str, np.ndarray] = {}
        cmr_test_prepared: Dict[str, PreparedVolume] = (
            {r.patient_id: self.preprocessor.prepare_volume(r, TaskMode.FULL_5C) for r in cmr_test_recs}
            if co_train
            else {}
        )

        for fold_idx, fold_spec in enumerate(folds):
            train_vols = [hu_vol_by_id[pid] for pid in fold_spec["train"]]
            test_ids = fold_spec["test_oof"]

            model = None
            if not mock_mode:
                assert device is not None
                model, fold_hist = self._train_single_fold(
                    train_hu_vols=train_vols,
                    cmr_train_vols=cmr_train_vols,
                    co_train=co_train,
                    fold_idx=fold_idx,
                    device=device,
                    dry_run=dry_run,
                )
                all_history.extend(fold_hist)
                # Salva checkpoint do fold 0 como referência leve de reuso
                if fold_idx == 0:
                    import torch

                    torch.save(
                        {
                            "strategy": strategy_key,
                            "fold": 0,
                            "config": self.config.to_serializable_dict(),
                            "model_state_dict": model.state_dict(),
                        },
                        str(strat_out / "checkpoint_fold0.pth"),
                    )

            # Inferência OOF nos pacientes inéditos deste fold
            for pid in test_ids:
                vol = hu_vol_by_id[pid]
                rec = hu_rec_by_id[pid]
                assert vol.mask_native is not None
                if mock_mode:
                    pred_native = self.post_processor.apply(vol.mask_native.copy())
                else:
                    assert model is not None and device is not None
                    pred_native = self.inferer.predict_volume(
                        model=model, device=device, prepared=vol, post_processor=self.post_processor, use_tta=True
                    )

                img_native, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
                img_out = img_dir / f"{pid}.nii.gz"
                gt_out = gt_dir / f"{pid}.nii.gz"
                pred_out = pred_dir / f"{pid}.nii.gz"

                out_img = nib.Nifti1Image(img_native.astype(np.float32), img_nii.affine)
                out_img.header["pixdim"] = img_nii.header["pixdim"]
                out_img.set_data_dtype(np.float32)
                nib.save(out_img, str(img_out))
                NiftiVolumeIO.save_mask_like(vol.mask_native, vol.native_affine, vol.native_header_pixdim, gt_out, np.int16)
                NiftiVolumeIO.save_mask_like(pred_native, vol.native_affine, vol.native_header_pixdim, pred_out, np.int16)

                p_dir = bundle_dir / pid
                p_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_out, p_dir / f"{pid}_LGE.nii.gz")
                shutil.copy2(gt_out, p_dir / f"{pid}_GT_5c.nii.gz")
                shutil.copy2(pred_out, p_dir / f"{pid}_PRED_{strategy_key}.nii.gz")

            # Acumula probabilidades para ensemble do patient_20 (e CMR-MULTI test em Co-FT)
            if not mock_mode:
                assert model is not None and device is not None
                for pid, prep in unlabeled_prepared.items():
                    p = self._predict_probs_native(model, device, prep)
                    unlabeled_prob_sum[pid] = p if pid not in unlabeled_prob_sum else (unlabeled_prob_sum[pid] + p)
                for pid, prep in cmr_test_prepared.items():
                    p = self._predict_probs_native(model, device, prep)
                    cmr_test_prob_sum[pid] = p if pid not in cmr_test_prob_sum else (cmr_test_prob_sum[pid] + p)

        # Avalia OOF completo nos 10 pacientes HU-UFJF
        evaluator = CardiacMetrics3DEvaluator(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=hu_out,
            task_mode=TaskMode.FULL_5C,
            cohort_name=f"HU-UFJF OOF 5-Fold ({strategy_label})",
        )
        hu_summary = evaluator.run()

        visualizer = SegmentationPanelVisualizer(
            image_dir=img_dir,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=hu_out / "figures",
            task_mode=TaskMode.FULL_5C,
            model_label=strategy_label,
        )
        visualizer.run()

        # Finaliza ensemble para casos sem GT (patient_20)
        for rec in hu_unlabeled_recs:
            pid = rec.patient_id
            prep = unlabeled_prepared[pid]
            if mock_mode:
                pred_u = np.zeros(prep.native_shape, dtype=np.int16)
            else:
                avg_probs = unlabeled_prob_sum[pid] / float(len(folds))
                pred_u = self.post_processor.apply(np.argmax(avg_probs, axis=0).astype(np.int16))

            img_native, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
            p_dir = bundle_dir / pid
            p_dir.mkdir(parents=True, exist_ok=True)
            img_p = p_dir / f"{pid}_LGE.nii.gz"
            pred_p = p_dir / f"{pid}_PRED_{strategy_key}_ensemble5f.nii.gz"
            out_img = nib.Nifti1Image(img_native.astype(np.float32), img_nii.affine)
            out_img.header["pixdim"] = img_nii.header["pixdim"]
            out_img.set_data_dtype(np.float32)
            nib.save(out_img, str(img_p))
            NiftiVolumeIO.save_mask_like(pred_u, prep.native_affine, prep.native_header_pixdim, pred_p, np.int16)
            visualizer.render_unlabeled_case(pid, img_p, pred_p)

        shutil.make_archive(str(hu_out / f"niftis_hu_ufjf_{strategy_key}"), "zip", str(bundle_dir))
        shutil.rmtree(bundle_dir, ignore_errors=True)
        shutil.rmtree(hu_out / "_tmp", ignore_errors=True)

        # Histórico de perda por fold
        if all_history:
            hist_csv = strat_out / "training_history.csv"
            with hist_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["fold", "epoch", "train_loss"])
                writer.writeheader()
                writer.writerows(all_history)

            fig, ax = plt.subplots(figsize=(7.5, 4.0))
            for f_idx in range(len(folds)):
                rows = [r for r in all_history if int(r["fold"]) == f_idx]
                ax.plot([int(r["epoch"]) for r in rows], [r["train_loss"] for r in rows], label=f"Fold {f_idx}")
            ax.set_title(f"Curva de Perda — {strategy_label}")
            ax.set_xlabel("Época")
            ax.set_ylabel("Loss")
            ax.legend(frameon=False, ncol=3)
            ax.grid(linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(str(strat_out / "training_curves.png"), dpi=150)
            plt.close(fig)

        result: Dict[str, object] = {"hu_ufjf_oof": hu_summary}

        # Em Co-Fine-Tuning, avalia retenção no CMR-MULTI Test (N=7) com o ensemble dos 5 folds
        if co_train and cmr_test_recs:
            cmr_out = strat_out / "cmr_multi_retention"
            c_img = cmr_out / "_tmp" / "images"
            c_gt = cmr_out / "_tmp" / "gt"
            c_pred = cmr_out / "_tmp" / "pred"
            for d in (c_img, c_gt, c_pred):
                d.mkdir(parents=True, exist_ok=True)
            for rec in cmr_test_recs:
                pid = rec.patient_id
                prep = cmr_test_prepared[pid]
                assert prep.mask_native is not None
                if mock_mode:
                    pred_c = self.post_processor.apply(prep.mask_native.copy())
                else:
                    avg_p = cmr_test_prob_sum[pid] / float(len(folds))
                    pred_c = self.post_processor.apply(np.argmax(avg_p, axis=0).astype(np.int16))
                img_n, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
                out_img = nib.Nifti1Image(img_n.astype(np.float32), img_nii.affine)
                out_img.header["pixdim"] = img_nii.header["pixdim"]
                out_img.set_data_dtype(np.float32)
                nib.save(out_img, str(c_img / f"{pid}.nii.gz"))
                NiftiVolumeIO.save_mask_like(prep.mask_native, prep.native_affine, prep.native_header_pixdim, c_gt / f"{pid}.nii.gz", np.int16)
                NiftiVolumeIO.save_mask_like(pred_c, prep.native_affine, prep.native_header_pixdim, c_pred / f"{pid}.nii.gz", np.int16)

            cmr_eval = CardiacMetrics3DEvaluator(
                gt_dir=c_gt,
                pred_dir=c_pred,
                output_dir=cmr_out,
                task_mode=TaskMode.FULL_5C,
                cohort_name=f"CMR-MULTI Test Retention ({strategy_label})",
            )
            result["cmr_multi_retention"] = cmr_eval.run()
            SegmentationPanelVisualizer(
                image_dir=c_img,
                gt_dir=c_gt,
                pred_dir=c_pred,
                output_dir=cmr_out / "figures",
                task_mode=TaskMode.FULL_5C,
                model_label=f"{strategy_label} (5F Ens.)",
            ).run()
            shutil.rmtree(cmr_out / "_tmp", ignore_errors=True)

        return result

    def run(self, dry_run: bool = False) -> Dict[str, object]:
        """Orquestra preparação das duas coortes, CV 5-fold (FT vs Co-FT) e relatório comparativo."""
        self._seed_everything(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Carrega HU-UFJF e gera folds estratificados por volume de fibrose
        hu_labeled, hu_unlabeled = HuUfjfCohortLoader(self.hu_dir).discover_records()
        hu_rec_by_id = {r.patient_id: r for r in hu_labeled}
        hu_vol_by_id = {
            r.patient_id: self.preprocessor.prepare_volume(r, TaskMode.FULL_5C) for r in hu_labeled
        }
        folds = HuStratifiedKFoldSplitter(self.config.num_folds).build_folds(
            list(hu_vol_by_id.values()), dry_run=dry_run
        )
        (self.output_dir / "hu_5fold_splits.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

        # 2. Carrega CMR-MULTI respeitando rigorosamente o split oficial SEED=42 (sem tocar no teste!)
        cmr_loader = CmrMultiCohortLoader(self.cmr_dir)
        all_cmr = cmr_loader.discover_records()
        cmr_splits = StratifiedSplitGenerator(BenchmarkSplitConfig(seed=self.config.seed)).split_train_val_test(
            cmr_loader.build_scar_presence_map(all_cmr), dry_run=dry_run
        )
        cmr_by_id = {r.patient_id: r for r in all_cmr}
        cmr_train_vols = [
            self.preprocessor.prepare_volume(cmr_by_id[cid], TaskMode.FULL_5C) for cid in cmr_splits["train"]
        ]
        cmr_test_recs = [cmr_by_id[cid] for cid in cmr_splits["test"]]

        cfg_dump = self.config.to_serializable_dict()
        cfg_dump["pretrained_checkpoint"] = str(self.pretrained_checkpoint) if self.pretrained_checkpoint else "imagenet"
        cfg_dump["num_hu_labeled"] = len(hu_labeled)
        cfg_dump["num_cmr_train_anchor"] = len(cmr_train_vols)
        (self.output_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2, ensure_ascii=False), encoding="utf-8")

        mock_mode = False
        device = None
        try:
            import segmentation_models_pytorch  # noqa: F401
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError as exc:
            if not dry_run:
                raise ImportError("PyTorch e segmentation_models_pytorch são obrigatórios.") from exc
            LOGGER.warning("PyTorch/SMP ausentes em CPU local (%s); executando dry-run estrutural.", exc)
            ds = HeartCenteredPatchDataset(list(hu_vol_by_id.values()), patch_size=self.config.patch_size)
            LOGGER.info("Sanity check HeartCenteredPatchDataset OK: %d amostras.", len(ds))
            mock_mode = True

        # 3. Exp 07A — Fine-Tuning Puro (HU-UFJF only)
        res_07a = self._run_cv_strategy(
            strategy_key="exp07a_finetune_hu_only",
            strategy_label="Exp 07A (Fine-Tune HU)",
            co_train=False,
            folds=folds,
            hu_rec_by_id=hu_rec_by_id,
            hu_vol_by_id=hu_vol_by_id,
            hu_unlabeled_recs=hu_unlabeled,
            cmr_train_vols=cmr_train_vols,
            cmr_test_recs=cmr_test_recs,
            device=device,
            mock_mode=mock_mode,
            dry_run=dry_run,
        )

        # 4. Exp 07B — Co-Fine-Tuning (HU-UFJF 4x + CMR-MULTI treino)
        res_07b = self._run_cv_strategy(
            strategy_key="exp07b_co_finetune_hu_cmr",
            strategy_label="Exp 07B (Co-FT HU+CMR)",
            co_train=True,
            folds=folds,
            hu_rec_by_id=hu_rec_by_id,
            hu_vol_by_id=hu_vol_by_id,
            hu_unlabeled_recs=hu_unlabeled,
            cmr_train_vols=cmr_train_vols,
            cmr_test_recs=cmr_test_recs,
            device=device,
            mock_mode=mock_mode,
            dry_run=dry_run,
        )

        report = {
            "experiment": "exp07_hu_ufjf_domain_adaptation",
            "dry_run": dry_run,
            "exp07a_finetune_hu_only": res_07a,
            "exp07b_co_finetune_hu_cmr": res_07b,
        }
        (self.output_dir / "comparison_summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("Exp 07 (Adaptação de Domínio HU-UFJF 5-Fold) concluído em %s", self.output_dir)
        return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(
        description="Executa CV 5-Fold na coorte HU-UFJF comparando Fine-Tuning Puro vs. Co-Fine-Tuning."
    )
    parser.add_argument("--cmr-dir", type=Path, required=True, help="Diretório do dataset CMR-MULTI.")
    parser.add_argument("--hu-dir", type=Path, required=True, help="Diretório da coorte privada HU-UFJF.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório de exportação.")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None, help="Checkpoint Exp 05B (U-Net 5c).")
    parser.add_argument("--finetune-epochs", type=int, default=60, help="Épocas por fold no Fine-Tuning Puro.")
    parser.add_argument("--co-finetune-epochs", type=int, default=45, help="Épocas por fold no Co-Fine-Tuning.")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamanho do mini-batch 2D.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória oficial (SEED=42).")
    parser.add_argument("--dry-run", action="store_true", help="Validação rápida em CPU.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI do Exp 07."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    config = HuDomainAdaptationConfig(
        finetune_epochs=args.finetune_epochs,
        co_finetune_epochs=args.co_finetune_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    pipeline = HuDomainAdaptationPipeline(
        config=config,
        cmr_dir=args.cmr_dir.resolve(),
        hu_dir=args.hu_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        pretrained_checkpoint=args.pretrained_checkpoint.resolve() if args.pretrained_checkpoint else None,
    )
    pipeline.run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
