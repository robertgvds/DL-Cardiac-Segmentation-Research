"""Pipeline POO completo da U-Net ResNet34 (PyTorch SMP) para LGE-CMR (4 e 5 classes).

Arquitetura e hiperparâmetros fundamentados na revisão 2024–2026 (Hoh et al. 2024
JCMR, Ba Mahel et al. 2025, Kollmann et al. 2024 e Top-5 CMR-MULTI MICCAI 2026):
  - Backbone `resnet34` pré-treinado em ImageNet com decoder U-Net + BatchNorm.
  - Reamostragem física isotrópica no plano XY (`1.0 mm × 1.0 mm`) + augmentation
    multi-resolução (desfoque gaussiano + escala + deformação elástica leve) para
    eliminar viés de resolução entre CMR-MULTI (`0.5–1.3 mm`) e HU-UFJF (`1.8–2.7 mm`).
  - Perda combinada `0.5 × Weighted Cross-Entropy + 0.5 × Multi-class Soft Dice`.
  - Otimização `AdamW(lr=3e-4, weight_decay=1e-4)` com warmup linear (5 épocas) +
    `CosineAnnealingLR` (120 épocas) e mixed precision (`AMP`) em GPU.
  - Inferência 2D em janela deslizante com ponderação gaussiana (`256×256`, 50%
    sobreposição) + TTA rotacional suave (`0°, -8°, +8°`) + remapeamento inverso
    para resolução nativa + pós-processamento 3D anatômico (maior componente
    conexo do coração + remoção de micro-ruído isolado de fibrose).
  - Exporta NIfTIs (`int16`, `affine`/`pixdim` intactos), métricas 3D clínicas +
    CMR-MULTI MICCAI 2026 e painéis qualitativos para CMR-MULTI e HU-UFJF.
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
from scipy.ndimage import binary_dilation, label as cc_label, rotate as nd_rotate

from src.data.cardiac_dataset import (
    CardiacAugmentor,
    CardiacSliceDataset,
    CmrMultiCohortLoader,
    HuUfjfCohortLoader,
    NiftiVolumeIO,
    PreparedVolume,
    TaskMode,
    VolumePreprocessor,
    VolumeRecord,
)
from src.data.splits import BenchmarkSplitConfig, StratifiedSplitGenerator
from src.evaluation.metrics_3d import BinarySurfaceMetrics, CardiacMetrics3DEvaluator
from src.evaluation.visualize import SegmentationPanelVisualizer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnetResNet34Config:
    """Hiperparâmetros imutáveis do experimento U-Net ResNet34."""

    task_mode: TaskMode = TaskMode.FULL_5C
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    target_spacing_mm: float = 1.0
    patch_size: int = 256
    epochs: int = 120
    warmup_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    tta_angles_deg: Tuple[float, ...] = (0.0, -8.0, 8.0)
    sliding_window_overlap: float = 0.5
    min_isolated_scar_voxels: int = 3
    seed: int = 42
    num_workers: int = 2

    @property
    def class_weights(self) -> Tuple[float, ...]:
        """Pesos da Cross-Entropy ponderada por classe conforme o modo de tarefa."""
        if self.task_mode is TaskMode.ANATOMY_4C:
            # 0=Bg, 1=VE, 2=MioTotal, 3=VD
            return (0.5, 1.5, 2.5, 1.5)
        # 0=Bg, 1=VE, 2=MioSaudavel, 3=Scar (forte desbalanceamento), 4=VD
        return (0.5, 1.5, 2.0, 4.5, 1.5)

    def to_serializable_dict(self) -> Dict[str, object]:
        """Converte configuração em dicionário JSON-serializável."""
        raw = asdict(self)
        raw["task_mode"] = self.task_mode.value
        raw["num_classes"] = self.task_mode.num_classes
        raw["class_weights"] = list(self.class_weights)
        raw["tta_angles_deg"] = list(self.tta_angles_deg)
        return raw


class CardiacPostProcessor:
    """Pós-processamento anatômico 3D na resolução nativa após argmax."""

    def __init__(self, task_mode: TaskMode, min_isolated_scar_voxels: int = 3) -> None:
        self.task_mode = task_mode
        self.min_isolated_scar_voxels = int(min_isolated_scar_voxels)

    def apply(self, pred_mask_3d: np.ndarray) -> np.ndarray:
        """Remove falsos positivos torácicos distantes e micro-ruído isolado de cicatriz.

        1. Mantém exclusivamente o maior componente conexo 3D (conectividade-26)
           da união cardíaca (`pred > 0`), vital em imagens de tórax completo
           como as da coorte clínica HU-UFJF.
        2. Em 5 classes, remove micro-componentes de fibrose (`< min_isolated_scar_voxels`)
           desconectados do miocárdio/VE (inspirado em Hoh et al., 2024).
        """
        cleaned = np.rint(pred_mask_3d).astype(np.int16, copy=True)
        fg = cleaned > 0
        if not np.any(fg):
            return cleaned

        struct_26 = np.ones((3, 3, 3), dtype=np.int32)
        # Dilatação leve no plano XY (2 px -> ponte de até 4 px) apenas para agrupar VE+Mio+VD
        # através de eventuais lacunas finas do septo sem conectar artefatos torácicos distantes.
        xy_struct = np.zeros((3, 3, 3), dtype=bool)
        xy_struct[:, :, 1] = True
        fg_grouped = binary_dilation(fg, structure=xy_struct, iterations=2)
        labeled_fg, num_fg = cc_label(fg_grouped, structure=struct_26)
        if num_fg > 1:
            # Escolhe o grupo que contém o maior volume real de voxels cardíacos
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
                    if int(np.count_nonzero(comp)) < self.min_isolated_scar_voxels and not np.any(
                        comp & anchor_region
                    ):
                        cleaned[comp] = 0

        return cleaned


class SlidingWindow2DInferer:
    """Inferência volumétrica 3D fatia a fatia com janela deslizante gaussiana e TTA.

    Permite inferir tanto volumes pré-recortados (`CMR-MULTI`, `<= 256 px` a 1.0 mm)
    quanto exames de tórax inteiro (`HU-UFJF`, `300–430 px` a 1.0 mm) sem qualquer
    recorte manual dependente de Ground Truth.
    """

    def __init__(
        self,
        num_classes: int,
        patch_size: int = 256,
        overlap: float = 0.5,
        tta_angles_deg: Sequence[float] = (0.0, -8.0, 8.0),
    ) -> None:
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.overlap = float(overlap)
        self.tta_angles_deg = tuple(float(a) for a in tta_angles_deg)
        self._gaussian_window = self._build_gaussian_window(self.patch_size)

    @staticmethod
    def _build_gaussian_window(size: int) -> np.ndarray:
        """Constrói mapa de pesos gaussiano 2D para mistura suave nas bordas dos patches."""
        coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        sigma = 0.45
        window = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
        return np.clip(window, 1e-3, 1.0).astype(np.float32)

    @staticmethod
    def _compute_tile_starts(dim: int, patch: int, overlap: float) -> List[int]:
        """Calcula posições iniciais de recorte cobrindo 100% da dimensão."""
        if dim <= patch:
            return [0]
        stride = max(1, int(round(patch * (1.0 - overlap))))
        starts = list(range(0, dim - patch + 1, stride))
        if starts[-1] != dim - patch:
            starts.append(dim - patch)
        return starts

    def _predict_single_slice_probs(
        self,
        model: object,
        device: object,
        slice_2d: np.ndarray,
    ) -> np.ndarray:
        """Prevê probabilidades `(C, H, W)` de uma fatia 2D usando janela deslizante."""
        import torch

        h, w = slice_2d.shape
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        padded = np.pad(
            slice_2d,
            ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)),
            mode="constant",
            constant_values=0.0,
        )
        ph, pw = padded.shape

        y_starts = self._compute_tile_starts(ph, self.patch_size, self.overlap)
        x_starts = self._compute_tile_starts(pw, self.patch_size, self.overlap)

        tiles: List[np.ndarray] = []
        coords: List[Tuple[int, int]] = []
        for y0 in y_starts:
            for x0 in x_starts:
                tiles.append(padded[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size])
                coords.append((y0, x0))

        batch_np = np.stack(tiles, axis=0)[:, None, :, :].astype(np.float32)
        batch_tensor = torch.from_numpy(batch_np).to(device)

        with torch.no_grad():
            use_amp = str(device).startswith("cuda")
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(batch_tensor)
            probs_batch = torch.softmax(logits.float(), dim=1).cpu().numpy()

        acc_probs = np.zeros((self.num_classes, ph, pw), dtype=np.float32)
        acc_weights = np.zeros((ph, pw), dtype=np.float32)
        for idx, (y0, x0) in enumerate(coords):
            acc_probs[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size] += (
                probs_batch[idx] * self._gaussian_window[None, :, :]
            )
            acc_weights[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size] += self._gaussian_window

        acc_probs /= np.maximum(acc_weights[None, :, :], 1e-6)
        return acc_probs[:, pad_top : pad_top + h, pad_left : pad_left + w]

    def predict_volume(
        self,
        model: object,
        device: object,
        prepared: PreparedVolume,
        post_processor: CardiacPostProcessor,
        use_tta: bool = True,
    ) -> np.ndarray:
        """Executa inferência 3D completa e devolve máscara `int16` no espaço nativo."""
        h_res, w_res, z_res = prepared.image_resampled.shape
        probs_res = np.zeros((self.num_classes, h_res, w_res, z_res), dtype=np.float32)
        angles = self.tta_angles_deg if use_tta else (0.0,)

        for z in range(z_res):
            slc = prepared.image_resampled[:, :, z]
            slc_prob_sum = np.zeros((self.num_classes, h_res, w_res), dtype=np.float32)
            for angle in angles:
                if abs(angle) < 1e-3:
                    p_ang = self._predict_single_slice_probs(model, device, slc)
                else:
                    rot_in = nd_rotate(slc, angle=angle, reshape=False, order=1, mode="constant", cval=0.0)
                    p_rot = self._predict_single_slice_probs(model, device, rot_in)
                    p_ang = np.stack(
                        [
                            nd_rotate(p_rot[c], angle=-angle, reshape=False, order=1, mode="constant", cval=0.0)
                            for c in range(self.num_classes)
                        ],
                        axis=0,
                    )
                slc_prob_sum += p_ang
            probs_res[:, :, :, z] = slc_prob_sum / float(len(angles))

        probs_native = VolumePreprocessor.resample_probs_to_native(probs_res, prepared.native_shape)
        pred_native = np.argmax(probs_native, axis=0).astype(np.int16)
        return post_processor.apply(pred_native)


class CombinedSegmentationLoss:
    """Perda híbrida `ce_weight * WeightedCE + dice_weight * MultiClassSoftDice`."""

    def __init__(self, class_weights: Sequence[float], ce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        import torch
        import torch.nn as nn

        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.num_classes = len(class_weights)
        self.weight_tensor = torch.tensor(list(class_weights), dtype=torch.float32)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.weight_tensor)

    def to(self, device: object) -> "CombinedSegmentationLoss":
        """Move pesos da perda para o dispositivo alvo (CPU/CUDA)."""
        import torch.nn as nn

        self.weight_tensor = self.weight_tensor.to(device)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.weight_tensor)
        return self

    def __call__(self, logits: object, targets: object) -> object:
        """Calcula perda combinada ignorando o fundo na parcela Soft Dice."""
        import torch
        import torch.nn.functional as F

        ce = self.ce_loss_fn(logits, targets)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Soft Dice sobre as classes de primeiro plano (1..C-1), ponderando por batch inteiro
        probs_fg = probs[:, 1:]
        target_fg = one_hot[:, 1:]
        dims = (0, 2, 3)
        intersection = torch.sum(probs_fg * target_fg, dim=dims)
        denom = torch.sum(probs_fg + target_fg, dim=dims)
        smooth = 1.0
        dice_per_class = (2.0 * intersection + smooth) / (denom + smooth)
        dice_loss = 1.0 - dice_per_class.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


class UnetResNet34Pipeline:
    """Orquestrador POO completo de treino, inferência, avaliação 3D e exportação NIfTI."""

    def __init__(
        self,
        config: UnetResNet34Config,
        cmr_dir: Path,
        output_dir: Path,
        hu_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.cmr_dir = cmr_dir
        self.hu_dir = hu_dir
        self.output_dir = output_dir
        self.preprocessor = VolumePreprocessor(target_spacing_mm=config.target_spacing_mm)
        self.post_processor = CardiacPostProcessor(
            task_mode=config.task_mode,
            min_isolated_scar_voxels=config.min_isolated_scar_voxels,
        )
        self.inferer = SlidingWindow2DInferer(
            num_classes=config.task_mode.num_classes,
            patch_size=config.patch_size,
            overlap=config.sliding_window_overlap,
            tta_angles_deg=config.tta_angles_deg,
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Fixa sementes determinísticas em Python, NumPy e PyTorch."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _build_model(self) -> object:
        """Instancia `segmentation_models_pytorch.Unet` com encoder ResNet34."""
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name=self.config.encoder_name,
            encoder_weights=self.config.encoder_weights,
            in_channels=1,
            classes=self.config.task_mode.num_classes,
            decoder_use_batchnorm=True,
        )

    def _compute_validation_score(
        self,
        model: object,
        device: object,
        val_volumes: Sequence[PreparedVolume],
    ) -> Tuple[float, Dict[str, float]]:
        """Calcula Dice 3D por estrutura na validação para seleção do melhor checkpoint."""
        model.eval()
        lv_scores: List[float] = []
        myo_scores: List[float] = []
        scar_scores: List[float] = []
        rv_scores: List[float] = []

        for vol in val_volumes:
            assert vol.mask_native is not None
            pred_native = self.inferer.predict_volume(
                model=model,
                device=device,
                prepared=vol,
                post_processor=self.post_processor,
                use_tta=False,
            )
            gt = vol.mask_native
            lv_scores.append(BinarySurfaceMetrics.compute_dice(gt == 1, pred_native == 1))
            if self.config.task_mode is TaskMode.ANATOMY_4C:
                myo_scores.append(BinarySurfaceMetrics.compute_dice(gt == 2, pred_native == 2))
                rv_scores.append(BinarySurfaceMetrics.compute_dice(gt == 3, pred_native == 3))
            else:
                myo_scores.append(
                    BinarySurfaceMetrics.compute_dice((gt == 2) | (gt == 3), (pred_native == 2) | (pred_native == 3))
                )
                d_scar = BinarySurfaceMetrics.compute_dice(gt == 3, pred_native == 3)
                if not math.isnan(d_scar):
                    scar_scores.append(d_scar)
                rv_scores.append(BinarySurfaceMetrics.compute_dice(gt == 4, pred_native == 4))

        mean_lv = float(np.nanmean(lv_scores)) if lv_scores else 0.0
        mean_myo = float(np.nanmean(myo_scores)) if myo_scores else 0.0
        mean_rv = float(np.nanmean(rv_scores)) if rv_scores else 0.0
        mean_scar = float(np.nanmean(scar_scores)) if scar_scores else float("nan")

        if self.config.task_mode is TaskMode.ANATOMY_4C:
            composite = (mean_lv + mean_myo + mean_rv) / 3.0
        else:
            scar_term = 0.0 if math.isnan(mean_scar) else mean_scar
            composite = 0.20 * mean_lv + 0.25 * mean_myo + 0.35 * scar_term + 0.20 * mean_rv

        return composite, {
            "val_dice_lv": mean_lv,
            "val_dice_myo_total": mean_myo,
            "val_dice_scar": mean_scar,
            "val_dice_rv": mean_rv,
            "val_composite": composite,
        }

    @staticmethod
    def _save_training_curves(history: Sequence[Dict[str, float]], out_path: Path) -> None:
        """Plota perda de treino e evolução do Dice 3D na validação."""
        if not history:
            return
        epochs = [int(h["epoch"]) for h in history]
        losses = [h["train_loss"] for h in history]
        comp = [h["val_composite"] for h in history]
        myo = [h["val_dice_myo_total"] for h in history]
        scar = [h["val_dice_scar"] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        axes[0].plot(epochs, losses, color="#2563eb", linewidth=1.8)
        axes[0].set_title("Perda de Treino (CE + Soft Dice)")
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Loss")
        axes[0].grid(linestyle="--", alpha=0.4)

        axes[1].plot(epochs, comp, label="Score Composto Val", color="#0f172a", linewidth=2.0)
        axes[1].plot(epochs, myo, label="Dice Mio Total", color="#16a34a", linewidth=1.5)
        if any(not math.isnan(v) for v in scar):
            axes[1].plot(epochs, scar, label="Dice Fibrose (GT+)", color="#dc2626", linewidth=1.5)
        axes[1].set_ylim(0.0, 1.02)
        axes[1].set_title("Validação 3D por Época")
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Dice 3D")
        axes[1].legend(frameon=False)
        axes[1].grid(linestyle="--", alpha=0.4)

        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    def _train_model(
        self,
        train_volumes: Sequence[PreparedVolume],
        val_volumes: Sequence[PreparedVolume],
        dry_run: bool = False,
    ) -> Tuple[object, object, Path]:
        """Executa ciclo de treinamento PyTorch com warmup + cosine LR e AMP."""
        import torch
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Dispositivo de treinamento: %s", device)

        augmentor = CardiacAugmentor()
        train_ds = CardiacSliceDataset(
            volumes=train_volumes,
            patch_size=self.config.patch_size,
            augmentor=augmentor,
            seed=self.config.seed,
            include_empty_slices=True,
        )
        batch_size = min(self.config.batch_size, len(train_ds))
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if dry_run else self.config.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        model = self._build_model().to(device)
        criterion = CombinedSegmentationLoss(
            class_weights=self.config.class_weights,
            ce_weight=self.config.ce_weight,
            dice_weight=self.config.dice_weight,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_epochs = 1 if dry_run else self.config.epochs
        warmup_epochs = min(self.config.warmup_epochs, max(1, total_epochs // 5))

        def lr_lambda(ep_idx: int) -> float:
            if ep_idx < warmup_epochs:
                return float(ep_idx + 1) / float(warmup_epochs)
            progress = float(ep_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = self.config.min_learning_rate / self.config.learning_rate
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        best_score = -1.0
        best_ckpt_path = self.output_dir / "checkpoint_best.pth"
        history: List[Dict[str, float]] = []

        for epoch in range(1, total_epochs + 1):
            model.train()
            running_loss = 0.0
            num_batches = 0

            for imgs, masks in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.item())
                num_batches += 1
                if dry_run:
                    break

            scheduler.step()
            avg_loss = running_loss / max(1, num_batches)
            current_lr = float(optimizer.param_groups[0]["lr"])

            # Avalia 3D a cada época nas primeiras 10 e a cada 2 épocas depois (ou última)
            should_eval = dry_run or epoch <= 10 or (epoch % 2 == 0) or (epoch == total_epochs)
            if should_eval:
                val_comp, val_metrics = self._compute_validation_score(model, device, val_volumes)
                row = {
                    "epoch": float(epoch),
                    "lr": current_lr,
                    "train_loss": avg_loss,
                    **val_metrics,
                }
                history.append(row)

                if val_comp > best_score:
                    best_score = val_comp
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_val_composite": best_score,
                            "config": self.config.to_serializable_dict(),
                            "model_state_dict": model.state_dict(),
                        },
                        str(best_ckpt_path),
                    )

                LOGGER.info(
                    "Epoch %03d/%03d | lr=%.2e | loss=%.4f | val_comp=%.4f | VE=%.4f Mio=%.4f Scar=%s VD=%.4f%s",
                    epoch,
                    total_epochs,
                    current_lr,
                    avg_loss,
                    val_comp,
                    val_metrics["val_dice_lv"],
                    val_metrics["val_dice_myo_total"],
                    "NaN" if math.isnan(val_metrics["val_dice_scar"]) else f"{val_metrics['val_dice_scar']:.4f}",
                    val_metrics["val_dice_rv"],
                    " [BEST]" if val_comp >= best_score else "",
                )

        # Persiste histórico CSV e curvas de treinamento
        if history:
            hist_csv = self.output_dir / "training_history.csv"
            with hist_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
                writer.writeheader()
                for row in history:
                    writer.writerow(
                        {k: ("" if isinstance(v, float) and math.isnan(v) else f"{v:.6f}") for k, v in row.items()}
                    )
            self._save_training_curves(history, self.output_dir / "training_curves.png")

        # Recarrega o melhor checkpoint antes da inferência de teste
        if best_ckpt_path.exists():
            ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            LOGGER.info("Melhor checkpoint restaurado (época %d, score=%.4f).", ckpt["epoch"], ckpt["best_val_composite"])

        return model, device, best_ckpt_path

    def _evaluate_and_export_cohort(
        self,
        cohort_key: str,
        cohort_display_name: str,
        labeled_records: Sequence[VolumeRecord],
        unlabeled_records: Sequence[VolumeRecord],
        model: Optional[object],
        device: Optional[object],
        mock_inference: bool = False,
    ) -> Dict[str, object]:
        """Executa inferência, exportação NIfTI (`int16`), métricas 3D e figuras de uma coorte."""
        cohort_out = self.output_dir / cohort_key
        images_dir = cohort_out / "_tmp_eval" / "images"
        gt_dir = cohort_out / "_tmp_eval" / "gt"
        pred_dir = cohort_out / "_tmp_eval" / "pred"
        nifti_bundle_dir = cohort_out / f"niftis_{cohort_key}_{self.config.task_mode.value}"

        for folder in (images_dir, gt_dir, pred_dir, nifti_bundle_dir):
            folder.mkdir(parents=True, exist_ok=True)

        model_label = f"U-Net ResNet34 ({self.config.task_mode.num_classes}c)"

        # 1. Casos rotulados (inferência + salvamento LGE/GT/PRED padronizado + avaliação)
        for rec in labeled_records:
            prepared = self.preprocessor.prepare_volume(rec, task_mode=self.config.task_mode)
            assert prepared.mask_native is not None

            if mock_inference:
                # Simulação determinística apenas para dry-run em CPU sem PyTorch instalado
                pred_native = self.post_processor.apply(prepared.mask_native.copy())
            else:
                assert model is not None and device is not None
                pred_native = self.inferer.predict_volume(
                    model=model,
                    device=device,
                    prepared=prepared,
                    post_processor=self.post_processor,
                    use_tta=True,
                )

            img_native, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
            img_out = images_dir / f"{rec.patient_id}.nii.gz"
            gt_out = gt_dir / f"{rec.patient_id}.nii.gz"
            pred_out = pred_dir / f"{rec.patient_id}.nii.gz"

            # Salva imagem e máscaras padronizadas preservando affine e pixdim originais
            out_img_nii = nib.Nifti1Image(img_native.astype(np.float32), img_nii.affine)
            out_img_nii.header["pixdim"] = img_nii.header["pixdim"]
            out_img_nii.set_data_dtype(np.float32)
            nib.save(out_img_nii, str(img_out))

            NiftiVolumeIO.save_mask_like(
                prepared.mask_native, prepared.native_affine, prepared.native_header_pixdim, gt_out, dtype=np.int16
            )
            NiftiVolumeIO.save_mask_like(
                pred_native, prepared.native_affine, prepared.native_header_pixdim, pred_out, dtype=np.int16
            )

            # Organiza pacote limpo por paciente para inspeção no 3D Slicer
            p_bundle = nifti_bundle_dir / rec.patient_id
            p_bundle.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_out, p_bundle / f"{rec.patient_id}_LGE.nii.gz")
            shutil.copy2(gt_out, p_bundle / f"{rec.patient_id}_GT_{self.config.task_mode.value}.nii.gz")
            shutil.copy2(pred_out, p_bundle / f"{rec.patient_id}_PRED_{self.config.task_mode.value}.nii.gz")

        # 2. Métricas 3D (Clínicas + CMR-MULTI MICCAI 2026) e painéis qualitativos
        evaluator = CardiacMetrics3DEvaluator(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=cohort_out,
            task_mode=self.config.task_mode,
            cohort_name=cohort_display_name,
        )
        summary = evaluator.run()

        visualizer = SegmentationPanelVisualizer(
            image_dir=images_dir,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=cohort_out / "figures",
            task_mode=self.config.task_mode,
            model_label=model_label,
        )
        visualizer.run()

        # 3. Casos clínicos sem GT (ex.: HU-UFJF patient_20): inferência + NIfTI + figura qualitativa
        for rec in unlabeled_records:
            prepared = self.preprocessor.prepare_volume(rec, task_mode=self.config.task_mode)
            if mock_inference:
                pred_native = np.zeros(prepared.native_shape, dtype=np.int16)
            else:
                assert model is not None and device is not None
                pred_native = self.inferer.predict_volume(
                    model=model,
                    device=device,
                    prepared=prepared,
                    post_processor=self.post_processor,
                    use_tta=True,
                )

            p_bundle = nifti_bundle_dir / rec.patient_id
            p_bundle.mkdir(parents=True, exist_ok=True)
            img_native, img_nii = NiftiVolumeIO.load_volume(rec.image_path, is_mask=False)
            img_bundle_path = p_bundle / f"{rec.patient_id}_LGE.nii.gz"
            out_img_nii = nib.Nifti1Image(img_native.astype(np.float32), img_nii.affine)
            out_img_nii.header["pixdim"] = img_nii.header["pixdim"]
            out_img_nii.set_data_dtype(np.float32)
            nib.save(out_img_nii, str(img_bundle_path))

            pred_bundle_path = p_bundle / f"{rec.patient_id}_PRED_{self.config.task_mode.value}.nii.gz"
            NiftiVolumeIO.save_mask_like(
                pred_native, prepared.native_affine, prepared.native_header_pixdim, pred_bundle_path, dtype=np.int16
            )
            visualizer.render_unlabeled_case(rec.patient_id, img_bundle_path, pred_bundle_path)

        # 4. Empacota NIfTIs em ZIP leve e limpa diretório temporário de avaliação
        zip_base = cohort_out / f"niftis_predicoes_{cohort_key}_{self.config.task_mode.value}"
        zip_path = Path(shutil.make_archive(str(zip_base), "zip", str(nifti_bundle_dir)))
        shutil.rmtree(nifti_bundle_dir, ignore_errors=True)
        shutil.rmtree(cohort_out / "_tmp_eval", ignore_errors=True)
        LOGGER.info("Pacote NIfTI da coorte %s salvo em %s", cohort_display_name, zip_path)
        return summary

    def run(self, dry_run: bool = False) -> Dict[str, object]:
        """Executa o fluxo completo: preparação, treino, avaliação CMR-MULTI e avaliação HU-UFJF."""
        self._seed_everything(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Salva configuração exata para reprodutibilidade total
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config.to_serializable_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 1. Descobre registros CMR-MULTI e aplica split estratificado oficial SEED=42
        cmr_loader = CmrMultiCohortLoader(self.cmr_dir)
        all_cmr_records = cmr_loader.discover_records()
        scar_map = cmr_loader.build_scar_presence_map(all_cmr_records)

        split_gen = StratifiedSplitGenerator(BenchmarkSplitConfig(seed=self.config.seed))
        splits = split_gen.split_train_val_test(scar_map, dry_run=dry_run)
        split_gen.save_splits_json(splits, self.output_dir / "splits_used.json")

        rec_by_id = {r.patient_id: r for r in all_cmr_records}
        train_records = [rec_by_id[cid] for cid in splits["train"]]
        val_records = [rec_by_id[cid] for cid in splits["val"]]
        test_records = [rec_by_id[cid] for cid in splits["test"]]

        LOGGER.info(
            "Preparando volumes CMR-MULTI (%s, %.2f mm): train=%d | val=%d | test=%d",
            self.config.task_mode.value,
            self.config.target_spacing_mm,
            len(train_records),
            len(val_records),
            len(test_records),
        )
        train_vols = [self.preprocessor.prepare_volume(r, self.config.task_mode) for r in train_records]
        val_vols = [self.preprocessor.prepare_volume(r, self.config.task_mode) for r in val_records]

        # 2. Valida disponibilidade de PyTorch/SMP (permite dry-run estrutural em CPU sem torch)
        mock_mode = False
        model = None
        device = None
        try:
            import segmentation_models_pytorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            if not dry_run:
                raise ImportError(
                    "PyTorch e segmentation_models_pytorch são obrigatórios para treino real."
                ) from exc
            LOGGER.warning(
                "PyTorch/SMP não instalados neste ambiente CPU local (%s); "
                "executando validação dry-run completa de dados, augmentation, NIfTI, métricas e figuras.",
                exc,
            )
            # Valida explicitamente o pipeline de dataset e augmentation em CPU
            aug = CardiacAugmentor()
            rng = np.random.default_rng(self.config.seed)
            ds = CardiacSliceDataset(train_vols, patch_size=self.config.patch_size, augmentor=aug)
            img0, msk0, _, _ = ds.samples[0]
            aug_img, aug_msk = aug(img0, msk0, rng)
            LOGGER.info(
                "Sanity check dry-run OK: %d fatias treino | shape=%s | rótulos únicos=%s",
                len(ds),
                aug_img.shape,
                np.unique(aug_msk).tolist(),
            )
            mock_mode = True

        if not mock_mode:
            model, device, _ = self._train_model(train_vols, val_vols, dry_run=dry_run)

        # 3. Avaliação no conjunto de teste inédito CMR-MULTI (N=7 no modo real)
        report: Dict[str, object] = {
            "task_mode": self.config.task_mode.value,
            "dry_run": dry_run,
            "mock_inference": mock_mode,
        }
        report["cmr_multi_test"] = self._evaluate_and_export_cohort(
            cohort_key="cmr_multi_test",
            cohort_display_name="CMR-MULTI Test",
            labeled_records=test_records,
            unlabeled_records=[],
            model=model,
            device=device,
            mock_inference=mock_mode,
        )

        # 4. Avaliação externa na coorte clínica privada HU-UFJF (se fornecida)
        if self.hu_dir is not None and self.hu_dir.exists():
            hu_loader = HuUfjfCohortLoader(self.hu_dir)
            hu_labeled, hu_unlabeled = hu_loader.discover_records()
            if hu_labeled:
                report["hu_ufjf_external"] = self._evaluate_and_export_cohort(
                    cohort_key="hu_ufjf_external",
                    cohort_display_name="HU-UFJF Private Cohort",
                    labeled_records=hu_labeled,
                    unlabeled_records=hu_unlabeled,
                    model=model,
                    device=device,
                    mock_inference=mock_mode,
                )

        (self.output_dir / "experiment_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("Pipeline U-Net ResNet34 (%s) finalizado em %s", self.config.task_mode.value, self.output_dir)
        return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Define argumentos CLI em inglês."""
    parser = argparse.ArgumentParser(
        description="Treina e avalia U-Net ResNet34 (4c ou 5c) em CMR-MULTI e coorte clínica HU-UFJF."
    )
    parser.add_argument("--cmr-dir", type=Path, required=True, help="Diretório do dataset CMR-MULTI (ou data_sample/).")
    parser.add_argument("--hu-dir", type=Path, default=None, help="Diretório opcional do dataset privado HU-UFJF.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório raiz para checkpoints, métricas e NIfTIs.")
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=[m.value for m in TaskMode],
        required=True,
        help="`anatomy_4c` (VE, MioTotal, VD sem Scar) ou `full_5c` (VE, MioSaudavel, Scar, VD).",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Número total de épocas de treino.")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamanho do mini-batch 2D.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Taxa de aprendizado inicial (AdamW).")
    parser.add_argument("--patch-size", type=int, default=256, help="Dimensão quadrada de entrada (múltiplo de 32).")
    parser.add_argument("--target-spacing-mm", type=float, default=1.0, help="Espaçamento isotrópico no plano XY (mm).")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão oficial SEED=42).")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers do DataLoader PyTorch.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa validação rápida em CPU sobre subconjunto mínimo (ex.: data_sample/).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI do pipeline U-Net ResNet34."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    config = UnetResNet34Config(
        task_mode=TaskMode(args.task_mode),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patch_size=args.patch_size,
        target_spacing_mm=args.target_spacing_mm,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    pipeline = UnetResNet34Pipeline(
        config=config,
        cmr_dir=args.cmr_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        hu_dir=args.hu_dir.resolve() if args.hu_dir is not None else None,
    )
    pipeline.run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
