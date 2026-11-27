# Exp 05A — U-Net ResNet34 (4 Classes — Anatomia sem Scar) no CMR-MULTI + HU-UFJF

## Objetivo
Avaliar o desempenho da **U-Net com encoder ResNet34 (ImageNet)** na segmentação anatômica das câmaras cardíacas sem exigir a separação da fibrose (unindo Miocárdio Saudável + Scar em uma única classe `Miocárdio Total`), tanto no conjunto de teste inédito do **CMR-MULTI (`N=7`, `SEED=42`)** quanto em generalização externa *zero-shot* na coorte clínica privada **HU-UFJF (`N=10` rotulados + `patient_20`)**.

## Configuração Baseada na Literatura (Hoh et al. 2024; CMR-MULTI 2026)
- **Classes (4):** `0=Fundo`, `1=VE`, `2=Miocárdio Total (Mio + Scar)`, `3=VD`.
- **Reamostragem física:** `1.0 mm × 1.0 mm` no plano XY (alinhando escala física entre CMR-MULTI e HU-UFJF), `patch_size=256`.
- **Augmentation (treino):** Rotação `±25°`, escala `0.80–1.25×`, translação `±12%`, deformação elástica leve, **blur gaussiano multi-resolução** (`σ ∈ [0.5, 1.6]`), brilho/contraste/gama e ruído (`flip=False`).
- **Perda:** `0.5 × WeightedCE([0.5, 1.5, 2.5, 1.5]) + 0.5 × SoftDice(ignore_bg=True)`.
- **Otimização:** `AdamW(lr=3e-4, wd=1e-4)`, warmup 5 épocas + `CosineAnnealingLR`, 120 épocas, `batch_size=16`, AMP.
- **Inferência:** Janela deslizante gaussiana `256×256` (sobreposição 50%) + TTA rotacional `(0°, -8°, +8°)` + pós-processamento 3D (maior componente conexo cardíaco).
- **Runner Kaggle:** `kaggle_jobs/dlcs_unet_resnet34_4c/` (`robertgvds/dlcs-unet-resnet34-4c`).

## Status
*Aguardando execução sequencial na GPU do Kaggle.*
