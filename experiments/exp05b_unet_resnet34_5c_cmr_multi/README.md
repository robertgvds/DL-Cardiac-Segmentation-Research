# Exp 05B — U-Net ResNet34 (5 Classes — Anatomia + Scar) no CMR-MULTI + HU-UFJF

## Objetivo
Avaliar o desempenho da **U-Net com encoder ResNet34 (ImageNet)** aprimorada com reamostragem física (`1.0 mm`), augmentation multi-resolução (Hoh et al., 2024), janela deslizante gaussiana e TTA na tarefa completa de **5 classes (incluindo Fibrose/Scar)**, comparando diretamente com Exp 03 (baseline legado) no **CMR-MULTI (`N=7`, `SEED=42`)** e avaliando generalização *zero-shot* na coorte clínica privada **HU-UFJF (`N=10` rotulados + `patient_20`)**.

## Configuração Baseada na Literatura (Hoh et al. 2024; CMR-MULTI 2026)
- **Classes (5):** `0=Fundo`, `1=VE`, `2=Miocárdio Saudável`, `3=Fibrose (Scar)`, `4=VD`.
- **Reamostragem física:** `1.0 mm × 1.0 mm` no plano XY, `patch_size=256`.
- **Augmentation (treino):** Rotação `±25°`, escala `0.80–1.25×`, translação `±12%`, deformação elástica leve, **blur gaussiano multi-resolução** (`σ ∈ [0.5, 1.6]`), brilho/contraste/gama e ruído (`flip=False`).
- **Perda:** `0.5 × WeightedCE([0.5, 1.5, 2.0, 4.5, 1.5]) + 0.5 × SoftDice(ignore_bg=True)`.
- **Otimização:** `AdamW(lr=3e-4, wd=1e-4)`, warmup 5 épocas + `CosineAnnealingLR`, 120 épocas, `batch_size=16`, AMP.
- **Inferência:** Janela deslizante gaussiana `256×256` (sobreposição 50%) + TTA rotacional `(0°, -8°, +8°)` + pós-processamento 3D (maior componente conexo cardíaco + limpeza de micro-ruído isolado de fibrose).
- **Métricas:** Dice 3D (GT+ e Board CMR-MULTI), HD95, HD máximo, ASD, Volume/Massa de fibrose (`1.05 g/mL`), vPCC, RAE e **Score Oficial Task 2 (MICCAI 2026)**.
- **Runner Kaggle:** `kaggle_jobs/dlcs_unet_resnet34_5c/` (`robertgvds/dlcs-unet-resnet34-5c`).

## Status
*Aguardando execução sequencial na GPU do Kaggle (após Exp 05A).*
