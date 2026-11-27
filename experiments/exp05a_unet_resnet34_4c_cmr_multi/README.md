# Exp 05A — U-Net ResNet34 (4 Classes — Anatomia sem Scar) no CMR-MULTI + HU-UFJF

## Configuração
- **Arquitetura:** `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=4, decoder_use_batchnorm=True)`
- **Classes (4):** `0=Fundo`, `1=VE`, `2=Miocárdio Total (Mio Saudável ∪ Scar)`, `3=VD`.
- **Pré-processamento & Augmentation:** Reamostragem física isotrópica `1.0 × 1.0 mm` no plano XY, normalização robusta (`p1–p99` + Z-score), `patch_size=256`, rotação `±25°`, escala `0.80–1.25×`, translação `±12%`, deformação elástica leve, **blur gaussiano multi-resolução** (Hoh et al., 2024), `flip=False`.
- **Treino:** `0.5 × WeightedCE([0.5, 1.5, 2.5, 1.5]) + 0.5 × SoftDice`, `AdamW(lr=3e-4, wd=1e-4)`, warmup 5 ep + `CosineAnnealingLR`, 120 épocas, `batch_size=16`, AMP, split oficial `SEED=42` (`35 / 5 / 7`).
- **Inferência:** Janela deslizante gaussiana `256×256` (sobreposição 50%) + TTA rotacional `(0°, -8°, +8°)` + pós-processamento 3D (maior componente conexo cardíaco).
- **Kernel Kaggle:** `robertgvds/dlcs-unet-resnet34-4c` (Concluído).

---

## 1. Resultados — CMR-MULTI Test Inédito (`N=7`, `SEED=42`)
- **Dice VE:** `0.8871 ± 0.0289` (mediana `0.8843`) | **HD95:** `7.58 mm` | **HD max:** `12.36 mm` | **ASD:** `1.42 mm`
- **Dice Miocárdio Total:** **`0.8004 ± 0.0505`** (mediana **`0.8076`**) | **HD95:** `4.89 mm` | **HD max:** `14.76 mm` | **ASD:** `1.01 mm`
- **Dice VD:** `0.7608 ± 0.0781` (mediana `0.7999`) | **HD95:** `11.68 mm` | **HD max:** `23.89 mm` | **ASD:** `2.60 mm`
- **Erro Volumétrico Miocárdio:** `7.86 ± 6.93 mL` (mediana `6.31 mL`)
- **Métricas MICCAI 2026 (Anatomia):** `Board DSC = 0.8161` | `HD médio = 17.00 mm` | `ASD médio = 1.68 mm` | `Score Geométrico = 0.4551`

Arquivos: [metrics_per_patient.csv](cmr_multi_test/metrics_per_patient.csv) | [summary.json](cmr_multi_test/summary.json) | [dice_boxplot.png](cmr_multi_test/dice_boxplot.png) | [overview_panel.png](cmr_multi_test/overview_panel.png)

---

## 2. Resultados — Avaliação Externa *Zero-Shot* HU-UFJF (`N=10` rotulados + `patient_20`)
*(Modelo treinado exclusivamente no CMR-MULTI e aplicado sem fine-tuning aos exames clínicos de tórax inteiro do HU-UFJF com espaçamento nativo `1.83–2.68 mm/px`.)*

- **Dice VE:** **`0.8235 ± 0.0566`** (mediana `0.8251`, máx. `0.8824`) | **HD95:** `13.43 mm` | **ASD:** `3.76 mm`
- **Dice Miocárdio Total:** `0.6470 ± 0.0893` (mediana `0.6588`, máx. `0.7714`) | **HD95:** `12.77 mm` | **ASD:** `2.89 mm`
- **Dice VD:** **`0.6896 ± 0.1272`** (mediana `0.7188`, máx. `0.8892`) | **HD95:** `15.15 mm` | **ASD:** `4.72 mm`
- **Métricas MICCAI 2026 (Anatomia):** `Board DSC = 0.7200` | `HD médio = 28.39 mm` | `ASD médio = 3.79 mm` | `Score Geométrico = 0.3608`

Arquivos: [metrics_per_patient.csv](hu_ufjf_external/metrics_per_patient.csv) | [summary.json](hu_ufjf_external/summary.json) | [dice_boxplot.png](hu_ufjf_external/dice_boxplot.png) *(painéis de imagem clínica HU-UFJF e NIfTIs mantidos exclusivamente em `outputs_local/` por sigilo hospitalar).*

> **NIfTIs preditos (`int16`, affine/pixdim preservados) e `checkpoint_best.pth`:** armazenados localmente (gitignored) em `outputs_local/exp05a_unet_resnet34_4c/export/`.
