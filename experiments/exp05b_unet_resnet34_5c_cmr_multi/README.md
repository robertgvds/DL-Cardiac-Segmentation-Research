# Exp 05B — U-Net ResNet34 (5 Classes — Anatomia + Scar) no CMR-MULTI + HU-UFJF

## Configuração
- **Arquitetura:** `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=5, decoder_use_batchnorm=True)`
- **Classes (5):** `0=Fundo`, `1=VE`, `2=Miocárdio Saudável`, `3=Fibrose (Scar)`, `4=VD`.
- **Pré-processamento & Augmentation:** Reamostragem física isotrópica `1.0 × 1.0 mm` no plano XY, normalização robusta (`p1–p99` + Z-score), `patch_size=256`, rotação `±25°`, escala `0.80–1.25×`, translação `±12%`, deformação elástica leve, **blur gaussiano multi-resolução** (Hoh et al., 2024), `flip=False`.
- **Treino:** `0.5 × WeightedCE([0.5, 1.5, 2.0, 4.5, 1.5]) + 0.5 × SoftDice`, `AdamW(lr=3e-4, wd=1e-4)`, warmup 5 ep + `CosineAnnealingLR`, 120 épocas, `batch_size=16`, AMP, split oficial `SEED=42` (`35 / 5 / 7`).
- **Inferência:** Janela deslizante gaussiana `256×256` (sobreposição 50%) + TTA rotacional `(0°, -8°, +8°)` + pós-processamento 3D (maior componente conexo cardíaco + remoção de micro-ruído isolado de fibrose).
- **Kernel Kaggle:** `robertgvds/dlcs-unet-resnet34-5c` (Concluído).

---

## 1. Resultados — CMR-MULTI Test Inédito (`N=7`, `SEED=42`)
- **Dice VE:** **`0.8896 ± 0.0272`** (mediana `0.8839`) | **HD95:** `7.34 mm` | **HD max:** `14.05 mm` | **ASD:** `1.36 mm`
- **Dice Miocárdio Total:** `0.7967 ± 0.0452` (mediana `0.7896`) | **HD95:** **`4.56 mm`** | **HD max:** `13.21 mm` | **ASD:** **`0.91 mm`**
- **Dice Miocárdio Saudável:** `0.7753 ± 0.0465` (mediana `0.7717`)
- **Dice Fibrose (`N=5` GT+):** `0.5480 ± 0.1341` (mediana `0.4796`, máx. **`0.7307`** em `LGE_SAX_007`, `0.6501` em `LGE_SAX_026`) | **HD95:** `11.95 mm` | **ASD:** `2.77 mm`
- **Dice VD:** **`0.7652 ± 0.0381`** (mediana `0.7604`) | **HD95:** `12.13 mm` | **HD max:** `19.77 mm` | **ASD:** `2.38 mm`
- **Saudáveis (`N=2`):** `LGE_SAX_008` = **0 FP (`0.00 mL`)**; `LGE_SAX_005` = 31 voxels (`0.12 mL`).
- **Métricas Oficiais MICCAI 2026 (Task 2 LGE):**
  - **Board DSC:** `0.7411`
  - **Scar Mass vPCC:** **`0.7890`**
  - **Scar Mass RAE (N=5):** **`0.2753`** (`27.53%`, erro absoluto mediano `6.41 mL`)
  - **Score Oficial Task 2:** **`0.7548`**

Arquivos: [metrics_per_patient.csv](cmr_multi_test/metrics_per_patient.csv) | [summary.json](cmr_multi_test/summary.json) | [dice_boxplot.png](cmr_multi_test/dice_boxplot.png) | [overview_panel.png](cmr_multi_test/overview_panel.png)

---

## 2. Resultados — Avaliação Externa *Zero-Shot* HU-UFJF (`N=10` rotulados + `patient_20`)
*(Modelo treinado exclusivamente no CMR-MULTI e aplicado sem fine-tuning aos exames clínicos de miocardite do HU-UFJF.)*

- **Dice VE:** **`0.8232 ± 0.0849`** (mediana `0.8411`, máx. `0.9042`) | **HD95:** `16.87 mm` | **ASD:** `4.62 mm`
- **Dice Miocárdio Total:** **`0.6513 ± 0.0897`** (mediana `0.6863`, máx. `0.7422`) | **HD95:** `11.51 mm` | **ASD:** **`2.58 mm`**
- **Dice Miocárdio Saudável:** `0.6324 ± 0.0801`
- **Dice Fibrose (`N=10`):** `0.1286 ± 0.1439` (mediana `0.0507`, máx. `0.3933` em `patient_7` e `0.3309` em `patient_19`)
- **Dice VD:** `0.6281 ± 0.1775` (mediana `0.7124`)
- **Métricas MICCAI 2026 (Task 2 Zero-Shot):** `Board DSC = 0.5531` | `vPCC = -0.0437` | `RAE = 5.4335` | `Task 2 Score = 0.4105`
- **Diagnóstico Científico:** Enquanto a anatomia transferiu bem *zero-shot* (`VE=0.823`, `Mio=0.651`, `ASD=2.58 mm`), a fibrose sofreu forte *domain shift* clínico/patológico: o CMR-MULTI concentra cicatrizes extensas/isquêmicas (`22–77 mL`), ao passo que a miocardite no HU-UFJF apresenta focos mesocárdicos/subepicárdicos pequenos (`1.5–10 mL` em 6/10 casos), gerando superestimação sem adaptação de domínio ou cascata ROI. Nos casos com maior carga fibrótica (`patient_7`: `36.3 mL`, `patient_19`: `34.8 mL`), o modelo atingiu `0.393` e `0.331` (`RAE=12.5%` no `patient_19`).

Arquivos: [metrics_per_patient.csv](hu_ufjf_external/metrics_per_patient.csv) | [summary.json](hu_ufjf_external/summary.json) | [dice_boxplot.png](hu_ufjf_external/dice_boxplot.png) *(painéis de imagem clínica HU-UFJF e NIfTIs mantidos exclusivamente em `outputs_local/` por sigilo hospitalar).*

> **NIfTIs preditos (`int16`, affine/pixdim preservados) e `checkpoint_best.pth`:** armazenados localmente (gitignored) em `outputs_local/exp05b_unet_resnet34_5c/export/`.
