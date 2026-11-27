# Exp 04 — nnU-Net v2 2D no CMR-MULTI LGE SAX (`Dataset501_CMRMULTI_LGE`)

## Configuração
- **Framework / Arquitetura:** `nnU-Net v2` (`2d`, planos auto-gerados em [nnUNetPlans.json](nnUNetPlans.json)).
- **Dataset:** `Dataset501_CMRMULTI_LGE` (5 classes: `0=Bg, 1=VE, 2=Mio Saudável, 3=Fibrose, 4=VD`).
- **Protocolo:** Split estratificado oficial `SEED=42` (`imagesTr=40` com Fold 0 `35 treino / 5 val`, `imagesTs=7` inéditos), `nnUNetTrainer_250epochs` (execução v2 Kaggle, GPU Tesla T4), inferência com TTA padrão da nnU-Net v2.
- **Kernel Kaggle:** `robertgvds/dlcs-nnunet-2d` (Concluído).

---

## Resultados Consolidados (Teste Inédito, `N=7`)
- **Dice VE:** **`0.9008 ± 0.0183`** (mediana `0.8987`, mín. `0.8769`, máx. `0.9246`) | **HD95:** `6.00 mm`
- **Dice Miocárdio Total:** **`0.8085 ± 0.0331`** (mediana `0.8123`, mín. `0.7577`, máx. `0.8545`) | **HD95:** `4.64 mm` *(mediana `3.03 mm`)* | **ASD:** **`0.89 mm`**
- **Dice Miocárdio Saudável:** **`0.7959 ± 0.0258`** (mediana `0.7969`)
- **Dice Fibrose (`N=5` GT+):** **`0.5846 ± 0.0907`** (mediana **`0.6316`**, máx. `0.6654` em `LGE_SAX_007`) | **HD95:** `13.79 mm` | **ASD:** `3.00 mm`
- **Dice VD:** **`0.7673 ± 0.0541`** (mediana `0.7694`, máx. `0.8253`) | **HD95:** `10.80 mm`
- **Volume de Fibrose:** erro absoluto médio `8.45 ± 10.55 mL` (mediana **`4.42 mL`**), erro relativo médio **`24.82%`** (mediana `20.01%`).
- **Saudáveis (`N=2`):** `LGE_SAX_008` = 5 voxels (`0.025 mL` FP); `LGE_SAX_005` = 205 voxels (`0.824 mL` FP).
- **Métricas Oficiais MICCAI 2026 (Task 2 LGE):**
  - **Board DSC:** `0.7204`
  - **Scar Mass vPCC:** **`0.9585`**
  - **Scar Mass RAE (`N=5`):** **`0.2482`**
  - **Score Oficial Task 2:** **`0.7682`**

## Artefatos Arquivados
- [metrics_per_patient.csv](metrics_per_patient.csv) | [summary.json](summary.json)
- [metrics_per_patient_miccai2026.csv](metrics_per_patient_miccai2026.csv) | [summary_miccai2026.json](summary_miccai2026.json)
- [nnUNetPlans.json](nnUNetPlans.json) | [dice_boxplot.png](dice_boxplot.png) | [overview_test_panel.png](overview_test_panel.png)

> **NIfTIs preditos (`int16`, affine/pixdim preservados):** armazenados localmente (gitignored) em `outputs_local/exp04_nnunet_v2_2d/export/niftis_predicoes_nnunet_2d.zip`.
> **Observação sobre pesos (`.pth`):** Esta execução (v2, 250 épocas) precedeu o commit `7f4627e` que adicionou a exportação de `checkpoint_best.pth` e `progress.png`; portanto o arquivo de pesos não está contido neste pacote de saída.
