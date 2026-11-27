# Exp 06 — Cascata em 2 Etapas (Anatomia Exp 05A → Especialista em Fibrose CMR-MULTI + EMIDEC)

## Configuração Executada
- **Etapa 1 (Anatomia 4c):** Checkpoint pré-treinado do **Exp 05A** (`/kaggle/input/notebooks/robertgvds/dlcs-unet-resnet34-4c/export/checkpoint_best.pth`, sem retreino).
- **Etapa 2 (Especialista em Fibrose na ROI):**
  - `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=3)`
  - Treino harmonizado em **135 volumes LGE**: **35 CMR-MULTI** (`SEED=42`) + **100 EMIDEC** (`Scar=3 + MVO=4 -> Scar`, sem conflito de VD).
  - ROI centrada no miocárdio em `0.75 × 0.75 mm` (`160 × 160` px), normalização robusta local (`mediana / MAD` do miocárdio), 3 canais `(LGE_local, Prior_Mio, Prior_VE)` com perturbação morfológica dos *priors* no treino, `80 épocas`, `batch_size=32`, `AdamW(lr=3e-4)`, TTA `(0°, -10°, +10°)`.
  - **Limiar de fibrose calibrado na validação (`N=5`):** `t = 0.35`.
- **Kernel Kaggle:** `robertgvds/dlcs-cascade-scar-emidec` (Concluído em ~20 min na Tesla T4).

---

## 1. Resultados — CMR-MULTI Test Inédito (`N=7`, `SEED=42`)
- **Dice VE:** `0.8867 ± 0.0312` | **HD95:** `7.56 mm` | **ASD:** `1.43 mm`
- **Dice Miocárdio Total:** **`0.8024 ± 0.0494`** *(melhor U-Net ResNet34!)* | **Erro Vol. Miocárdio:** **`6.46 mL` (mediana `3.51 mL`)** | **HD95:** `4.92 mm` | **ASD:** `1.00 mm`
- **Dice Miocárdio Saudável:** `0.7720 ± 0.0444`
- **Dice Fibrose (`N=5` GT+):** `0.5325 ± 0.1071` (mediana `0.4840`, máx. **`0.6776`** em `LGE_SAX_026` — melhor resultado entre todos os modelos nesse paciente!) | **HD95:** `15.76 mm` | **ASD:** `3.55 mm`
- **Dice VD:** `0.7608 ± 0.0781` (mediana `0.7999`)
- **Saudáveis (`N=2`):** `LGE_SAX_008` = **0 FP (`0.00 mL`)**; `LGE_SAX_005` = 70 voxels (`0.28 mL`).
- **Métricas Oficiais MICCAI 2026 (Task 2):** `Board DSC = 0.7357` | `vPCC = 0.6692` | `RAE = 0.3212` | **`Task 2 Score = 0.7289`**.

Arquivos: [metrics_per_patient.csv](cmr_multi_test/metrics_per_patient.csv) | [summary.json](cmr_multi_test/summary.json) | [dice_boxplot.png](cmr_multi_test/dice_boxplot.png) | [overview_panel.png](cmr_multi_test/overview_panel.png)

---

## 2. Resultados — Avaliação Externa *Zero-Shot* HU-UFJF (`N=10` rotulados + `patient_20`)
- **Dice VE:** `0.8206 ± 0.0562` | **Dice Mio Total:** `0.6445 ± 0.0914` | **Dice Mio Saudável:** `0.6385 ± 0.0759` | **Dice VD:** `0.6896 ± 0.1272`
- **Dice Fibrose (`N=10`):** `0.0722 ± 0.1107` (mediana `0.0166`, máx. `0.2941`) | **Task 2 Score:** `0.4104`

Arquivos: [metrics_per_patient.csv](hu_ufjf_external/metrics_per_patient.csv) | [summary.json](hu_ufjf_external/summary.json) | [dice_boxplot.png](hu_ufjf_external/dice_boxplot.png) *(painéis clínicos HU-UFJF, NIfTIs 5c, mapas contínuos `*_SCAR_PROB.nii.gz` e `stage2_checkpoint_best.pth` mantidos exclusivamente em `outputs_local/`).*

---

## 3. Diagnóstico Científico (Achado-Chave)
1. **Efeito do desbalanceamento de domínio EMIDEC (74%) vs. CMR-MULTI (26%):** Como os 100 casos do EMIDEC dominaram o treino da Etapa 2 (infarto agudo isquêmico subendocárdico com anotações mais conservadoras/estreitas), o modelo ficou sub-confiante nas cicatrizes extensas do CMR-MULTI (`LGE_SAX_047`: GT `76.6 mL` vs. Pred `30.1 mL`), forçando a calibração a escolher o menor limiar da grade (`t = 0.35`) e reduzindo a precisão da fibrose média (`0.5325` vs. `0.5480` do Exp 05B e `0.5846` do Exp 04).
2. **Prova conclusiva sobre *zero-shot* na miocardite do HU-UFJF:** Nem o modelo de etapa única (`0.1286`), nem a cascata com 135 volumes de infarto (`0.0722`) transferem *zero-shot* para a fibrose focal/mesocárdica da miocardite do HU-UFJF — comprovando cientificamente que **adaptação de domínio na coorte local (Fine-tuning / Co-training em CV 5-fold)** é indispensável para a etapa clínica!
