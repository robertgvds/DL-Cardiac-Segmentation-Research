# Log Vivo de Experimentos — Segmentação Cardíaca e de Fibrose em LGE-CMR

Todos os experimentos no CMR-MULTI utilizam rigorosamente o **mesmo split estratificado por paciente (`SEED=42`)** para comparação justa:
- **Treino (35 volumes — 26 com fibrose, 9 sem):** `LGE_SAX_001, 002, 004, 006, 009, 012, 013, 015, 016, 017, 018, 019, 020, 021, 022, 023, 024, 025, 027, 029, 030, 031, 033, 035, 036, 037, 038, 039, 040, 041, 042, 043, 044, 045, 046`
- **Validação (5 volumes — 4 com fibrose, 1 sem):** `LGE_SAX_003, 010, 011, 032, 034`
- **Teste Inédito (7 volumes — 5 com fibrose, 2 sem):** `LGE_SAX_005, 007, 008, 014, 026, 028, 047`

---

## 1. Resumo Consolidado (CMR-MULTI LGE SAX — Teste Inédito N=7)

| Experimento | Arquitetura / Configuração | Dice VE | Dice Miocárdio Total | Dice Mio Saudável | Dice Fibrose (N=5) | Dice VD | Score MICCAI 2026 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 01** | Fine-tuning UKBB TF1 (4 ch) na coorte HU-UFJF (CV 10-fold) | — *(coorte local)* | — | — | — | — | — | Concluído (`v0.1`) |
| **Exp 02A** | UKBB Inicial — FCN TF1 (4 ch), sem escala, `lr=1e-5`, `bs=1`, `flip=True`, 4000 iter | 0.6227 ± 0.0883 | 0.4773 ± 0.0906 | — | — | 0.3835 ± 0.1219 | — | Concluído (`v0.2`) |
| **Exp 02B** | UKBB Corrigido — FCN TF1 (4 ch), `1.4 mm/px`, `192×192`, `lr=1e-4`, `bs=2`, `flip=False`, `rot=15°`, 4000 iter | 0.8807 ± 0.0322 | 0.7741 ± 0.0507 | — | — | 0.7611 ± 0.0566 | — | Concluído (`v0.2`) |
| **Exp 03** | U-Net ResNet34 Legado — PyTorch SMP (5 ch), ImageNet, `200×200` (pad `224×224`), `0.5 CEw + 0.5 Dice`, 70 ep, `AdamW 4e-4` | 0.8832 ± 0.0397 | 0.7950 ± 0.0583 | **0.7798 ± 0.0475** | **0.5580 ± 0.1085** *(med 0.5722)* | 0.7466 ± 0.0833 | — | Concluído (`v0.2`) |
| **Exp 04** | **nnU-Net v2 2D** — `Dataset501_CMRMULTI_LGE` (5 ch), Fold 0 (`SEED=42`), `nnUNetTrainer_100epochs` | *A executar* | *A executar* | *A executar* | *A executar* | *A executar* | *A executar* | Em preparação (`exp/nnunet-2d-cmr-multi`) |
| **Exp 05A** | **U-Net ResNet34 (4c — sem Scar)** — SMP ImageNet, `1.0 mm`, `256×256`, multi-res aug, 120 ep, TTA + SW | 0.8871 ± 0.0289 | **0.8004 ± 0.0505** *(med 0.8076)* | — | — | 0.7608 ± 0.0781 *(med 0.7999)* | Anat `0.4551` *(DSC `0.8161`, ASD `1.68 mm`)* | Concluído (`exp/unet-resnet34-cmr-multi`) |
| **Exp 05B** | **U-Net ResNet34 (5c — com Scar)** — SMP ImageNet, `1.0 mm`, `256×256`, multi-res aug, 120 ep, TTA + SW | **0.8896 ± 0.0272** | 0.7967 ± 0.0452 *(ASD `0.91 mm`)* | 0.7753 ± 0.0465 | 0.5480 ± 0.1341 *(máx **0.7307**)* | **0.7652 ± 0.0381** | **Task 2 `0.7548`** *(vPCC `0.789`, RAE `0.275`)* | Concluído (`exp/unet-resnet34-cmr-multi`) |

---

## 2. Generalização Externa *Zero-Shot* — Coorte Clínica Privada HU-UFJF (`N=10` rotulados)

Avaliação direta (sem *fine-tuning* local) de modelos treinados exclusivamente no CMR-MULTI sobre os 10 pacientes rotulados de miocardite do HU-UFJF (`1.83–2.68 mm/px`, tórax inteiro):

| Experimento | Dice VE | Dice Mio Total | Dice Mio Saudável | Dice Fibrose (N=10) | Dice VD | HD95 Mio (mm) | ASD Mio (mm) | Score MICCAI 2026 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 05A (4c — sem Scar)** | **0.8235 ± 0.0566** | 0.6470 ± 0.0893 | — | — | **0.6896 ± 0.1272** | 12.77 ± 7.11 | 2.89 ± 0.93 | Anat `0.3608` *(DSC `0.7200`)* |
| **Exp 05B (5c — com Scar)** | 0.8232 ± 0.0849 | **0.6513 ± 0.0897** | 0.6324 ± 0.0801 | 0.1286 ± 0.1439 *(máx 0.3933)* | 0.6281 ± 0.1775 | **11.51 ± 4.33** | **2.58 ± 0.62** | Task 2 `0.4105` *(vPCC `-0.044`, RAE `5.43`)* |

---

## 3. Detalhamento por Paciente de Teste — CMR-MULTI (`N=7`)

| Paciente | Fibrose? | UKBB (02B) VE / Mio / VD | UNet Legado (03) VE / Mio / Scar / VD | **Exp 05A (4c)** VE / Mio / VD | **Exp 05B (5c)** VE / Mio / Scar / VD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `LGE_SAX_005` | Não | 0.909 / 0.784 / 0.787 | 0.934 / 0.795 / — / 0.817 | 0.903 / 0.800 / 0.834 | 0.911 / 0.805 / — *(0.12 mL FP)* / 0.789 |
| `LGE_SAX_007` | Sim | 0.901 / 0.723 / 0.663 | 0.877 / 0.813 / 0.677 / 0.664 | 0.860 / 0.782 / 0.673 | 0.878 / 0.790 / **0.731** / 0.693 |
| `LGE_SAX_008` | Não | 0.924 / 0.832 / 0.837 | 0.928 / 0.839 / — / 0.821 | **0.934** / 0.843 / **0.837** | 0.933 / **0.855** / — *(0 FP)* / 0.798 |
| `LGE_SAX_014` | Sim | 0.843 / 0.703 / 0.796 | 0.818 / 0.675 / 0.382 / 0.669 | 0.851 / 0.701 / 0.800 | **0.884** / **0.720** / **0.442** / **0.805** |
| `LGE_SAX_026` | Sim | 0.872 / 0.839 / 0.777 | 0.865 / 0.855 / 0.572 / 0.810 | **0.905** / **0.855** / 0.806 | 0.899 / 0.844 / **0.650** / 0.751 |
| `LGE_SAX_028` | Sim | 0.874 / 0.765 / 0.749 | 0.890 / 0.802 / 0.602 / 0.641 | 0.884 / **0.808** / 0.645 | 0.870 / 0.784 / 0.480 / **0.760** |
| `LGE_SAX_047` | Sim | 0.842 / 0.773 / 0.721 | 0.871 / 0.786 / 0.556 / 0.803 | 0.872 / **0.815** / 0.731 | 0.852 / 0.779 / 0.438 / 0.759 |

---

## 4. Principais Conclusões Científicas
1. **Impacto da engenharia espacial no UKBB (Exp 02A → 02B):** Reamostrar para `1.4 mm/px` (`192×192`), elevar `lr` para `1e-4`, `batch=2` e remover `flip` elevou o Dice em **+41,4% (VE)**, **+62,2% (Miocárdio)** e **+98,5% (VD)**.
2. **Comparação 4 classes (Exp 05A) vs. 5 classes (Exp 05B) no CMR-MULTI:**
   - A reamostragem física (`1.0 mm`) + augmentation multi-resolução (Hoh et al., 2024) + TTA elevou a estabilidade e bateu recordes em **VE (`0.8896`)**, **Miocárdio Total (`0.8004` em 4c; `ASD = 0.91 mm` em 5c)** e **VD (`0.7652`, desvio-padrão reduzido pela metade)**, além de recuperar fortemente o outlier `LGE_SAX_014`.
   - Treinar em 4 classes sem Scar deu ganho discreto no Miocárdio Total (`0.8004` vs. `0.7967`), mas incluir Scar em 5 classes manteve anatomia equivalente e alcançou **Score Task 2 MICCAI 2026 = `0.7548`** (`vPCC = 0.7890`, `RAE = 0.2753`, pico `0.7307` em `LGE_SAX_007`, `0 FP` em `LGE_SAX_008`).
3. **Generalização *Zero-Shot* no HU-UFJF (descoberta-chave para o TCC/Mestrado):**
   - A padronização física (`1.0 mm`) + blur multi-resolução + janela deslizante gaussiana + LCC 3D permitiu segmentação anatômica *zero-shot* robusta no tórax completo do HU-UFJF (**VE `0.823`**, **Miocárdio `0.651`**, **VD `0.690`**, `ASD Mio = 2.58 mm ≈ 1 pixel nativo`).
   - Já a fibrose *zero-shot* caiu para `0.1286` (com melhor acerto nos casos mais extensos: `0.393` no `patient_7` e `0.331` no `patient_19`), evidenciando o *domain shift* patológico entre infarto extenso (CMR-MULTI) e miocardite focal/difusa de baixo volume (HU-UFJF). Isso reforça diretamente a estratégia de **Cascata em 2 etapas (Etapa 1 anatomia → Etapa 2 ROI do miocárdio) + nnU-Net v2 + adaptação/fine-tuning na coorte local**.
