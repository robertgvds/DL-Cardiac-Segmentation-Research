# Log Vivo de Experimentos — Segmentação Cardíaca e de Fibrose em LGE-CMR

Todos os experimentos no CMR-MULTI utilizam rigorosamente o **mesmo split estratificado por paciente (`SEED=42`)** para comparação justa:
- **Treino (35 volumes — 26 com fibrose, 9 sem):** `LGE_SAX_001, 002, 004, 006, 009, 012, 013, 015, 016, 017, 018, 019, 020, 021, 022, 023, 024, 025, 027, 029, 030, 031, 033, 035, 036, 037, 038, 039, 040, 041, 042, 043, 044, 045, 046`
- **Validação (5 volumes — 4 com fibrose, 1 sem):** `LGE_SAX_003, 010, 011, 032, 034`
- **Teste Inédito (7 volumes — 5 com fibrose, 2 sem):** `LGE_SAX_005, 007, 008, 014, 026, 028, 047`

---

## 1. Resumo Consolidado (CMR-MULTI LGE SAX — Teste Inédito N=7)

| Experimento | Arquitetura / Configuração | Dice VE | Dice Miocárdio Total | Dice Mio Saudável | Dice Fibrose (N=5) | Dice VD | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 01** | Fine-tuning UKBB TF1 (4 ch) na coorte HU-UFJF (CV 10-fold) | — *(coorte local)* | — | — | — | — | Concluído (`v0.1`) |
| **Exp 02A** | UKBB Inicial — FCN TF1 (4 ch), sem escala, `lr=1e-5`, `bs=1`, `flip=True`, 4000 iter | 0.6227 ± 0.0883 | 0.4773 ± 0.0906 | — | — | 0.3835 ± 0.1219 | Concluído (`v0.2`) |
| **Exp 02B** | UKBB Corrigido — FCN TF1 (4 ch), `1.4 mm/px`, `192×192`, `lr=1e-4`, `bs=2`, `flip=False`, `rot=15°`, 4000 iter | **0.8807 ± 0.0322** | 0.7741 ± 0.0507 | — | — | **0.7611 ± 0.0566** | Concluído (`v0.2`) |
| **Exp 03** | U-Net ResNet34 — PyTorch SMP (5 ch), ImageNet, `200×200` (pad `224×224`), `0.5 CEw + 0.5 Dice`, 70 ep, `AdamW 4e-4` | **0.8832 ± 0.0397** | **0.7950 ± 0.0583** | 0.7798 ± 0.0475 | **0.5580 ± 0.1085** *(med 0.5722)* | 0.7466 ± 0.0833 | Concluído (`v0.2`) |
| **Exp 04** | **nnU-Net v2 2D** — `Dataset501_CMRMULTI_LGE` (5 ch), Fold 0 (`SEED=42`), `nnUNetTrainer_100epochs` | *A executar* | *A executar* | *A executar* | *A executar* | *A executar* | Em preparação (`exp/nnunet-2d-cmr-multi`) |
| **Exp 05A** | **U-Net ResNet34 (4c — sem Scar)** — SMP ImageNet, `1.0 mm`, `256×256`, multi-res aug, 120 ep, TTA + SW + teste externo HU-UFJF | *A executar* | *A executar* | — | — | *A executar* | Em preparação (`exp/unet-resnet34-cmr-multi`) |
| **Exp 05B** | **U-Net ResNet34 (5c — com Scar)** — SMP ImageNet, `1.0 mm`, `256×256`, multi-res aug, 120 ep, TTA + SW + teste externo HU-UFJF | *A executar* | *A executar* | *A executar* | *A executar* | *A executar* | Em preparação (`exp/unet-resnet34-cmr-multi`) |

> **Métricas Oficiais MICCAI 2026 (CMR-MULTI) incorporadas a partir do Exp 05:** Além de Dice 3D (GT+) e HD95 (mm), `src/evaluation/metrics_3d.py` calcula `Board DSC`, `HD máximo (mm)`, `ASD (mm)`, `Massa de Fibrose (1.05 g/mL)`, `vPCC`, `RAE` e `Score Task 2 = 0.7 × DSC_board + 0.3 × (0.5 × max(0, vPCC) + 0.5 / (1 + RAE))`, além da avaliação externa na coorte privada **HU-UFJF** (`N=10` rotulados + `patient_20`).

---

## 2. Detalhamento por Paciente de Teste (Exp 02B vs. Exp 03)

| Paciente | Tem Fibrose? | UKBB VE | UKBB Mio | UKBB VD | UNet VE | UNet Mio Total | UNet Mio Saudável | UNet Fibrose | UNet VD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LGE_SAX_005` | Não | 0.9094 | 0.7836 | 0.7865 | 0.9340 | 0.7952 | 0.7963 | — *(FP mínimo)* | 0.8174 |
| `LGE_SAX_007` | Sim | 0.9012 | 0.7227 | 0.6625 | 0.8773 | 0.8127 | 0.7861 | **0.6766** | 0.6645 |
| `LGE_SAX_008` | Não | 0.9242 | 0.8324 | 0.8366 | 0.9276 | 0.8394 | 0.8394 | — *(0 FP)* | 0.8211 |
| `LGE_SAX_014` | Sim | 0.8430 | 0.7034 | 0.7956 | 0.8175 | 0.6748 | 0.7771 | 0.3823 *(outlier)* | 0.6693 |
| `LGE_SAX_026` | Sim | 0.8722 | 0.8391 | 0.7766 | 0.8645 | 0.8545 | 0.8059 | 0.5722 | 0.8097 |
| `LGE_SAX_028` | Sim | 0.8735 | 0.7651 | 0.7488 | 0.8903 | 0.8023 | 0.7679 | 0.6023 | 0.6412 |
| `LGE_SAX_047` | Sim | 0.8415 | 0.7725 | 0.7213 | 0.8712 | 0.7861 | 0.6858 | 0.5564 | 0.8034 |

---

## 3. Principais Conclusões dos Baselines
1. **Impacto da engenharia espacial no UKBB:** Reamostrar para `1.4 mm/px` (`192×192`), elevar `lr` para `1e-4`, `batch=2` e remover `flip` elevou o Dice em **+41,4% (VE)**, **+62,2% (Miocárdio)** e **+98,5% (VD)**.
2. **Desempenho da U-Net ResNet34 em 5 classes:** Empatou no VE, venceu em 6/7 casos no Miocárdio Total, manteve falsos positivos nulos/desprezíveis em saudáveis (`005` e `008`) e atingiu média de **0.6019** na fibrose nos 4 casos sem o outlier `LGE_SAX_014` (máximo **0.6766**).
3. **Justificativa para nnU-Net v2 e Cascata:** O caso `LGE_SAX_014` (Dice Mio 0.6748 / Fibrose 0.3823) evidencia o limite de modelos 2D diretos com poucos volumes, motivando a migração oficial para **nnU-Net v2** (2D → 3D Fullres → Cascata Anatomia + Fibrose).
