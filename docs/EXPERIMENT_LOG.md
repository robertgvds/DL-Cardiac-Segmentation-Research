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
| **Exp 03** | U-Net ResNet34 — PyTorch SMP (5 ch), ImageNet, `200×200` (pad `224×224`), `0.5 CEw + 0.5 Dice`, 70 ep, `AdamW 4e-4` | 0.8832 ± 0.0397 | 0.7950 ± 0.0583 | 0.7798 ± 0.0475 | 0.5580 ± 0.1085 *(med 0.5722)* | 0.7466 ± 0.0833 | — | Concluído (`v0.2`) |
| **Exp 04** | **nnU-Net v2 2D** — `Dataset501_CMRMULTI_LGE` (5 ch), Fold 0 (`SEED=42`), `nnUNetTrainer_250epochs` | **0.9008 ± 0.0183** | **0.8085 ± 0.0331** | **0.7959 ± 0.0258** | **0.5846 ± 0.0907** *(med **0.6316**)* | **0.7673 ± 0.0541** | **Task 2 `0.7682`** *(vPCC `0.9585`, RAE `0.2482`)* | Concluído (`exp/nnunet-2d-cmr-multi`) |
| **Exp 04A (v1)** | **nnU-Net v2 2D `ResEncM` (4c — sem Scar)** — `Dataset500`, `nnUNetTrainer_250epochs`, `DA=4` | — | — | — | — | — | — | Cancelado na ep. 113/250 (`307 s/ep`, proj. ~21,3h > teto 12h Kaggle) |
| **Exp 04B (v1)** | **nnU-Net v2 2D `ResEncM` (5c — com Scar)** — `Dataset501`, `nnUNetTrainer_250epochs`, `DA=4` | — | — | — | — | — | — | Cancelado na ep. 106/250 (`312 s/ep`, proj. ~21,7h > teto 12h Kaggle) |

---

## 2. Detalhamento por Paciente de Teste (Exp 02B vs. Exp 03 vs. Exp 04)

| Paciente | Fibrose? | UKBB (02B) VE / Mio / VD | UNet ResNet34 (03) VE / Mio / Scar / VD | **nnU-Net v2 2D (04)** VE / Mio Total / Mio Saud. / Fibrose / VD | HD95 Mio (04) | Erro Vol. Fibrose (04) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `LGE_SAX_005` | Não | 0.909 / 0.784 / 0.787 | 0.934 / 0.795 / — / 0.817 | 0.918 / 0.807 / 0.811 / — *(0.82 mL FP)* / 0.763 | 2.27 mm | 0.82 mL (205 vox) |
| `LGE_SAX_007` | Sim | 0.901 / 0.723 / 0.663 | 0.877 / 0.813 / 0.677 / 0.664 | **0.913** / 0.812 / 0.771 / 0.665 / 0.680 | 3.03 mm | 4.42 mL (`20.0%`) |
| `LGE_SAX_008` | Não | 0.924 / 0.832 / 0.837 | 0.928 / 0.839 / — / 0.821 | 0.925 / 0.836 / 0.836 / — *(0.02 mL FP)* / 0.825 | 2.87 mm | 0.02 mL (5 vox) |
| `LGE_SAX_014` | Sim | 0.843 / 0.703 / 0.796 | 0.818 / 0.675 / 0.382 / 0.669 | **0.899** / **0.758** / **0.792** / **0.522** / **0.824** | 11.06 mm | 16.17 mL (`43.7%`) |
| `LGE_SAX_026` | Sim | 0.872 / 0.839 / 0.777 | 0.865 / 0.855 / 0.572 / 0.810 | **0.884** / 0.854 / 0.806 / **0.647** / 0.769 | 3.02 mm | 5.92 mL (`15.2%`) |
| `LGE_SAX_028` | Sim | 0.874 / 0.765 / 0.749 | 0.890 / 0.802 / 0.602 / 0.641 | 0.890 / 0.776 / **0.797** / 0.457 / 0.716 | 5.22 mm | **2.76 mL (`7.3%`)** |
| `LGE_SAX_047` | Sim | 0.842 / 0.773 / 0.721 | 0.871 / 0.786 / 0.556 / 0.803 | **0.877** / **0.815** / **0.758** / **0.632** / 0.793 | 4.99 mm | 29.03 mL (`37.9%`) |

---

## 3. Principais Conclusões
1. **Impacto da engenharia espacial no UKBB (Exp 02A → 02B):** Reamostrar para `1.4 mm/px` (`192×192`), elevar `lr` para `1e-4`, `batch=2` e remover `flip` elevou o Dice em **+41,4% (VE)**, **+62,2% (Miocárdio)** e **+98,5% (VD)**.
2. **Saltos da nnU-Net v2 2D (Exp 04):** Superou **todos os baselines** em todas as 5 métricas (`VE = 0.9008`, `Mio Total = 0.8085`, `Mio Saudável = 0.7959`, `Fibrose = 0.5846` com mediana `0.6316`, `VD = 0.7673`), reduziu fortemente a variância inter-paciente, recuperou o outlier `LGE_SAX_014` (`+14,0 p.p.` na fibrose sobre Exp 03) e atingiu correlação volumétrica de massa de fibrose **`vPCC = 0.9585`** (`RAE = 24.82%`, **Task 2 Score MICCAI 2026 = `0.7682`**).
3. **Próximos passos motivados pelos resultados:** Os pequenos falsos positivos de fibrose nos casos saudáveis (`5 voxels` em `008` e `205 voxels` em `005`) penalizam o `Board DSC` estrito do challenge (`0.7204`), motivando: (a) **exportação de `checkpoint_best.pth` + avaliação externa no HU-UFJF**, (b) **nnU-Net v2 3D Fullres** e (c) **Cascata em 2 etapas (Anatomia → ROI do Miocárdio para Fibrose)**.
