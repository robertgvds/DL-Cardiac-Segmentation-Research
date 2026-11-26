# Exp 02 — Fine-Tuning UKBB (TF 1.x) no CMR-MULTI LGE SAX (`v0.2-cmr-multi-benchmark`)

## Protocolo
- **Dataset:** `robertgvds/cmr-multi-lge-sax` (47 volumes rotulados, split estratificado `35 / 5 / 7`, `SEED=42`).
- **Classes (4 canais):** `0=Fundo`, `1=VE`, `2=Miocárdio Total (Mio + Scar)`, `3=VD` (remanejado de `4 -> 3`).

## Comparação Exp 02A (Inicial) vs. Exp 02B (Corrigido)
| Configuração | Dice VE | Dice Miocárdio Total | Dice VD |
| :--- | :---: | :---: | :---: |
| **Exp 02A** (sem escala, `lr=1e-5`, `bs=1`, `flip=True`, 4000 iter) | 0.6227 ± 0.0883 | 0.4773 ± 0.0906 | 0.3835 ± 0.1219 |
| **Exp 02B** (`1.4 mm/px`, `192×192`, `lr=1e-4`, `bs=2`, `flip=False`, `rot=15°`, 4000 iter) | **0.8807 ± 0.0322** | **0.7741 ± 0.0507** | **0.7611 ± 0.0566** |

Métricas por paciente do **Exp 02B** registradas em [metrics_test_n7.csv](metrics_test_n7.csv).
