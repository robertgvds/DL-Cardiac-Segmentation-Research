# Exp 04B — nnU-Net v2 2D `ResEncM` (5 Classes — Anatomia + Scar) no CMR-MULTI + HU-UFJF

## Objetivo
Avaliar a **nnU-Net v2 2D com Residual Encoder M** (`nnUNetPlannerResEncM`, Isensee et al., MICCAI 2024) nas **5 classes completas** (`0=Fundo, 1=VE, 2=Mio Saudável, 3=Scar, 4=VD`) sob o mesmo protocolo do **Exp 05B (U-Net ResNet34 5c)**, testando no **CMR-MULTI (`N=7`, `SEED=42`)** e em generalização externa *zero-shot* na coorte clínica privada **HU-UFJF (`N=10` rotulados + `patient_20`)**, preservando os pesos treinados em pacote reutilizável.

## Configuração Testada (v1)
- **Dataset:** `Dataset501_CMRMULTI_LGE` (Fold 0 `SEED=42`: `35 / 5 / 7`).
- **Planner / Arquitetura:** `nnUNetPlannerResEncM` (`nnUNetResEncUNetMPlans`, `2d`, `ResidualEncoderUNet`).
- **Treino:** `nnUNetTrainer_250epochs`, `nnUNet_n_proc_DA=4`, `nnUNet_compile=false`, `--npz`.
- **Kernel Kaggle:** `robertgvds/dlcs-nnunet-2d-5c` (GPU `NvidiaTeslaT4`).

## Status e Diagnóstico de Execução (v1 — Cancelado)
- **Status:** **Cancelado manualmente na época `106/250`** (~10h30 decorridas) para evitar estouro do limite de 12h do Kaggle e preservar a cota semanal restante (`1h 38m` de 30h).
- **Desempenho medido na Tesla T4:** **`~312 s/época`** (~5,2 min/época), projetando **~21,7 horas** para completar 250 épocas.
- **Causa raiz:** Mesma do Exp 04A (`batch_size` 2D auto-escalado ~4–5× + blocos residuais profundos do `ResEncM` sob gargalo de 4 vCPUs + Tesla T4).
- **Lição aprendida / Próxima execução (após renovação da cota):** Usar `ExperimentPlanner` padrão (`PlainConvUNet`, ~55 s/época → ~1,5h em `100 épocas` ou ~3,5h em `250 épocas`) ou `ResEncM` limitado a `nnUNetTrainer_50epochs` (~4,3h).
