# Exp 04A — nnU-Net v2 2D `ResEncM` (4 Classes — Anatomia sem Scar) no CMR-MULTI + HU-UFJF

## Objetivo
Avaliar a **nnU-Net v2 2D com Residual Encoder M** (`nnUNetPlannerResEncM`, Isensee et al., MICCAI 2024) em **4 classes anatômicas** (`0=Fundo, 1=VE, 2=Miocárdio Total [Mio+Scar], 3=VD`) sob o mesmo protocolo do **Exp 05A (U-Net ResNet34 4c)**, testando no **CMR-MULTI (`N=7`, `SEED=42`)** e em generalização externa *zero-shot* na coorte clínica privada **HU-UFJF (`N=10` rotulados + `patient_20`)**.

## Configuração Testada (v1)
- **Dataset:** `Dataset500_CMRMULTI_LGE_4C` (Fold 0 `SEED=42`: `35 / 5 / 7`).
- **Planner / Arquitetura:** `nnUNetPlannerResEncM` (`nnUNetResEncUNetMPlans`, `2d`, `ResidualEncoderUNet`).
- **Treino:** `nnUNetTrainer_250epochs`, `nnUNet_n_proc_DA=4`, `nnUNet_compile=false`, `--npz`.
- **Kernel Kaggle:** `robertgvds/dlcs-nnunet-2d-4c` (GPU `NvidiaTeslaT4`).

## Status e Diagnóstico de Execução (v1 — Cancelado)
- **Status:** **Cancelado manualmente na época `113/250`** (~10h30 decorridas) para evitar estouro do limite de 12h do Kaggle e preservar a cota semanal restante (`1h 38m` de 30h).
- **Desempenho medido na Tesla T4:** **`~307 s/época`** (~5,1 min/época), projetando **~21,3 horas** para completar 250 épocas (contra ~4h da `PlainConvUNet` padrão no Exp 04 base).
- **Causa raiz:** O `nnUNetPlannerResEncM` escala automaticamente o `batch_size` 2D em ~4–5× e utiliza blocos residuais mais profundos (`[1,3,4,6,6,6]`), elevando fortemente o custo por época nas 4 vCPUs + Tesla T4 do Kaggle.
- **Lição aprendida / Próxima execução (após renovação da cota):** Usar `ExperimentPlanner` padrão (`PlainConvUNet`, ~55 s/época → ~1,5h em `100 épocas` ou ~3,5h em `250 épocas`) ou `ResEncM` limitado a `nnUNetTrainer_50epochs` (~4,3h).
