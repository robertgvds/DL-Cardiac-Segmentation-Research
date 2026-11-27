# Exp 04B — nnU-Net v2 2D `ResEncM` (5 Classes — Anatomia + Scar) no CMR-MULTI + HU-UFJF

## Objetivo
Avaliar a **nnU-Net v2 2D com Residual Encoder M** (`nnUNetPlannerResEncM`, Isensee et al., MICCAI 2024) nas **5 classes completas** (`0=Fundo, 1=VE, 2=Mio Saudável, 3=Scar, 4=VD`) sob o mesmo protocolo do **Exp 05B (U-Net ResNet34 5c)**, testando no **CMR-MULTI (`N=7`, `SEED=42`)** e em generalização externa *zero-shot* na coorte clínica privada **HU-UFJF (`N=10` rotulados + `patient_20`)**, além de preservar os pesos treinados em pacote reutilizável para inferência futura.

## Configuração
- **Dataset:** `Dataset501_CMRMULTI_LGE` (Fold 0 `SEED=42`: `35 / 5 / 7`).
- **Planner / Arquitetura:** `nnUNetPlannerResEncM` (`nnUNetResEncUNetMPlans`, `2d`, `ResidualEncoderUNet`).
- **Treino:** `nnUNetTrainer_250epochs`, `nnUNet_n_proc_DA=4`, `nnUNet_compile=false`, `--npz`.
- **Avaliação:** Métricas 3D brutas e pós-processadas (LCC 3D cardíaco + limpeza de micro-ruído de fibrose) + métricas oficiais **MICCAI 2026** (`Task 2 Score`, `vPCC`, `RAE`), NIfTIs `int16` e pacote reutilizável de pesos (`nnunet_model_bundle_full_5c.zip`).
- **Runner Kaggle:** `kaggle_jobs/dlcs_nnunet_2d_5c/` (`robertgvds/dlcs-nnunet-2d-5c`).

## Status
*Em preparação para execução na GPU do Kaggle.*
