# Exp 04A — nnU-Net v2 2D `ResEncM` (4 Classes — Anatomia sem Scar) no CMR-MULTI + HU-UFJF

## Objetivo
Avaliar a **nnU-Net v2 2D com Residual Encoder M** (`nnUNetPlannerResEncM`, Isensee et al., MICCAI 2024) em **4 classes anatômicas** (`0=Fundo, 1=VE, 2=Miocárdio Total [Mio+Scar], 3=VD`) sob o mesmo protocolo do **Exp 05A (U-Net ResNet34 4c)**, testando no **CMR-MULTI (`N=7`, `SEED=42`)** e em generalização externa *zero-shot* na coorte clínica privada **HU-UFJF (`N=10` rotulados + `patient_20`)**.

## Configuração
- **Dataset:** `Dataset500_CMRMULTI_LGE_4C` (Fold 0 `SEED=42`: `35 / 5 / 7`).
- **Planner / Arquitetura:** `nnUNetPlannerResEncM` (`nnUNetResEncUNetMPlans`, `2d`, `ResidualEncoderUNet`).
- **Treino:** `nnUNetTrainer_250epochs`, `nnUNet_n_proc_DA=4`, `nnUNet_compile=false`, `--npz`.
- **Avaliação:** Métricas 3D brutas e pós-processadas (LCC 3D cardíaco) + métricas oficiais **MICCAI 2026**, NIfTIs `int16` e pacote reutilizável de pesos (`nnunet_model_bundle_anatomy_4c.zip`).
- **Runner Kaggle:** `kaggle_jobs/dlcs_nnunet_2d_4c/` (`robertgvds/dlcs-nnunet-2d-4c`).

## Status
*Em preparação para execução na GPU do Kaggle.*
