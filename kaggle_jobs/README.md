# Kaggle GPU Runners (`kaggle_jobs/`)

Cada subpasta contém apenas um **runner fino** (`kernel-metadata.json` + `run_job.py`) que clona a branch do experimento em `https://github.com/robertgvds/DL-Cardiac-Segmentation-Research.git`, instala `nnunetv2` e executa os módulos de `src/` na GPU do Kaggle:

- `dlcs_nnunet_2d/`: Baseline nnU-Net v2 2D em `Dataset501_CMRMULTI_LGE` (Fase 4).
- `dlcs_nnunet_3d/`: Evolução 3D Fullres (próxima etapa).
- `dlcs_nnunet_cascade/`: Cascata em 2 etapas — Anatomia → Fibrose (próxima etapa).
