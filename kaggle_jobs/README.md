# Kaggle GPU Jobs (`kaggle_jobs/`)

Runners finos executados na GPU do Kaggle via CLI (`kaggle kernels push`). Toda a lógica científica permanece versionada em `src/`.

| Pasta / Slug | Experimento | Branch Alvo | Datasets Montados (Private) | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `dlcs_nnunet_2d` (`robertgvds/dlcs-nnunet-2d`) | **Exp 04 (base)** | `exp/nnunet-2d-cmr-multi` | `cmr-multi-lge-sax` | nnU-Net v2 2D `PlainConvUNet` (5 classes, benchmark inicial concluído) |
| `dlcs_nnunet_2d_4c` (`robertgvds/dlcs-nnunet-2d-4c`) | **Exp 04A** | `exp/nnunet-2d-cmr-multi` | `cmr-multi-lge-sax`, `hu-ufjf-lge-sax` | nnU-Net v2 2D `ResEncM` (4 classes: VE, Mio Total [Mio+Scar], VD) + teste externo HU-UFJF + pesos |
| `dlcs_nnunet_2d_5c` (`robertgvds/dlcs-nnunet-2d-5c`) | **Exp 04B** | `exp/nnunet-2d-cmr-multi` | `cmr-multi-lge-sax`, `hu-ufjf-lge-sax` | nnU-Net v2 2D `ResEncM` (5 classes: VE, Mio Saudável, Scar, VD) + teste externo HU-UFJF + pesos |
