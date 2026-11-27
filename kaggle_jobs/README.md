# Kaggle GPU Jobs (`kaggle_jobs/`)

Runners finos executados na GPU do Kaggle via CLI (`kaggle kernels push`). Toda a lógica científica permanece versionada em `src/`.

| Pasta / Slug | Experimento | Branch Alvo | Datasets Montados (Private) | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `dlcs_nnunet_2d` (`robertgvds/dlcs-nnunet-2d`) | **Exp 04 (base)** | `exp/nnunet-2d-cmr-multi` | `cmr-multi-lge-sax` | nnU-Net v2 2D `PlainConvUNet` (5 classes, benchmark concluído: `Task 2 = 0.7682`) |
| `dlcs_nnunet_2d_4c` (`robertgvds/dlcs-nnunet-2d-4c`) | **Exp 04A** | `exp/nnunet-2d-cmr-multi` | `cmr-multi-lge-sax`, `hu-ufjf-lge-sax` | nnU-Net v2 2D `ResEncM` (4c: VE, MioTotal, VD) + HU-UFJF + pacote de pesos *(ajustar épocas/planner na T4)* |
| `dlcs_nnunet_2d_5c` (`robertgvds/dlcs-nnunet-2d-5c`) | **Exp 04B** | `exp/nnunet-2d-cmr-multi` | `cmr-multi-lge-sax`, `hu-ufjf-lge-sax` | nnU-Net v2 2D `ResEncM` (5c: VE, MioSaudável, Scar, VD) + HU-UFJF + pacote de pesos *(ajustar épocas/planner na T4)* |
| `dlcs_unet_resnet34_4c` (`robertgvds/dlcs-unet-resnet34-4c`) | **Exp 05A** | `exp/unet-resnet34-cmr-multi` | `cmr-multi-lge-sax`, `hu-ufjf-lge-sax` | U-Net ResNet34 (4c: VE, MioTotal [Mio+Scar], VD) + teste externo HU-UFJF *(concluído)* |
| `dlcs_unet_resnet34_5c` (`robertgvds/dlcs-unet-resnet34-5c`) | **Exp 05B** | `exp/unet-resnet34-cmr-multi` | `cmr-multi-lge-sax`, `hu-ufjf-lge-sax` | U-Net ResNet34 (5c: VE, MioSaudável, Scar, VD) + teste externo HU-UFJF *(concluído)* |
