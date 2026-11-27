# Kaggle GPU Jobs (`kaggle_jobs/`)

Runners finos executados na GPU do Kaggle via CLI (`kaggle kernels push`). Toda a lógica científica permanece em `src/`.

| Pasta / Slug | Experimento | Branch Alvo | Datasets Montados (Private) | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `dlcs_unet_resnet34_4c` (`robertgvds/dlcs-unet-resnet34-4c`) | **Exp 05A** | `exp/unet-resnet34-cmr-multi` | `cmr-multi-lge-sax`, `hu-ufjf-lge-sax` | U-Net ResNet34 (4 classes: VE, Mio Total [Mio+Scar], VD) + teste externo HU-UFJF |
| `dlcs_unet_resnet34_5c` (`robertgvds/dlcs-unet-resnet34-5c`) | **Exp 05B** | `exp/unet-resnet34-cmr-multi` | `cmr-multi-lge-sax`, `hu-ufjf-lge-sax` | U-Net ResNet34 (5 classes: VE, Mio Saudável, Scar, VD) + teste externo HU-UFJF |

> **Concorrência Kaggle:** Dispare sempre **um kernel de GPU por vez** após confirmar que a branch foi enviada ao GitHub (`git push origin <branch>`).
