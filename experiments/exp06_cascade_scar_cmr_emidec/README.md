# Exp 06 — Cascata em 2 Etapas (Anatomia Exp 05A → Especialista em Fibrose CMR-MULTI + EMIDEC)

## Objetivo
Aplicar a **Regra de Ouro da literatura LGE-CMR (Cascata em 2 Etapas)** e desbloquear os **100 volumes rotulados do EMIDEC** (que não possuem rótulo de VD) para treinar um especialista de fibrose em alta resolução na ROI do miocárdio (**135 volumes LGE reais**: 35 CMR-MULTI + 100 EMIDEC).

## Arquitetura
- **Etapa 1 (Anatomia 4c):** Reaproveita diretamente o `checkpoint_best.pth` do **Exp 05A (`dlcs-unet-resnet34-4c`)** para localizar VE, Miocárdio Total e VD.
- **Etapa 2 (Especialista em Fibrose na ROI):**
  - `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=3)`
  - Recorte centrado no miocárdio em alta resolução **`0.75 × 0.75 mm`** (`160 × 160` px = `120 × 120 mm`).
  - **Normalização robusta local relativa ao miocárdio (`mediana / MAD`)** — análoga ao critério clínico `n-SD`, eliminando *domain shift* de contraste entre CMR-MULTI, EMIDEC e HU-UFJF.
  - Entrada 3 canais `(LGE_local, Prior_Mio, Prior_VE)` com perturbação morfológica aleatória dos *priors* no treino + simulação multi-resolução (Hoh et al., 2024).
  - Calibração do limiar de probabilidade na validação (`N=5`, `SEED=42`), fusão 5 classes restrita à ROI do miocárdio (+2 px) e exportação do mapa probabilístico contínuo `*_SCAR_PROB.nii.gz`.
- **Runner Kaggle:** `kaggle_jobs/dlcs_cascade_scar_emidec/` (`robertgvds/dlcs-cascade-scar-emidec`).

## Status
*Em preparação para execução rápida (~10–13 min) na GPU do Kaggle.*
