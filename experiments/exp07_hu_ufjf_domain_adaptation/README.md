# Exp 07 — Adaptação de Domínio na Coorte Clínica HU-UFJF (CV 5-Fold: Fine-Tuning vs. Co-Fine-Tuning)

## Objetivo
Comparar diretamente, em **Validação Cruzada 5-Fold estratificada por volume de fibrose (`N=10` pacientes OOF, zero *data leakage*)** na coorte de miocardite do **HU-UFJF**, o salto de desempenho sobre o *zero-shot* (**Exp 05B**: `VE=0.823`, `Mio=0.651`, `Scar=0.129`, `VD=0.628`) usando duas estratégias a partir dos pesos do **Exp 05B**:
1. **Exp 07A (`finetune_hu_only`):** *Fine-tuning* puro nos 8 pacientes HU-UFJF de treino de cada fold (`60 épocas`, LR diferencial `encoder=3e-5` / `decoder=1.5e-4`, patches `256×256` centrados no coração com *jitter* ±36 px).
2. **Exp 07B (`co_finetune_hu_cmr`):** *Co-fine-tuning* regularizado combinando os 8 pacientes HU-UFJF (*oversampling* 4×) + os 35 pacientes de treino do CMR-MULTI (`45 épocas`), avaliando também a retenção no CMR-MULTI Test (`N=7`) para medir *catastrophic forgetting*.

## Runner Kaggle
- `kaggle_jobs/dlcs_hu_domain_adaptation/` (`robertgvds/dlcs-hu-domain-adaptation`, ~14–16 min na Tesla T4).

## Status
*Em preparação para execução na GPU do Kaggle.*
