# DL-Cardiac-Segmentation-Research

Framework reprodutível em **PyTorch + nnU-Net v2** para segmentação automática 2D, 3D e em cascata de câmaras cardíacas (VE, Miocárdio, VD) e fibrose/cicatriz miocárdica em **LGE-CMR** (Ressonância Magnética Cardíaca com Realce Tardio), desenvolvido na Universidade Federal de Juiz de Fora (UFJF).

## Visão Geral da Arquitetura

- **Código e métricas leves no GitHub:** Toda a lógica de dados (`src/data/`), avaliação 3D (`src/evaluation/`), pós-processamento (`src/postprocessing/`), runners Kaggle (`kaggle_jobs/`) e resultados resumidos (`experiments/`, `docs/EXPERIMENT_LOG.md`).
- **Datasets pesados exclusivamente no Kaggle (Private):** Montados diretamente em `/kaggle/input/` durante execuções em GPU via Kaggle CLI, mantendo o repositório Git ultraleve (`< 2 MB`).
- **Execução controlada pelo PC (WSL2) → GPU Kaggle:** Testes rápidos locais com `--dry-run` em `data_sample/` (2 volumes), versionamento manual na branch do experimento e disparo remoto via `kaggle kernels push`.

## Datasets Oficiais (Kaggle Private — `robertgvds`)

| Slug Kaggle | Conteúdo | Classes |
| :--- | :--- | :--- |
| `robertgvds/cmr-multi-lge-sax` | 40 `SAX_TR` + 7 `SAX_VAL` + 13 `SAX_TST` + planilhas `.xlsx` | `0=Bg, 1=VE, 2=Mio Saudável, 3=Fibrose, 4=VD` |
| `robertgvds/emidec-lge-sax` | 100 `train` + 50 `test` desaninhados | `0=Bg, 1=VE, 2=Mio Saudável, 3=Scar, 4=MVO` *(sem VD)* |
| `robertgvds/hu-ufjf-lge-sax` | 10 pacientes rotulados + `patient_20` (NIfTI RAS puro) + `.mat` anonimizados | `seg_anat` (4c), `seg_scar` (bin), `seg_5class` (padrão CMR-MULTI) |
| `robertgvds/hu-ufjf-dicom-archive` | Backup estruturado dos DICOMs anonimizados (11 pacientes) | — |

Detalhes completos de rótulos e regras anti-leakage em [docs/DATASETS.md](docs/DATASETS.md).

## Baselines Consolidados (CMR-MULTI LGE SAX, Teste Inédito N=7, `SEED=42`)

| Experimento | Dice VE | Dice Mio Total | Dice Fibrose (N=5) | Dice VD |
| :--- | :---: | :---: | :---: | :---: |
| **Exp 02A — UKBB Inicial (TF1)** | 0.6227 ± 0.0883 | 0.4773 ± 0.0906 | — | 0.3835 ± 0.1219 |
| **Exp 02B — UKBB Corrigido (1.4 mm/px, TF1)** | **0.8807 ± 0.0322** | 0.7741 ± 0.0507 | — | **0.7611 ± 0.0566** |
| **Exp 03 — U-Net ResNet34 (PyTorch, 5 ch)** | **0.8832 ± 0.0397** | **0.7950 ± 0.0583** | **0.5580 ± 0.1085** | 0.7466 ± 0.0833 |

Histórico completo por paciente em [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

## Setup Rápido (WSL2 Ubuntu 24.04)

```bash
python3 -m venv ~/venvs/dlcs
source ~/venvs/dlcs/bin/activate
pip install -e .
```

Para credenciais do Kaggle CLI (nunca versionar):
```bash
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
# Salvar token em ~/.kaggle/access_token (chmod 600)
```

## Estrutura do Repositório

```text
DL-Cardiac-Segmentation-Research/
├── docs/              # Dicionário de datasets, log de experimentos e notas de artigos
├── scripts/           # Automação de empacotamento e upload privado para o Kaggle
├── src/               # Pipeline modular POO (data, evaluation, postprocessing)
├── kaggle_jobs/       # Runners finos executados na GPU do Kaggle via CLI
├── experiments/       # Apenas artefatos leves (CSVs, gráficos, configs, resumos)
└── legacy/            # Código histórico limpo (UKBB TF1, Elastix, baseline SMP)
```
