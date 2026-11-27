# Revisão Bibliográfica — nnU-Net v2 (`ResEnc`), Anisotropia 2D/3D e LGE-CMR (2024–2026)

> **PDF arquivado localmente (gitignored):** `docs/papers/pdfs/2024_Isensee_nnUNet_Revisited_MICCAI2024.pdf`

---

## 1. Artigo Principal — Isensee et al. (MICCAI 2024), *nnU-Net Revisited*

- **Título:** *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*
- **Autores:** Fabian Isensee, Tassilo Wald, Constantin Ulrich, Michael Baumgartner, Saikat Roy, Klaus Maier-Hein, Paul F. Jaeger
- **Publicação:** MICCAI 2024 (`arXiv:2404.09556`) — [Abstract](https://arxiv.org/abs/2404.09556) | [PDF](https://arxiv.org/pdf/2404.09556)
- **Achados Centrais (6 benchmarks, incluindo RM Cardíaca ACDC):**
  - **nnU-Net v2 continua no topo em 2024–2026:** Superou sistematicamente Vision Transformers (`SwinUNETR`, `SwinUNETRV2`, `nnFormer`, `CoTr`), Mamba (`U-Mamba Bot/Enc`) e `MONAI Auto3DSeg` (`SegResNet`, `DiNTS`).
  - **Ablação do U-Mamba:** Remover os blocos Mamba mantendo o encoder residual (`No-Mamba Base`) alcançou desempenho igual ou superior (`91.89%` vs. `91.79%` no ACDC), provando que os ganhos vinham do **encoder residual dentro da nnU-Net**, não do Mamba.
  - **Papel do `nnUNetPlannerResEncM` (Residual Encoder M):** Introduzido para substituir o baseline antigo de 2018 (`PlainConvUNet`). Requer ~`9.1 GB` VRAM (encaixa com folga na **Tesla T4 de 16 GB** do Kaggle), superou a nnU-Net padrão em todos os 6 datasets e atingiu o melhor resultado entre todos os `ResEnc` em RM cardíaca (**ACDC: `91.99%`** vs. `91.54%` da padrão). Variantes `ResEnc L/XL` exigem 22–36 GB VRAM e trazem ganho apenas em bases muito maiores (`n > 350`).

---

## 2. Evidências Recentes em LGE-CMR / Cicatriz Miocárdica (2024–2026)

1. **Yang et al. (2026) — *LGE-CMRnet* (*JCMR*, 28(1):102712, DOI `10.1016/j.jocmr.2026.102712`):**
   - 567 pacientes IAM treinados em Siemens 1.5T/3T (**espessura Z `8–8.5 mm`**) e validados **externamente em 158 pacientes** GE 3T (**`8 mm`**).
   - Utilizou **2D nnU-Net** após ROI ventricular, atingindo **Scar DSC `0.83 ± 0.11`** (`r = 0.90`) e **MVO DSC `0.88 ± 0.11`** (`r = 0.98`) na coorte externa cross-vendor — confirmando que **2D nnU-Net** evita artefatos de interpolação em Z quando aplicado *zero-shot* a scanners com espessura de fatia diferente (caso exato de CMR-MULTI → HU-UFJF).
2. **Bernard et al. (2026) — *The MYOSAIQ Challenge* (*MELBA* 2026:032, `arXiv:2608.29246`):**
   - 439 volumes LGE (16 centros, 3 fabricantes): `nnU-Net` (`MI DSC 0.669`), `Cascaded U-Net` (`0.700`) e `MedNeXt` (`0.688`) superaram largamente foundation models automáticos (`mcMedSAM-auto`: `0.544`).
3. **CMR-MULTI / CMRSeg 2026 (MICCAI 2026) e Amirrajab et al. (`arXiv:2608.07291`):**
   - A maioria das soluções orais de destaque do challenge baseia-se em **nnU-Net v2 por vista / ensembles estruturados** ou híbridos com refinamento de cicatriz.
4. **Ramzan et al. (ISBI 2026, `arXiv:2511.14702`) e On et al. (MICCAI STACOM / 2025, `arXiv:2504.02668`):**
   - Confirmam que `250 épocas` é o ponto ideal de custo-benefício em bases pequenas (`~40 casos`) na Tesla T4 (`100 épocas` tende a subajustar bordas finas de fibrose), e que pós-processamento anatômico (maior componente conexo + restrição de micro-ruído de fibrose) + futura **cascata em 2 etapas** produzem ganhos expressivos de precisão clínica.

---

## 3. Decisão Arquitetural Adotada (Exp 04A e Exp 04B)

- **Arquitetura / Planner:** `nnU-Net v2` **2D** com **`nnUNetPlannerResEncM`** (`nnUNetResEncUNetMPlans` — encoder residual moderno de Isensee et al., MICCAI 2024).
- **Orçamento na Tesla T4:** **`nnUNetTrainer_250epochs`** com **`nnUNet_n_proc_DA=4`** (evita *thrashing* nas 4 vCPUs do Kaggle) e **`nnUNet_compile=false`** (elimina lentidão na Turing `sm_75`), reduzindo o tempo de ~4h para ~1,8–2,5h por job.
- **Paridade completa com Exp 05A/05B (U-Net ResNet34):**
  - **Exp 04A (`Dataset500_CMRMULTI_LGE_4C` — 4 classes sem Scar):** `0=Bg, 1=VE, 2=Mio Total (Mio+Scar), 3=VD`.
  - **Exp 04B (`Dataset501_CMRMULTI_LGE` — 5 classes com Scar):** `0=Bg, 1=VE, 2=Mio Saudável, 3=Scar, 4=VD`.
  - Ambos avaliados em **CMR-MULTI Test (`N=7`, `SEED=42`)** e **HU-UFJF Private Cohort (`N=10` rotulados padronizados + `patient_20`)**, com métricas **brutas e pós-processadas (LCC 3D + limpeza de micro-scar)**, **Métricas Oficiais MICCAI 2026**, ZIPs de NIfTIs (`int16`, affine/pixdim preservados) e **pacote completo de pesos (`checkpoint_best.pth` + `checkpoint_final.pth` + `plans.json` + `dataset.json`)** para inferência futura sem retreinamento!
