# Dicionário Oficial de Datasets — LGE-CMR

Todos os datasets pesados residem exclusivamente como **Private Datasets** na Collection **`DL-Cardiac-Segmentation`** sob a conta Kaggle `robertgvds`. Nenhum volume médico ou dado hospitalar é versionado neste repositório Git.

> **Atenção Técnica (Kaggle Input):** Ao montar datasets em `/kaggle/input/<slug>/`, o Kaggle descompacta automaticamente arquivos `.nii.gz` para `.nii`. Todos os scripts em `src/data/` devem aceitar transparentemente ambas as extensões.

---

## 1. `robertgvds/cmr-multi-lge-sax` (Private — 110 arquivos)
- **Modalidade e vista:** LGE-CMR em eixo curto ventricular (`SAX`). Vistas `2CH`, `4CH`, `RAS` e todo `CINE_MULTI` foram removidos.
- **Composição:**
  - `SAX_TR/{image, anno}`: **40 volumes** (`LGE_SAX_001`..`040`, sendo 30 com fibrose e 10 sem).
  - `SAX_VAL/{image, anno}`: **7 volumes** (`LGE_SAX_041`..`047`, sendo 5 com fibrose e 2 sem).
  - `SAX_TST/image`: **13 volumes** sem máscara pública (`LGE_SAX_048`..`060`).
  - Planilhas: `dataset_lge.xlsx`, `dataset_lge_valid.xlsx`, `dataset_lge_test.xlsx`.
- **Geometria:** Pré-recortado em `200×200`, `pixdim` XY `0.49–1.26 mm`, espessura Z `8–10 mm`, 3 fabricantes de scanner.
- **Mapa de Rótulos (5 classes):**
  - `0` = Fundo
  - `1` = Cavidade do Ventrículo Esquerdo (VE / LV)
  - `2` = Miocárdio Saudável do VE (LV Myo)
  - `3` = Fibrose / Cicatriz Miocárdica (Scar)
  - `4` = Cavidade do Ventrículo Direito (VD / RV)

---

## 2. `robertgvds/emidec-lge-sax` (Private — 250 arquivos)
- **Composição (desaninhada e limpa):**
  - `train/Case_XXXX/{image, mask}.nii[.gz]`: **100 volumes** (33 Normais `Case_N*` + 67 Patológicos `Case_P*`).
  - `test/Case_XXXX/image.nii[.gz]`: **50 volumes** (`Case_101`..`150`, sem máscara pública).
- **Mapa de Rótulos (5 classes — SEM VD):**
  - `0` = Fundo
  - `1` = Cavidade do VE
  - `2` = Miocárdio Saudável
  - `3` = Infarto / Cicatriz (Scar)
  - `4` = Obstrução Microvascular (MVO)
- **Alerta Crítico de Harmonização:** O label `4` no EMIDEC é **MVO**, enquanto no CMR-MULTI é **VD**. Para treino conjunto (`Dataset510_LGE_UNIFIED_4C`), mapear `3+4 -> 3` no EMIDEC e `4 -> 0` no CMR-MULTI.

---

## 3. `robertgvds/hu-ufjf-lge-sax` (Private — Sigilo Hospitalar — 53 arquivos)
- **Origem:** Coorte clínica local de miocardite (HU-UFJF), convertida diretamente dos `.mat` revisados (Medviso Segment) e DICOM preservando geometria RAS real (validada contra `SimpleITK` com `corr = 1.0`, erro de matriz `affine < 2e-5 mm`).
- **Estrutura:**
  - `nifti/patient_X/` (`X ∈ {1, 2, 3, 5, 6, 7, 9, 10, 18, 19}` — 10 rotulados com anatomia e fibrose via `setstruct.Scar.Result`):
    - `img.nii[.gz]` (`float32`)
    - `seg_anat.nii[.gz]` (`int16`: `0=Fundo, 1=VE, 2=Mio Total, 3=VD`)
    - `seg_scar.nii[.gz]` (`int16`: `0=Fundo, 1=Fibrose`)
    - `seg_5class.nii[.gz]` (`int16`: padrão unificado CMR-MULTI `0=Fundo, 1=VE, 2=Mio Saudável, 3=Fibrose, 4=VD`)
  - `nifti/patient_20/img.nii[.gz]`: extraído da série DICOM `DE_single-shot_EC_psir_p2_MAG` (sem máscara manual).
  - `raw_mat_anonymized/patient_X.mat`: 10 arquivos `.mat` originais anonimizados (LGPD).
  - `manifest.csv` e `README.md`.

---

## 4. `robertgvds/hu-ufjf-dicom-archive` (Private — Backup Anonimizado)
- Contém os 11 pacotes ZIP anonimizados (`patient_1_dicom.zip`..`patient_20_dicom.zip`, 13.671 arquivos DICOM reais sem nomes civis, executáveis ou lixo de SO) + `dicom_archive_manifest.json`.
- Mantido separado de `hu-ufjf-lge-sax` para evitar montagem desnecessária de ~770 MB em kernels de treino.

---

## 5. Convenção de IDs `nnUNetv2` e Regras Anti-Leakage
- `Dataset501_CMRMULTI_LGE`: 5 classes CMR-MULTI (`imagesTr`=40, `imagesTs`=7 fixos `SEED=42`).
- `Dataset502_EMIDEC_LGE`: 5 classes EMIDEC (sem VD).
- `Dataset503_HUUFJF_LGE`: Coorte HU-UFJF (`seg_5class`).
- `Dataset510_LGE_UNIFIED_4C`: União harmonizada CMR-MULTI + EMIDEC (4 classes, sem VD).
- `Dataset520_SCAR_CASCADE_STAGE2`: Etapa 2 da cascata restrita à ROI do miocárdio (margem 8–12 px).
- **Anti-Leakage (Futuros Datasets):** Em **CARE-MyoPS (MyoPS++)**, Centro C = 45 casos do **MS-CMRSeg 2019 / MyoPS 2020** e Centro A = 81 casos do **EMIDEC**. Nunca combinar sem deduplicação prévia. Ignorar sempre datasets atriais (`LAScarQS`, `SLAWT`, `RAS_TR`).
