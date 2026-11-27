# Revisão Bibliográfica — U-Net com Encoder ResNet34 em LGE-CMR e RM Cardíaca (2024–2026)

> **PDF arquivado localmente (gitignored):** `docs/papers/pdfs/2024_Hoh_Impact_LGE_Resolution_UNet_ResNet34_Scar_Segmentation.pdf`

---

## 1. Artigo Principal — Hoh et al. (2024), *JCMR*

- **Título:** *Impact of late gadolinium enhancement image acquisition resolution on neural network based automatic scar segmentation*
- **Autores:** Tobias Hoh, Isabel Margolis, Jonathan Weine, Thomas Joyce, Robert Manka, Miriam Weisskopf, Nikola Cesarovic, Maximilian Fuetterer, Sebastian Kozerke
- **Publicação:** *Journal of Cardiovascular Magnetic Resonance*, 2024, 26(1):101031
- **DOI / Acesso:** `10.1016/j.jocmr.2024.101031` — [PMC10981112](https://pmc.ncbi.nlm.nih.gov/articles/PMC10981112/)
- **Dataset e Classes:** 36 volumes 3D LGE suínos (IAM agudo, 2.049 fatias SAX 2D, validação cruzada 5-fold por volume) avaliados sob resoluções no plano de `0.7 mm`, `1.2 mm`, `1.7 mm` e treino multi-resolução (`0.7–1.7 mm`). Classes: Fundo, Miocárdio Saudável, Fibrose/Cicatriz (`I > SD5`).

### Arquitetura e Hiperparâmetros
- **Rede:** `segmentation_models_pytorch` (SMP) **2D U-Net com encoder ResNet34** (pré-treinado em ImageNet), 5 estágios de *downsampling*, canais do decoder `(256, 128, 64, 32, 16)` com `BatchNorm`.
- **Entrada e Normalização:** Recorte centrado no VE (`128 × 128`), normalização de intensidade para `[0, 1]`.
- **Perda e Otimização:** Dice Loss multiclasse, `Adam(lr=1e-3)`, `batch_size=16`, `100 épocas`.
- **Augmentation e Pós-processamento:** Deformação elástica aleatória, transformações de intensidade e ruído gaussiano; **TTA** (*Test-Time Augmentation*) e abertura morfológica na máscara de cicatriz para remover falsos positivos isolados de 1–2 voxels.

### Principais Resultados e Conclusões
- **Desempenho:** Dice médio `0.87–0.88` (miocárdio) e `0.59–0.66` (cicatriz); Dice mediano `0.88–0.89` (miocárdio) e `0.70–0.79` (cicatriz).
- **ResNet34 vs. ResNet50:** Os autores testaram explicitamente o encoder **ResNet50** e constataram **ausência de ganho sobre ResNet34**, concluindo que a U-Net ResNet34 já possui capacidade suficiente para segmentação de fibrose em LGE-CMR sem sofrer maior *overfitting*.
- **Achado Crítico de Generalização (diretamente aplicável ao HU-UFJF):** Treinar em resolução única gera forte viés volumétrico quando testado em exames de menor resolução espacial. O **treino multi-resolução** (simulação de resoluções mais grossas via filtragem/desfoque gaussiano) eliminou o viés de escala (`0.0 p.p.` erro mediano de fração de cicatriz).

---

## 2. Evidências Complementares Recentes (2024–2026)

1. **CMR-MULTI / CMRSeg 2026 (MICCAI 2026) — Top-5 com SMP U-Net ResNet34:**
   - Solução classificada em **5º lugar geral** (`adidukre-coder/cmr-multi-genmi`) utilizou **2D SMP U-Net com encoder ResNet34 (ImageNet)** (`~24,4M` parâmetros por vista) em `256 × 256` na Task 2 (LGE), alcançando **0.79 no Task 2 Score**, **0.77 LGE DSC**, **0.91 vPCC de massa de cicatriz** e **0.37 RAE** frente a pipelines pesados de nnU-Net 3D, MedNeXt e U-Mamba.
2. **Ba Mahel et al. (2025), *Bioengineering* 12(6):665 (`10.3390/bioengineering12060665`, [PMC12189363](https://pmc.ncbi.nlm.nih.gov/articles/PMC12189363/)):**
   - Benchmark no **EMIDEC 2020 (LGE-MRI)** e ACDC 2017 (`256 × 256`, Dice Loss, `Adam lr=1e-4`, 100 épocas): adicionar backbone **ResNet34** à UNet++ elevou o desempenho de `93.90%` para **`95.28%` Dice/F1** (`+2.38%` mIoU), ficando competitivo frente a arquiteturas com atenção (`ResNeSt50d + SE` atingiu `96.70%`).
3. **Kollmann et al. (2024), *Scientific Reports* 14:11009 (`10.1038/s41598-024-61417-4`, [PMC11094053](https://pmc.ncbi.nlm.nih.gov/articles/PMC11094053/)):**
   - 2D U-Net com **ResNet34** em modelo suíno de infarto (`256 × 256`, General Dice Loss, Adam) superou o Dice inter-observador humano (`LV=0.88 vs. 0.86`, `Myo=0.84 vs. 0.82`) com elevada eficiência amostral.
4. **Pirruccello et al. (2024), *Nature Communications* 15(1):4304 ([PMC11109224](https://pmc.ncbi.nlm.nih.gov/articles/PMC11109224/)):**
   - Confirmou a robustez da U-Net ResNet34 (ImageNet) em larga escala no UK Biobank multi-vista (SAX, 2CH, 3CH, 4CH).

### Síntese: A U-Net ResNet34 é Promissora?
**Sim.** A literatura 2024–2026 confirma que a **2D U-Net com encoder ResNet34 pré-treinado em ImageNet** continua altamente promissora como baseline forte, rápido (~6–10 min na Tesla T4) e eficiente em amostras pequenas (`N=35–47` volumes), especialmente quando combinada com:
- **Reamostragem física isotrópica no plano (`1.0 × 1.0 mm`)** para eliminar o *domain shift* de escala entre **CMR-MULTI** (`0.49–1.26 mm/px`, `200×200`) e **HU-UFJF** (`1.83–2.68 mm/px`, tórax completo `142–208 px`).
- **Augmentation multi-resolução (blur gaussiano + escala `0.80–1.25×` + deformação elástica leve + rotação `±25°`, `flip=False`)** inspirado em Hoh et al. (2024).
- **Perda combinada `Weighted CE + Soft Dice`**, `AdamW(lr=3e-4)` com *warmup* + *CosineAnnealingLR* (120 épocas).
- **Inferência 2D em janela deslizante com ponderação gaussiana (`256×256`, sobreposição 50%) + TTA rotacional suave + pós-processamento 3D (maior componente conexo do coração + remoção de ruído isolado de fibrose)**.

---

## 3. Protocolo Oficial de Métricas do Challenge CMR-MULTI (MICCAI 2026)

Fonte verificada: `https://mwm2026.github.io/cmr-multi` (CMRSeg 2026, MICCAI 2026).

- **Task 2 (LGE Scar Segmentation & Quantitative Analysis):**
  - **DSC (Dice Similarity Coefficient):** avaliado por classe de primeiro plano; na convenção estrita do *leaderboard*, `both empty = 1.0` e `one empty = 0.0` (penalizando falsos positivos de fibrose em exames saudáveis). Reportamos separadamente tanto o **Dice GT+** (apenas pacientes com fibrose no GT, padrão clínico) quanto o **Board DSC**.
  - **Massa de Fibrose (g):** $\text{ScarMass} = N_{\text{voxels}} \times V_{\text{voxel (mL)}} \times 1.05\ \text{g/mL}$.
  - **vPCC:** correlação de Pearson volumétrica entre massa de fibrose predita e GT.
  - **RAE:** erro absoluto relativo médio da massa de fibrose nos casos com fibrose no GT ($\text{mean}(|\hat{m} - m| / m)$).
  - **Fórmula Oficial Task 2:**
    $$\text{Score}_{\text{Task2}} = 0.7 \times \text{DSC}_{\text{board}} + 0.3 \times \left(0.5 \times \max(0, \text{vPCC}) + \frac{0.5}{1 + \text{RAE}}\right)$$
- **Métricas Geométricas de Superfície (Task 1 / Clínicas Complementares):**
  - **HD (mm):** distância de Hausdorff **máxima** em milímetros (convenção oficial do challenge).
  - **HD95 (mm):** distância de Hausdorff no percentil 95 (padrão clínico robusto a outliers).
  - **ASD / ASSD (mm):** distância média simétrica de superfície em milímetros.
  - **Score Geométrico Anatômico (adaptado para Exp 05A — 4 classes):**
    $$\text{Score}_{\text{Anat}} = 0.4 \times \text{DSC} + \frac{0.3}{1 + \text{HD}} + \frac{0.3}{1 + \text{ASD}}$$
