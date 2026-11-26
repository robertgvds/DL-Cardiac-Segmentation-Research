# Exp 01 — Fine-Tuning UKBB (TF 1.x) na Coorte HU-UFJF (`v0.1-ukbb-hu-ufjf`)

- **Objetivo:** Prova de conceito inicial (IC / SEMIC-UFJF) adaptando a FCN pré-treinada em CINE do UK Biobank (TensorFlow 1.x, 4 classes: Fundo, VE, Miocárdio Total, VD) para exames LGE-CMR da coorte clínica do HU-UFJF.
- **Achados técnicos preservados:**
  - Evidenciou o *domain shift* severo entre CINE (sangue brilhante, miocárdio escuro) e LGE (sangue escuro, cicatriz brilhante).
  - Motivou a investigação de alinhamento espacial determinístico e a reconstrução geométrica RAS exata a partir de `setstruct.ImageOrientation` / `ImagePosition` nos `.mat` do Medviso Segment (consolidada na Fase 2 em `robertgvds/hu-ufjf-lge-sax`).
- **Código histórico limpo:** Preservado em `legacy/ukbb_tf1/` e `legacy/elastix_registration/`.
