# Exp 03 — U-Net ResNet34 (PyTorch SMP, 5 Classes) no CMR-MULTI (`v0.2-cmr-multi-benchmark`)

## Configuração
- **Arquitetura:** `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=5)`
- **Pré-processamento:** Resolução nativa `200×200` com zero-padding simétrico para `224×224`, recorte percentil (1–99%) + Z-score por volume.
- **Augmentation (apenas treino):** Translação leve, variação de brilho/contraste e ruído gaussiano (`flip=False`).
- **Perda:** `0.5 × CrossEntropy ponderada (bg=0.5, myo=2.0, scar=4.0) + 0.5 × Dice Loss multiclasse`.
- **Otimização:** `AdamW(lr=4e-4)`, `CosineAnnealingLR`, 70 épocas, `batch_size=16`, split estratificado `35 / 5 / 7` (`SEED=42`).

## Resultados Consolidados (Teste Inédito, N=7)
- **Dice VE:** `0.8832 ± 0.0397` (mediana `0.8773`)
- **Dice Miocárdio Total:** `0.7950 ± 0.0583` (mediana `0.8023` — superou UKBB Corrigido em 6/7 pacientes)
- **Dice Miocárdio Saudável:** `0.7798 ± 0.0475`
- **Dice Fibrose (N=5):** `0.5580 ± 0.1085` (mediana `0.5722`, máx. `0.6766`; média `0.6019` excluindo outlier `LGE_SAX_014`)
- **Dice VD:** `0.7466 ± 0.0833` (mediana `0.8034`)
- **Saudáveis (`LGE_SAX_005`, `008`):** Zero / quase zero falsos positivos de fibrose.

Métricas por paciente registradas em [metrics_test_n7.csv](metrics_test_n7.csv).
