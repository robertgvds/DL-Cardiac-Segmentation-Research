from sa_segmenter import SASegmenter

fold_map = {
    1: "19",
    2: "5",
    3: "6",
    4: "2",
    5: "7",
    6: "18",
    7: "3",
    8: "1",
    9: "10",
    10: "9"
}

segmenter = SASegmenter(
    data_dir=r"./input/lge_dataset",
    model_path=None, 
    gpu_id='0',
    image_pattern='*lge.nii.gz'
)

# Executa o K-Fold
segmenter.run_cv_testing(
    fold_to_patient_dict=fold_map,
    models_base_dir=r"./segmentation/models",
    iteration=2000
)