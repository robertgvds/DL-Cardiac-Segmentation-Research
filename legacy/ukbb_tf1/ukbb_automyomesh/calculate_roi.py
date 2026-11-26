"""
Calculadora Independente de ROI (Dice Score).

Este script lê as segmentações já preditas pelo modelo e compara-as
com as anotações manuais (Ground Truth) para calcular o Dice Similarity Coefficient
para as três estruturas cardíacas principais.
"""

import os
import glob
import re
import argparse
import numpy as np
import nibabel as nib


# CONFIGURAÇÕES DE DIRETÓRIOS
DEFAULT_PRED_DIR = "output/niftis_extraidos"
DEFAULT_GT_DIR = "input/niftis_extraidos"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculadora de ROI (Dice Score) para Segmentações NIfTI")
    parser.add_argument("-p", "--pred_dir", type=str, default=DEFAULT_PRED_DIR, help="Diretório com as predições (_seg_pred.nii.gz)")
    parser.add_argument("-g", "--gt_dir", type=str, default=DEFAULT_GT_DIR, help="Diretório com os Ground Truths (_seg_anat.nii.gz)")
    return parser.parse_args()

def extract_patient_id(filename: str) -> str:
    """Extrai o identificador base do paciente (ex: Patient_1) do nome do ficheiro."""
    match = re.search(r'(Patient_\d+)', filename)
    if match:
        return match.group(1)
    return None

def calculate_dice(pred: np.ndarray, target: np.ndarray, roi_label: int) -> float:
    """Calcula o Dice Similarity Coefficient para uma classe específica da ROI."""
    pred_roi = (pred == roi_label)
    target_roi = (target == roi_label)
    
    intersection = np.logical_and(pred_roi, target_roi).sum()
    total = pred_roi.sum() + target_roi.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total

def main():
    args = parse_arguments()
    pred_dir_abs = os.path.abspath(args.pred_dir)
    gt_dir_abs = os.path.abspath(args.gt_dir)
    
    pred_files = sorted(glob.glob(os.path.join(pred_dir_abs, "*_seg.nii.gz")))
    
    if not pred_files:
        print(f"Nenhum ficheiro de predição encontrado em: {pred_dir_abs}")
        return

    print("=" * 70)
    print("Iniciando Cálculo de ROI (Dice Score)")
    print("=" * 70)

    global_endo_lv, global_epi_lv, global_endo_rv = [], [], []

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        patient_id = extract_patient_id(filename)
        
        if not patient_id:
            continue
            
        gt_path = os.path.join(gt_dir_abs, f"{patient_id}_seg_anat.nii.gz")
        
        print(f"\n-> Analisando {patient_id}")
        
        if not os.path.exists(gt_path):
            print(f"   Aviso: Ficheiro Ground Truth não encontrado ({os.path.basename(gt_path)}).")
            continue
            
        try:
            pred_data = nib.load(pred_path).get_fdata()
            gt_data = nib.load(gt_path).get_fdata()
            
            # Verifica se as dimensões batem (crucial para o Dice)
            if pred_data.shape != gt_data.shape:
                print(f"   Erro: Dimensões incompatíveis. Predição {pred_data.shape} vs GT {gt_data.shape}")
                continue
            
            dice_endo_lv = calculate_dice(pred_data, gt_data, 1)
            dice_epi_lv = calculate_dice(pred_data, gt_data, 2)
            dice_endo_rv = calculate_dice(pred_data, gt_data, 3)
            
            global_endo_lv.append(dice_endo_lv)
            global_epi_lv.append(dice_epi_lv)
            global_endo_rv.append(dice_endo_rv)
            
            print(f"   [Label 1] Endocárdio LV : {dice_endo_lv:.4f}")
            print(f"   [Label 2] Epicárdio LV  : {dice_epi_lv:.4f}")
            print(f"   [Label 3] Endocárdio RV : {dice_endo_rv:.4f}")
            
        except Exception as e:
            print(f"   Erro ao processar as matrizes: {str(e)}")

    if global_endo_lv:
        print("\n" + "=" * 70)
        print("MÉTRICAS FINAIS - MÉDIA GLOBAIS")
        print("=" * 70)
        print(f"Endocárdio LV (Ventrículo Esquerdo) : {np.mean(global_endo_lv):.4f}")
        print(f"Epicárdio LV  (Miocárdio VE)        : {np.mean(global_epi_lv):.4f}")
        print(f"Endocárdio RV (Ventrículo Direito)  : {np.mean(global_endo_rv):.4f}")
        print("=" * 70)
    else:
        print("\nNenhuma métrica pôde ser calculada. Verifique os ficheiros de entrada.")

if __name__ == '__main__':
    main()