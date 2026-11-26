import os
import glob
import shutil
import subprocess
import re
import numpy as np
import nibabel as nib

INPUT_DIR = "input/niftis_extraidos"
OUTPUT_DIR = "output/niftis_extraidos"
MODELS_BASE_DIR = "new_model"

def extract_patient_number(filename: str) -> int:
    match = re.search(r'Patient_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def calculate_dice(pred: np.ndarray, target: np.ndarray, roi_label: int) -> float:
    pred_roi = (pred == roi_label)
    target_roi = (target == roi_label)
    
    intersection = np.logical_and(pred_roi, target_roi).sum()
    total = pred_roi.sum() + target_roi.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total

def get_model_prefix(fold_num: int) -> str:
    fold_dir = os.path.join(MODELS_BASE_DIR, f"fold_{fold_num}")
    subpastas = glob.glob(os.path.join(fold_dir, "FCN_sa_*"))
    
    if not subpastas:
        raise FileNotFoundError(f"Pasta do modelo não encontrada: {fold_dir}")
        
    pasta_modelo = subpastas[0]
    arquivos_index = glob.glob(os.path.join(pasta_modelo, "*.ckpt-*.index"))
    
    if not arquivos_index:
        raise FileNotFoundError(f"Ficheiro .index não encontrado em: {pasta_modelo}")

    ultimo_index = sorted(arquivos_index)[-1]
    return ultimo_index.replace('.index', '')

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_img.nii.gz")))
    if not image_files:
        print(f"Nenhuma imagem encontrada em {INPUT_DIR}")
        return

    print("=" * 70)
    print("Iniciando Pipeline de Avaliação (run_seg.py integrado)")
    print("=" * 70)

    global_endo_lv, global_epi_lv, global_endo_rv = [], [], []

    for index, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        patient_num = extract_patient_number(filename)
        
        if patient_num is None:
            continue
            
        patient_id = f"Patient_{patient_num}"
        fold_num = index + 1
        
        print(f"\n-> Processando {patient_id} usando Fold {fold_num}")
        
        try:
            # Encontra o prefixo do modelo para este fold
            model_prefix = get_model_prefix(fold_num)
            
            # Copia a imagem de entrada para a pasta de saída
            # Isso obriga o run_seg.py a guardar o resultado diretamente no output
            out_img_path = os.path.join(OUTPUT_DIR, filename)
            shutil.copy2(img_path, out_img_path)
            
            # Chama o run_seg.py passando a imagem que está na pasta de output
            subprocess.run([
                "python", "run_seg.py", 
                "-i", out_img_path, 
                "-m", model_prefix
            ], check=True)
            
            # O ficheiro gerado pelo deploy_network terá o prefixo 'seg_' na pasta output
            pred_path = os.path.join(OUTPUT_DIR, f"seg_{filename}")
            gt_path = os.path.join(INPUT_DIR, f"{patient_id}_seg_anat.nii.gz")
            
            # Renomeia o resultado gerado para o padrão desejado
            final_pred_path = os.path.join(OUTPUT_DIR, f"{patient_id}_seg_pred.nii.gz")
            if os.path.exists(pred_path):
                os.rename(pred_path, final_pred_path)
            
            # Faz a avaliação do ROI (Dice Score)
            if os.path.exists(final_pred_path) and os.path.exists(gt_path):
                pred_data = nib.load(final_pred_path).get_fdata()
                gt_data = nib.load(gt_path).get_fdata()
                
                dice_endo_lv = calculate_dice(pred_data, gt_data, 1)
                dice_epi_lv = calculate_dice(pred_data, gt_data, 2)
                dice_endo_rv = calculate_dice(pred_data, gt_data, 3)
                
                global_endo_lv.append(dice_endo_lv)
                global_epi_lv.append(dice_epi_lv)
                global_endo_rv.append(dice_endo_rv)
                
                print(f"   [ROI Dice] Endo LV (1): {dice_endo_lv:.4f}")
                print(f"   [ROI Dice] Epi LV  (2): {dice_epi_lv:.4f}")
                print(f"   [ROI Dice] Endo RV (3): {dice_endo_rv:.4f}")
            else:
                print(f"   Aviso: GROUND TRUTH não encontrado. ROI ignorado.")
                
            # Limpa a imagem original que foi copiada para o output
            if os.path.exists(out_img_path):
                os.remove(out_img_path)
                
        except Exception as e:
            print(f"   Erro ao processar: {e}")

    if global_endo_lv:
        print("\n" + "=" * 70)
        print("MÉTRICAS FINAIS DA VALIDAÇÃO CRUZADA GLOBAL (MÉDIA)")
        print("=" * 70)
        print(f"Endocárdio LV : {np.mean(global_endo_lv):.4f}")
        print(f"Epicárdio LV  : {np.mean(global_epi_lv):.4f}")
        print(f"Endocárdio RV : {np.mean(global_endo_rv):.4f}")
        print("=" * 70)

if __name__ == '__main__':
    main()