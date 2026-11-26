"""
Ajuste Manual Determinístico Direto (Cine vs LGE)
Gira a matriz de pixels bruta e ignora a compensação do Affine, 
forçando o 3D Slicer e a Rede Neural a enxergarem a nova orientação.
"""

import os
import glob
import numpy as np
import nibabel as nib

INPUT_DIR = "input/niftis_brutos"
OUTPUT_DIR = "input/niftis_alinhados_rede"

LGE_ROT90 = 1        
LGE_FLIP_X = True   
CINE_ROT90 = 3       
CINE_FLIP_X = True  

def process_deterministic_alignment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifti_files = glob.glob(os.path.join(INPUT_DIR, "*.nii.gz"))
    
    if not nifti_files:
        print(f"Nenhum arquivo em {INPUT_DIR}.")
        return

    for filepath in nifti_files:
        filename = os.path.basename(filepath)
        is_lge = "lge" in filename.lower()
        
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine
        
        rot_k = LGE_ROT90 if is_lge else CINE_ROT90
        flip = LGE_FLIP_X if is_lge else CINE_FLIP_X
        
        # A MÁGICA SIMPLIFICADA: Gira apenas o Array de Pixels
        if flip:
            data = np.flip(data, axis=0)
            
        if rot_k > 0:
            data = np.rot90(data, k=rot_k, axes=(0, 1))
            
        # Cria a imagem nova atrelando os pixels girados à matriz física original intocada
        new_img = nib.Nifti1Image(data.astype(img.get_data_dtype()), affine)
        
        out_path = os.path.join(OUTPUT_DIR, filename)
        nib.save(new_img, out_path)
        
        tipo = "LGE" if is_lge else "Cine"
        print(f" -> {filename} ({tipo}) | Flip X: {flip} | Rot: {rot_k * 90}º aplicados fisicamente.")

    print("\nAjuste concluído! Arraste os arquivos para o Slicer para confirmar o giro.")

if __name__ == "__main__":
    process_deterministic_alignment()