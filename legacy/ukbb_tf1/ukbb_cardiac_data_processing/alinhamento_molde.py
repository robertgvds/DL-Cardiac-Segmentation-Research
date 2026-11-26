"""
Visual Content-Based 2D Alignment (LGE & Cine)
Usa detecção de bordas (Sobel) + Bounding Box Anatómico para impedir o tombamento.
"""

import os
import glob
import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage.filters import sobel

# =========================================================================
# CONFIGURAÇÃO: Aponte para a imagem CINE que funciona como molde
# =========================================================================
TEMPLATE_PATH = "input/patient.nii.gz"
INPUT_DIR = "input/niftis_brutos"
OUTPUT_DIR = "input/niftis_alinhados_rede"

def flip_axis(data, affine, axis):
    w = data.shape[axis]
    new_data = np.flip(data, axis=axis)
    new_affine = affine.copy()
    new_affine[:, axis] = -affine[:, axis]
    new_affine[:, 3] = affine[:, 3] + affine[:, axis] * (w - 1)
    return new_data, new_affine

def rot90_2d(data, affine, k):
    k = k % 4
    new_data = data
    new_affine = affine.copy()
    
    for _ in range(k):
        h = new_data.shape[1]
        new_data = np.rot90(new_data, k=1, axes=(0, 1))
        c0, c1 = new_affine[:, 0].copy(), new_affine[:, 1].copy()
        new_affine[:, 0] = -c1
        new_affine[:, 1] = c0
        new_affine[:, 3] = new_affine[:, 3] + c1 * (h - 1)
        
    return new_data, new_affine

def get_all_2d_orientations(data, affine):
    variations = []
    for flip_x in [False, True]:
        d1, a1 = data, affine
        if flip_x:
            d1, a1 = flip_axis(d1, a1, axis=0)
        for k in range(4):
            d2, a2 = rot90_2d(d1, a1, k)
            variations.append((d2, a2, flip_x, k))
    return variations

def extract_middle_slice(data):
    if data.ndim == 4:
        return data[:, :, data.shape[2]//2, 0]
    return data[:, :, data.shape[2]//2]

def get_anatomical_ratio(slice_data):
    """Mede a proporção real do corpo ignorando as bordas pretas."""
    # Considera como corpo qualquer pixel com brilho superior a 5% do máximo (ignora ruído)
    threshold = np.max(slice_data) * 0.05
    mask = slice_data > threshold
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return 1.0
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    altura_real = rmax - rmin
    largura_real = cmax - cmin
    
    if largura_real == 0: return 1.0
    return altura_real / largura_real

def calculate_visual_similarity(slice_molde, slice_teste):
    # 1. TRAVA ANTI-DEITAR COM BOUNDING BOX REAL
    ratio_molde = get_anatomical_ratio(slice_molde)
    ratio_teste = get_anatomical_ratio(slice_teste)
    
    penalidade = 0
    # Se o molde for vertical (> 1.05) e o teste ficar horizontal (< 0.95), aplica punição extrema
    if (ratio_molde > 1.05 and ratio_teste < 0.95) or (ratio_molde < 0.95 and ratio_teste > 1.05):
        penalidade = 2.0
        
    # 2. FILTRO SOBEL (Lida com as cores invertidas entre LGE e Cine)
    edges_molde = sobel(slice_molde)
    edges_teste = sobel(slice_teste)
    
    edges_teste_resized = resize(edges_teste, edges_molde.shape, anti_aliasing=True, preserve_range=True)
    
    std_molde = np.std(edges_molde)
    std_teste = np.std(edges_teste_resized)
    
    if std_molde == 0 or std_teste == 0:
        return -10 - penalidade
        
    norm_molde = (edges_molde - np.mean(edges_molde)) / std_molde
    norm_teste = (edges_teste_resized - np.mean(edges_teste_resized)) / std_teste
    
    ncc = np.mean(norm_molde * norm_teste)
    return ncc - penalidade

def align_all_patients_by_vision():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Carregando o Molde Visual Perfeito...")
    template_img = nib.load(TEMPLATE_PATH)
    template_slice = extract_middle_slice(template_img.get_fdata())
    
    patient_files = glob.glob(os.path.join(INPUT_DIR, "*_img.nii.gz"))
    
    for filepath in patient_files:
        filename = os.path.basename(filepath)
        patient_base = filename.replace("_img.nii.gz", "")
        print(f"\nAnalisando contornos do paciente: {patient_base}")
        
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine
        
        variations = get_all_2d_orientations(data, affine)
        
        best_score = -float('inf')
        best_transform = None
        
        for (v_data, v_affine, flip_x, rot_k) in variations:
            v_slice = extract_middle_slice(v_data)
            score = calculate_visual_similarity(template_slice, v_slice)
            
            if score > best_score:
                best_score = score
                best_transform = (flip_x, rot_k)
                
        win_flip_x, win_rot_k = best_transform
        print(f"  -> Orientação Ótima! (Score: {best_score:.3f}) | Flip X: {win_flip_x} | Rot90: {win_rot_k}x")
        
        def apply_and_save(input_path, out_suffix):
            if not os.path.exists(input_path): return
            nii = nib.load(input_path)
            d, a = nii.get_fdata(), nii.affine
            
            if win_flip_x:
                d, a = flip_axis(d, a, axis=0)
            d, a = rot90_2d(d, a, win_rot_k)
            
            out_path = os.path.join(OUTPUT_DIR, f"{patient_base}{out_suffix}")
            nib.save(nib.Nifti1Image(d.astype(nii.get_data_dtype()), a, nii.header), out_path)

        apply_and_save(filepath, "_img.nii.gz")
        apply_and_save(os.path.join(INPUT_DIR, f"{patient_base}_seg_anat.nii.gz"), "_seg_anat.nii.gz")
        apply_and_save(os.path.join(INPUT_DIR, f"{patient_base}_seg_scar.nii.gz"), "_seg_scar.nii.gz")
        
        print("  -> Guardado.")

if __name__ == "__main__":
    align_all_patients_by_vision()