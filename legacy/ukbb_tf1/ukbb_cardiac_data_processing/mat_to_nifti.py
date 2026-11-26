"""
MATLAB to NIfTI Converter (Pure Extraction)
Extrai os dados originais sem forçar rotações ou flips.
Corrige apenas a dimensão 4D (Tempo vs Profundidade).
"""

import os
import glob
import scipy.io as sio
import numpy as np
import nibabel as nib
from skimage.draw import polygon
from typing import Tuple, Any, Optional

DEFAULT_INPUT_DIR = "input/mats_originais" 
DEFAULT_OUTPUT_DIR = "input/niftis_brutos"

LABEL_ENDO_LV = 1  
LABEL_EPI_LV = 2   
LABEL_ENDO_RV = 3  
LABEL_SCAR = 1     

def build_pure_affine(setstruct: Any) -> np.ndarray:
    try:
        dx = getattr(setstruct, 'ResolutionX', 1.0)
        dy = getattr(setstruct, 'ResolutionY', 1.0)
        thickness = getattr(setstruct, 'SliceThickness', 1.0)
        gap = getattr(setstruct, 'SliceGap', 0.0)
        dz = thickness + gap

        pos = getattr(setstruct, 'ImagePositionPatient', [0.0, 0.0, 0.0])
        ori = getattr(setstruct, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        row_dir = np.array(ori[:3])
        col_dir = np.array(ori[3:])
        slice_dir = np.cross(row_dir, col_dir)

        affine = np.eye(4)
        affine[0:3, 0] = row_dir * dx
        affine[0:3, 1] = col_dir * dy
        affine[0:3, 2] = slice_dir * dz
        affine[0:3, 3] = np.array(pos)

        return affine
    except Exception:
        return np.eye(4)

def get_coordinates(setstruct: Any, attr_name: str, z: int, t: int, z_size: int, t_size: int) -> Optional[np.ndarray]:
    if not hasattr(setstruct, attr_name): return None
    val = getattr(setstruct, attr_name)
    if len(val.shape) == 3: return val[:, t, z]
    elif len(val.shape) == 2:
        if t_size == 1: return val[:, z]
        elif z_size == 1: return val[:, t]
    return None

def draw_mask_polygon(mask: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, label: int, slice_shape: Tuple[int, int], mask_idx: tuple) -> None:
    if x_coords is not None and y_coords is not None:
        valid_points = ~np.isnan(x_coords)
        if np.sum(valid_points) > 3:  
            cx = np.round(x_coords[valid_points]).astype(int) - 1
            cy = np.round(y_coords[valid_points]).astype(int) - 1
            rr, cc = polygon(cx, cy, shape=slice_shape)
            mask[mask_idx][rr, cc] = label

def process_mat_files(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))
    
    for mat_filepath in mat_files:
        patient_name = os.path.basename(mat_filepath).replace('.mat', '')
        print(f"Extraindo: {patient_name}")
        
        try:
            mat_data = sio.loadmat(mat_filepath, squeeze_me=True, struct_as_record=False)
            
            raw_image = None
            if 'im' in mat_data and hasattr(mat_data['im'], 'shape') and getattr(mat_data['im'], 'size', 0) > 0:
                raw_image = mat_data['im']
                
            setstruct = None
            if 'setstruct' in mat_data:
                setstruct = mat_data['setstruct']
                if isinstance(setstruct, np.ndarray) and getattr(setstruct, 'size', 0) > 0:
                    setstruct = setstruct[0]
                    
            if raw_image is None and setstruct is not None and hasattr(setstruct, 'IM'):
                if getattr(setstruct.IM, 'size', 0) > 0:
                    raw_image = setstruct.IM
                    
            if raw_image is None or getattr(raw_image, 'size', 0) == 0: continue
                
            patient_affine = np.eye(4)
            
            if setstruct is not None:
                patient_affine = build_pure_affine(setstruct)
                z_size = int(getattr(setstruct, 'ZSize', 1))
                t_size = int(getattr(setstruct, 'TSize', 1))
                
                if len(raw_image.shape) == 4 and raw_image.shape[2] == t_size and raw_image.shape[3] == z_size:
                    raw_image = np.ascontiguousarray(np.transpose(raw_image, (0, 1, 3, 2)))
                
                original_slice_shape = raw_image.shape[:2]
                anat_seg = np.zeros_like(raw_image, dtype=np.uint8)
                scar_seg = np.zeros_like(raw_image, dtype=np.uint8)
                
                if hasattr(setstruct, 'EndoX'):
                    for t in range(t_size):
                        for z in range(z_size):
                            mask_idx = (slice(None), slice(None), z, t) if len(raw_image.shape) == 4 else (slice(None), slice(None), z)
                            draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'EpiX', z, t, z_size, t_size), get_coordinates(setstruct, 'EpiY', z, t, z_size, t_size), LABEL_EPI_LV, original_slice_shape, mask_idx)
                            draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'EndoX', z, t, z_size, t_size), get_coordinates(setstruct, 'EndoY', z, t, z_size, t_size), LABEL_ENDO_LV, original_slice_shape, mask_idx)
                            draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'RVEndoX', z, t, z_size, t_size), get_coordinates(setstruct, 'RVEndoY', z, t, z_size, t_size), LABEL_ENDO_RV, original_slice_shape, mask_idx)

                    # Bloco da Fibrose Restaurado
                    if hasattr(setstruct, 'Roi'):
                        for roi in (setstruct.Roi if isinstance(setstruct.Roi, np.ndarray) else [setstruct.Roi]):
                            try:
                                z_roi, t_roi = int(getattr(roi, 'Z', 1)) - 1, int(getattr(roi, 'T', 1)) - 1
                                if 0 <= z_roi < z_size and 0 <= t_roi < t_size:
                                    cx, cy = np.round(roi.X).astype(int) - 1, np.round(roi.Y).astype(int) - 1
                                    rr, cc = polygon(cx, cy, shape=original_slice_shape)
                                    idx_scar = (slice(None), slice(None), z_roi, t_roi) if len(raw_image.shape) == 4 else (slice(None), slice(None), z_roi)
                                    scar_seg[idx_scar][rr, cc] = LABEL_SCAR
                            except Exception: pass

            nib.save(nib.Nifti1Image(raw_image, affine=patient_affine), os.path.join(output_dir, f"{patient_name}_img.nii.gz"))
            if np.any(anat_seg): nib.save(nib.Nifti1Image(anat_seg, affine=patient_affine), os.path.join(output_dir, f"{patient_name}_seg_anat.nii.gz"))
            if np.any(scar_seg): nib.save(nib.Nifti1Image(scar_seg, affine=patient_affine), os.path.join(output_dir, f"{patient_name}_seg_scar.nii.gz"))

        except Exception as e:
            print(f"Erro em {patient_name}: {e}")

if __name__ == "__main__":
    process_mat_files(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)