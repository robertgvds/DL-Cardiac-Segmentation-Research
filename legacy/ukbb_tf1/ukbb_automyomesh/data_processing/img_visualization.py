import nibabel as nib
import cv2
import numpy as np

image = nib.load('/home/robertgvds/Projetos/IC/segmentacao-fisiocomp/AutoMyoMesh/input/2/sa_corrigido.nii.gz').get_fdata()


#cv2.imwrite('temp.png', image[:, :, 10, 10])

print(image.shape)
