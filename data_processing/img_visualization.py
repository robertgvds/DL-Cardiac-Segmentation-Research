import nibabel as nib
import cv2
import numpy as np

image = nib.load('/home/robertgvds/Projetos/IC/dl-cardiac-segmentation/dl-cardiac-segmentation-research/data/niiti/1/cine_sa.nii.gz').get_fdata()

cv2.imwrite('temp.png', image[:, :, 10, 10])

print(image.shape)
