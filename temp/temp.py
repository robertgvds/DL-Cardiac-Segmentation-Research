import nibabel as nib
import numpy as np

img_demo = nib.load("input/pacient/sa.nii.gz")
img_sua = nib.load("input/2/sa_estatico.nii.gz")

print("Demo Affine:")
print(np.round(img_demo.affine, 2))
print("\nSua Affine:")
print(np.round(img_sua.affine, 2))

data_demo = img_demo.get_fdata()
data_sua = img_sua.get_fdata()
print(f"Demo Min/Max : {np.min(data_demo):.2f} / {np.max(data_demo):.2f}")
print(f"Sua Min/Max  : {np.min(data_sua):.2f} / {np.max(data_sua):.2f}")

# Carrega a sua imagem estática
img_sua = nib.load("input/2/sa_estatico.nii.gz")
dados = img_sua.get_fdata()

# O Eixo Y (índice 1) está com os sinais trocados. Vamos espelhar a matriz NumPy!
dados_espelhados = np.flip(dados, axis=1)

# Cria um novo NIfTI com a matriz na posição correta para a rede
img_corrigida = nib.Nifti1Image(dados_espelhados, img_sua.affine, img_sua.header)
nib.save(img_corrigida, "input/2/sa_corrigido.nii.gz")
print("Imagem espelhada e salva com sucesso!")
