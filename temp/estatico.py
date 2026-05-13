import nibabel as nib
import numpy as np

# Carrega a imagem problemática (LGE)
img = nib.load("input/lge/sa.nii.gz")
dados = img.get_fdata()

print(f"Shape original: {dados.shape}") 
# Provavelmente vai printar (160, 150, 14)

# Vamos adicionar 5 pixels de borda preta (zeros) no começo do Eixo Y 
# e 5 pixels no final do Eixo Y. Assim, 150 + 5 + 5 = 160.
# As regras de padding são: (Eixo X), (Eixo Y), (Eixo Z)
dados_padded = np.pad(dados, ((0, 0), (5, 5), (0, 0)), mode='constant')

print(f"Shape corrigido: {dados_padded.shape}")

# Salva a imagem curada
img_corrigida = nib.Nifti1Image(dados_padded, img.affine, img.header)
nib.save(img_corrigida, "input/lge/sa.nii.gz")
print("✅ Imagem corrigida para 160x160 com sucesso!")