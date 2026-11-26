import os
from register import register_cine_to_lge

# Configure aqui o caminho absoluto ou relativo para o seu dataset
DATASET_DIR = "./input/dataset_final"

def run_registration_pipeline():
    print("Iniciando Pipeline de Registro: Cine -> LGE")
    
    # Itera sobre todas as pastas dentro do diretório do dataset
    for patient_folder in sorted(os.listdir(DATASET_DIR)):
        patient_path = os.path.join(DATASET_DIR, patient_folder)
        
        # Ignorar arquivos soltos, buscar apenas diretórios (paciente_1, paciente_2...)
        if not os.path.isdir(patient_path):
            continue
            
        print(f"\nProcessando: {patient_folder}")
        
        # Definir os caminhos dos arquivos de entrada
        cine_path = os.path.join(patient_path, f"{patient_folder}_sa.nii.gz")
        cine_seg_path = os.path.join(patient_path, f"{patient_folder}_sa_seg.nii.gz")
        lge_path = os.path.join(patient_path, f"{patient_folder}_lge.nii.gz")
        
        # Definir o caminho de saída (a nova máscara sobre a LGE)
        output_seg_path = os.path.join(patient_path, f"{patient_folder}_lge_myo_seg.nii.gz")
        
        # Checagem de segurança
        if not (os.path.exists(cine_path) and os.path.exists(lge_path) and os.path.exists(cine_seg_path)):
            print(f"  [AVISO] Arquivos faltando em {patient_folder}. Pulando...")
            continue
            
        # Executar a função do worker
        try:
            register_cine_to_lge(cine_path, lge_path, cine_seg_path, output_seg_path)
        except Exception as e:
            print(f"  [ERRO] Falha ao registrar {patient_folder}: {e}")

if __name__ == "__main__":
    run_registration_pipeline()