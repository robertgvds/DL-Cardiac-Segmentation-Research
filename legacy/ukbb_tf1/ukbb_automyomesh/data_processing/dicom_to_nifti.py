import os
import subprocess
import argparse
import re
from pathlib import Path

def convert_patient_dicoms(raw_patient_dir, output_base_dir, patient_id):
    """
    Varre a pasta do paciente e converte as sequências DCM para NIfTI.
    """
    out_dir = os.path.join(output_base_dir, str(patient_id))
    os.makedirs(out_dir, exist_ok=True)

    # Mapeamento corrigido para bater exatamente com os nomes das pastas (com underlines)
    sequence_mapping = {
        "CINE-EC": "cine_sa",          # eixo curto -> rede ukbb
        "CINE-2C_VE": "cine_2ch",      # 2 camaras (ventrículo esquerdo) -> rede ukbb
        "CINE-2C_VD": "cine_2ch_vd",   # 2 camaras (ventrículo direito)
        "CINE-4": "cine_4ch",          # 4 camaras -> rede ukbb
        "CINE-3C": "cine_3ch",         # 3 camaras
        "CINE-LOVT": "cine_lovt",      
        "CINE-ROVT": "cine_rovt",      
        "_PSIR": "lge_psir",           # LGE (PSIR - Fase)
        "_MAG": "lge_mag",             # LGE (Magnitude)
        "care_bolus": "care_bolus",    
        "haste": "haste",              
        "LOCALIZADOR": "localizer",    # Localizadores
        "Localizers": "localizer",
        "trufi_loc": "localizer_trufi"
    }

    # seq não podem ter número no final (exigência da rede)
    core_sequences = ["cine_sa", "cine_2ch", "cine_4ch"]

    for root, dirs, files in os.walk(raw_patient_dir):
        # Ignora pastas que não contêm arquivos DICOM (.dcm ou .IMA)
        if not any(f.lower().endswith(('.dcm', '.ima')) for f in files):
            continue

        folder_name = os.path.basename(root)
        target_name = None
        
        # Procura a chave correspondente no nome da pasta
        for key, mapped_name in sequence_mapping.items():
            # Convertendo ambos para maiúsculas para evitar erros de case sensitivity
            if key.upper() in folder_name.upper():
                target_name = mapped_name
                break
        
        # Se encontrou um mapeamento conhecido
        if target_name:
            if target_name not in core_sequences:
                # CORREÇÃO: Extrai o número da série que está após o underline (ex: _0013)
                series_match = re.search(r'_(\d+)$', folder_name)
                if series_match:
                    target_name = f"{target_name}_{series_match.group(1)}"
        else:
            # Se não encontrou no dicionário, limpa o nome original
            clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', folder_name)
            target_name = re.sub(r'_+', '_', clean_name).strip('_')

        print(f"  -> Extraindo: {folder_name} \n     -> Salvando como: {target_name}.nii.gz")
        
        cmd = [
            "dcm2niix",
            "-z", "y",
            "-f", target_name,
            "-o", out_dir,
            root 
        ]
        
        # Executa a conversão
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print(f"✅ Finalizado: {target_name}.nii.gz\n")
        else:
            print(f"❌ Erro ao converter {folder_name}: {result.stderr}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte DICOMs de múltiplos pacientes para NIfTI.")
    
    # Args: Agora o input deve ser a pasta RAIZ que contém todas as pastas dos pacientes
    parser.add_argument("-i", "--input", required=True, help="Caminho para a pasta raiz contendo as pastas dos pacientes (ex: /mnt/c/imgs/dicom)")
    
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_OUT = REPO_ROOT / "data" / "niiti"
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUT), help="Pasta de destino para os arquivos NIfTI convertidos (padrão: data/niiti)")

    args = parser.parse_args()

    root_dir = args.input

    if not os.path.exists(root_dir):
        print(f"Erro: O diretório raiz de entrada '{root_dir}' não existe.")
        exit(1)

    print("=======================================================")
    print(f"Iniciando conversão em lote a partir de: {root_dir}")
    print("=======================================================\n")

    # Varre a pasta raiz para encontrar as pastas individuais de cada paciente
    for item in sorted(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, item)
        
        # Garante que é uma pasta (ex: "PACIENTE 5")
        if os.path.isdir(patient_path):
            # Tenta extrair o número do paciente do nome da pasta (ex: "PACIENTE 5" -> "5")
            match = re.search(r'\d+', item)
            patient_id = match.group(0) if match else item.replace(" ", "_")
            
            print(f"=======================================================")
            print(f"🩺 Processando: {item} (Atribuído ID: {patient_id})")
            print(f"=======================================================")
            
            convert_patient_dicoms(
                raw_patient_dir=patient_path,
                output_base_dir=args.output,
                patient_id=patient_id
            )
    
    print("\n🎉 Conversão em lote concluída com sucesso! Os pacientes estão separados em pastas.")