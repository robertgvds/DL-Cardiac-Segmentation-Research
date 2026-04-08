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

    # Mapeamento completo baseado na sua estrutura do hospital
    sequence_mapping = {
        "CINE-EC": "cine_sa",          # eixo curto -> rede ukbb
        "CINE-2C VE": "cine_2ch",      # 2 camaras (ventrículo esquerdo) -> rede ukbb
        "CINE-2C VD": "cine_2ch_vd",   # 2 camaras (ventrículo direito)
        "CINE-4": "cine_4ch",          # 4 camaras -> rede ukbb
        "CINE-3C": "cine_3ch",         # 3 camaras
        "CINE-LOVT": "cine_lovt",      # ?
        "CINE-ROVT": "cine_rovt",      # ?
        "_PSIR": "lge_psir",           # LGE (PSIR - Fase)
        "_MAG": "lge_mag",             # LGE (Magnitude)
        "care_bolus": "care_bolus",    # ?
        "haste": "haste",              # ?
        "LOCALIZADOR": "localizer",    # Localizadores
        "Localizers": "localizer",
        "trufi_loc": "localizer_trufi"
    }

    # seq não podem ter número no final (exigência da rede)
    core_sequences = ["cine_sa", "cine_2ch", "cine_4ch"]

    print(f"\n[{patient_id}] Analisando diretório e renomeando todas as sequências...")

    for root, dirs, files in os.walk(raw_patient_dir):
        if not any(f.lower().endswith('.dcm') for f in files):
            continue

        folder_name = os.path.basename(root)
        target_name = None
        
        
        for key, mapped_name in sequence_mapping.items():
            if key in folder_name:
                target_name = mapped_name
                break
        
        # evita ficheiros sobrescritos
        if target_name:
            if target_name not in core_sequences:
                # extrai o número da série no final do nome
                series_match = re.search(r'-\s*(\d+)$', folder_name)
                if series_match:
                    target_name = f"{target_name}_{series_match.group(1)}"
        else:
            # se não encontrou no dicionário, usa o nome original
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
            print(f"Finalizado: {target_name}.nii.gz")
        else:
            print(f"Erro ao converter {folder_name}: {result.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte DICOMs de CINE MRI para NIfTI.")
    
    # Args
    parser.add_argument("-i", "--input", required=True, help="Caminho para a pasta bruta do paciente contendo os DICOMs")
    parser.add_argument("-p", "--patient_id", required=True, help="Identificador do paciente (ex: 1, Paciente_01)")
    
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_OUT = REPO_ROOT / "data" / "niiti"
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUT), help="Pasta de destino para os arquivos NIfTI convertidos (padrão: data/niiti)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Erro: O diretório de entrada '{args.input}' não existe.")
        exit(1)

    convert_patient_dicoms(
        raw_patient_dir=args.input,
        output_base_dir=args.output,
        patient_id=args.patient_id
    )
    
    print("\nConversão concluída!")