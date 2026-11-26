import dicom2nifti
import dicom2nifti.settings as settings
import os

settings.disable_validate_slice_increment()
settings.disable_validate_orthogonal()
settings.disable_validate_orientation()

DICOM_DIR = "C:/imgs/dicom/PACIENTE 2/ANNA_DALVA_DE_LIMA_SOUZA/CORPO_CARDIO_20230214_092600_919000" 
OUTPUT_DIR = "dicom/2"

def convert_ima_to_nifti():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Lendo DICOMs (.IMA) em: {DICOM_DIR}")
    print("[*] Convertendo e forçando padrão RAS sem validação estrita...")
    
    try:
        dicom2nifti.convert_directory(DICOM_DIR, OUTPUT_DIR, compression=True, reorient=True)
        print(f"[+] Conversão concluída! Verifique a pasta: {OUTPUT_DIR}")
    except Exception as e:
        print(f"[-] Erro na conversão: {e}")

if __name__ == "__main__":
    convert_ima_to_nifti()