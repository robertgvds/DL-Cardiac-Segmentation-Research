import os
from datetime import datetime
import subprocess
import argparse

def execute_segmentation(seq_name, data_dir, process_seq):
    print("===================================================")
    print(f"🧠 Deploying the segmentation network for {seq_name}...")
    print("===================================================")
    
    comando = [
        "python3", "common/deploy_network.py",
        "--seq_name", seq_name,
        "--data_dir", data_dir,
        "--model_path", "trained_model/FCN_sa"
    ]
    
    # Se o utilizador pedir para processar a sequência, adicionamos a flag nativa do modelo
    if process_seq:
        comando.append("--process_seq")
        print("Modo CINE (4D Sequence) Ativado!")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    


    
    try:
        subprocess.run(comando, env=env, check=True)
        print("\nSegmentação finalizada com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"\nErro crítico ao executar a segmentação! Código: {e.returncode}")

def execute_commands(input_file, process_seq):
    filename = os.path.basename(input_file)          
    seq_name = filename.replace('.nii.gz', '').replace('.nii', '')
    patient_dir = os.path.dirname(input_file)
    
    print("========================================================================================")
    print("🚀 Starting the pipeline for patient data processing.")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Target Dir: {patient_dir}")
    print(f"🎞️ Sequence: {seq_name}")
    print("========================================================================================")
    
    execute_segmentation(seq_name, patient_dir, process_seq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline Wrapper - AutoMyoMesh")
    parser.add_argument("-i", "--input", required=True, help="Caminho completo para a imagem")
    # Adicionamos a nova flag ao SEU terminal
    parser.add_argument("--seq", action="store_true", help="Processa a sequência de tempo inteira (Imagens 4D)")
    
    args = parser.parse_args()
    
    execute_commands(args.input, args.seq)