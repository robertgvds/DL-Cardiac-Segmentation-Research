import os
import sys
import subprocess
import argparse
import numpy as np
import nibabel as nib
from datetime import datetime

# =====================================================================
# CONFIGURAÇÕES E CAMINHOS PADRÃO
# =====================================================================
DEFAULT_INPUT_FILEPATH = "input/10/10_lge.nii.gz"
MODELS_BASE_DIR = "trained_models/models_cv/fold_9/FCN_sa_level5_filter16_22333_batch1_iter2000_lr1e-05/FCN_sa_level5_filter16_22333_batch1_iter2000_lr1e-05.ckpt-000"
RELATIVE_DEPLOY_SCRIPT = "ukbb_model/common/deploy_network.py"

def apply_canonical_alignment(input_path: str) -> str:
    """
    Endireita a imagem (Orientação Canônica RAS) sem destruir as coordenadas físicas
    do paciente. Isto resolve o problema das imagens cortadas/esmagadas no Slicer,
    garantindo que a imagem seja vista ortogonalmente sem 'roubar' caixas delimitadoras
    de outros pacientes.
    """
    print(f"📐 Reorientando a imagem para o plano ortogonal (Canonical RAS)...")
    
    img = nib.load(input_path)
    
    # Transforma a imagem para a orientação mais reta possível nativamente
    canonical_img = nib.as_closest_canonical(img)
    
    # Salva com o sufixo _reformat para ser compatível com o limpador
    base_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path).replace('.nii.gz', '').replace('.nii', '')
    new_path = os.path.join(base_dir, f"{base_name}_reformat.nii.gz")
    
    nib.save(canonical_img, new_path)
    print(f"✅ Imagem endireitada e salva como: {os.path.basename(new_path)}")
    
    return new_path

def execute_segmentation(seq_name: str, data_dir: str, model_path: str, process_seq: bool, project_root: str) -> None:
    print("=" * 88)
    print(f"🧠 Deploying the segmentation network for {seq_name}...")
    print("=" * 88)
    
    absolute_deploy_script = os.path.join(project_root, RELATIVE_DEPLOY_SCRIPT)
    
    deploy_command = [
        sys.executable, absolute_deploy_script,
        "--seq_name", seq_name,
        "--data_dir", data_dir,
        "--model_path", model_path
    ]
    
    if process_seq:
        deploy_command.append("--process_seq")
        print("🎬 Modo CINE (4D Sequence) Ativado Automaticamente!")
    else:
        print("🖼️ Modo Imagem Estática (3D) Ativado.")
    
    environment_vars = os.environ.copy()
    environment_vars["CUDA_VISIBLE_DEVICES"] = "0"
    environment_vars["TF_CPP_MIN_LOG_LEVEL"] = "2"
    environment_vars["TF_USE_LEGACY_KERAS"] = "1"
    
    if "PYTHONPATH" in environment_vars:
        environment_vars["PYTHONPATH"] = f"{project_root}:{environment_vars['PYTHONPATH']}"
    else:
        environment_vars["PYTHONPATH"] = project_root

    try:
        subprocess.run(deploy_command, env=environment_vars, check=True, cwd=project_root)
        print("\n✅ Processamento da rede finalizado com sucesso!")
            
    except subprocess.CalledProcessError as error:
        print(f"\n❌ Erro crítico ao executar a segmentação! Código: {error.returncode}")

def run_pipeline(input_filepath: str, model_path: str, force_seq: bool, align_img: bool) -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    absolute_input_path = os.path.abspath(input_filepath)
    
    if not os.path.exists(absolute_input_path):
        print(f"Erro: O ficheiro '{absolute_input_path}' não foi encontrado.")
        return

    # --- ETAPA DE ALINHAMENTO CANÔNICO ---
    if align_img:
        absolute_input_path = apply_canonical_alignment(absolute_input_path)

    filename = os.path.basename(absolute_input_path)          
    seq_name = filename.replace('.nii.gz', '').replace('.nii', '')
    patient_dir = os.path.dirname(absolute_input_path)
    
    # --- AUTO-DETEÇÃO DE DIMENSÕES (3D vs 4D) ---
    print("\nAnalisando dimensões do ficheiro NIfTI...")
    img = nib.load(absolute_input_path)
    is_4d = len(img.shape) == 4
    
    process_seq = is_4d or force_seq

    print("\n" + "=" * 88)
    print("Starting the pipeline for patient data processing.")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Dir: {patient_dir}")
    print(f"Sequence: {seq_name}")
    print(f"Dimensões NIfTI: {img.shape}")
    print("=" * 88)
    
    execute_segmentation(seq_name, patient_dir, model_path, process_seq, project_root)

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Wrapper - AutoMyoMesh")
    parser.add_argument("-i", "--input", required=False, help="Caminho completo para a imagem")
    parser.add_argument("-m", "--model", required=False, help="Caminho base (prefixo) do modelo .ckpt")
    parser.add_argument("--seq", action="store_true", help="Forçar processamento de sequência 4D")
    
    # Flag para realizar o alinhamento visual perfeito para o 3D Slicer
    parser.add_argument("--align", action="store_true", help="Endireita a imagem (Canonical RAS) para visualização perfeita no Slicer sem cortar dados")
    
    args = parser.parse_args()
    
    final_input_path = args.input if args.input else DEFAULT_INPUT_FILEPATH
    final_model_path = args.model if args.model else MODELS_BASE_DIR
    
    run_pipeline(final_input_path, final_model_path, args.seq, args.align)

if __name__ == '__main__':
    main()