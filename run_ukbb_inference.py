import os
import urllib.request
import subprocess
import argparse
from pathlib import Path
import sys

def download_models(model_dir):
    """Garante que os pesos da rede neural estão descarregados."""
    URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/trained_model/'
    models = ['FCN_sa', 'FCN_la_2ch', 'FCN_la_4ch', 'FCN_la_4ch_seg4']
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("🔄 Verificando pesos dos modelos...")
    for model_name in models:
        for ext in ['.meta', '.index', '.data-00000-of-00001']:
            filename = f"{model_name}{ext}"
            filepath = os.path.join(model_dir, filename)
            if not os.path.exists(filepath):
                print(f"   Baixando {filename}...")
                urllib.request.urlretrieve(URL + filename, filepath)
    print("✅ Modelos prontos!")

def prepare_compatibility_links(data_dir):
    """
    Cria atalhos temporários para enganar a rede ukbb, que exige nomes fixos (sa, la_2ch, la_4ch).
    Assim mantemos o nosso padrão semântico (cine_sa) sem quebrar o código original do autor.
    """
    mapping = {
        'cine_sa.nii.gz': 'sa.nii.gz',
        'cine_2ch.nii.gz': 'la_2ch.nii.gz',
        'cine_4ch.nii.gz': 'la_4ch.nii.gz'
    }
    print("\n🔗 Criando links de compatibilidade para a rede...")
    for patient_id in os.listdir(data_dir):
        p_dir = os.path.join(data_dir, patient_id)
        if os.path.isdir(p_dir):
            for src_name, dst_name in mapping.items():
                src_path = os.path.join(p_dir, src_name)
                dst_path = os.path.join(p_dir, dst_name)
                # Se a nossa imagem existe e o atalho ainda não, cria o atalho (symlink)
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)

def run_command(cmd, cwd, env, step_name="Processo"):
    """Executa um comando no terminal com bloco Try/Except rigoroso para debug."""
    print(f"\n▶ Executando: {step_name}")
    print(f"  Comando: {cmd}")
    
    try:
        # capture_output=True permite-nos ler o erro exato do terminal se falhar
        result = subprocess.run(cmd, shell=True, cwd=cwd, env=env, check=True, text=True, capture_output=True)
        # Se quiser ver o log completo de sucesso, descomente a linha abaixo:
        # print(result.stdout) 
        print(f"✅ {step_name} concluído com sucesso!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERRO CRÍTICO EM: {step_name}")
        print("="*50)
        print("📝 LOG DE ERRO (STDERR):")
        print(e.stderr)
        print("-" * 50)
        print("📝 SAÍDA PADRÃO ANTES DO ERRO (STDOUT):")
        print(e.stdout)
        print("="*50)
        print("⚠️ Parando o pipeline para investigação...")
        sys.exit(1) # Para o script imediatamente para vermos o erro

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Inferência - UKBB Cardiac")
    parser.add_argument("-d", "--data_dir", required=True, help="Caminho absoluto para a pasta dos pacientes")
    parser.add_argument("-o", "--output_csv", default="results", help="Pasta para salvar os CSVs")
    parser.add_argument("-g", "--gpu", default="0", help="ID da GPU a ser utilizada")
    args = parser.parse_args()

    REPO_ROOT = Path(__file__).resolve().parent
    UKBB_DIR = REPO_ROOT / "models" / "ukbb_cardiac"
    MODEL_DIR = UKBB_DIR / "trained_model"
    CSV_DIR = REPO_ROOT / args.output_csv
    
    os.makedirs(CSV_DIR, exist_ok=True)

    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO_ROOT}/models:{env.get('PYTHONPATH', '')}"
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    download_models(MODEL_DIR)
    
    # Cria os atalhos mágicos!
    prepare_compatibility_links(args.data_dir)

    print('\n=============================================')
    print('  🧠 INICIANDO SEGMENTAÇÃO EIXO CURTO (SA)')
    print('=============================================')
    run_command(f"python3 common/deploy_network.py --seq_name sa --data_dir '{args.data_dir}' --model_path trained_model/FCN_sa", 
                cwd=UKBB_DIR, env=env, step_name="Rede Eixo Curto (SA)")
    
    run_command(f"python3 short_axis/eval_ventricular_volume.py --data_dir '{args.data_dir}' --output_csv '{CSV_DIR}/ventricular_volume.csv'", 
                cwd=UKBB_DIR, env=env, step_name="Cálculo Volume (SA)")

    run_command(f"python3 short_axis/eval_wall_thickness.py --data_dir '{args.data_dir}' --output_max_csv '{CSV_DIR}/temp_max.csv' --output_csv '{CSV_DIR}/wall_thickness.csv'", 
                cwd=UKBB_DIR, env=env, step_name="Cálculo Espessura da Parede (SA)")

    print('\n=============================================')
    print('  🧠 INICIANDO SEGMENTAÇÃO EIXO LONGO (LA)')
    print('=============================================')
    run_command(f"python3 common/deploy_network.py --seq_name la_2ch --data_dir '{args.data_dir}' --model_path trained_model/FCN_la_2ch", 
                cwd=UKBB_DIR, env=env, step_name="Rede Eixo Longo 2 Câmaras (LA 2CH)")
    
    run_command(f"python3 common/deploy_network.py --seq_name la_4ch --data_dir '{args.data_dir}' --model_path trained_model/FCN_la_4ch", 
                cwd=UKBB_DIR, env=env, step_name="Rede Eixo Longo 4 Câmaras (LA 4CH)")
    
    run_command(f"python3 common/deploy_network.py --seq_name la_4ch --data_dir '{args.data_dir}' --seg4 --model_path trained_model/FCN_la_4ch_seg4", 
                cwd=UKBB_DIR, env=env, step_name="Rede Eixo Longo 4 Câmaras (Segmentação 4)")

    run_command(f"python3 long_axis/eval_atrial_volume.py --data_dir '{args.data_dir}' --output_csv '{CSV_DIR}/atrial_volume.csv'", 
                cwd=UKBB_DIR, env=env, step_name="Cálculo Volume Atrial (LA)")

    print('\n🎉 PIPELINE FINALIZADO COM SUCESSO!')
    print(f"Os relatórios clínicos foram salvos em: {CSV_DIR}")