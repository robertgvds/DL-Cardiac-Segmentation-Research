import os
import urllib.request
import subprocess
import argparse
from pathlib import Path

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

def run_command(cmd, cwd, env):
    """Executa um comando no terminal tratando os erros."""
    print(f"\n▶ Executando: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"❌ Erro ao executar: {cmd}")
        # exit(1) # Removido para não parar o pipeline inteiro se uma etapa falhar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Inferência - UKBB Cardiac")
    parser.add_argument("-d", "--data_dir", required=True, help="Caminho absoluto para a pasta dos pacientes (ex: /kaggle/input/.../niiti)")
    parser.add_argument("-o", "--output_csv", default="results", help="Pasta para salvar os CSVs de resultados")
    parser.add_argument("-g", "--gpu", default="0", help="ID da GPU a ser utilizada")
    args = parser.parse_args()

    # Configuração de Caminhos
    REPO_ROOT = Path(__file__).resolve().parent
    UKBB_DIR = REPO_ROOT / "models" / "ukbb_cardiac"
    MODEL_DIR = UKBB_DIR / "trained_model"
    CSV_DIR = REPO_ROOT / args.output_csv
    
    os.makedirs(CSV_DIR, exist_ok=True)

    # Configuração do Ambiente (equivalente ao export do bash)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO_ROOT}/models:{env.get('PYTHONPATH', '')}"
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['TF_CPP_MIN_LOG_LEVEL'] = '2' # Esconde avisos chatos do TensorFlow

    # 1. Download dos Pesos (Ignora as imagens de demo!)
    download_models(MODEL_DIR)

    print('\n=============================================')
    print('  🧠 INICIANDO SEGMENTAÇÃO EIXO CURTO (SA)')
    print('=============================================')
    run_command(f"python3 common/deploy_network.py --seq_name cine_sa --data_dir '{args.data_dir}' --model_path trained_model/FCN_sa", cwd=UKBB_DIR, env=env)
    
    print('\n📊 Avaliando Volumes Ventriculares (SA)...')
    run_command(f"python3 short_axis/eval_ventricular_volume.py --data_dir '{args.data_dir}' --output_csv '{CSV_DIR}/ventricular_volume.csv'", cwd=UKBB_DIR, env=env)

    print('\n📏 Avaliando Espessura da Parede (SA)...')
    # Bug do output_max_csv corrigido diretamente na chamada!
    run_command(f"python3 short_axis/eval_wall_thickness.py --data_dir '{args.data_dir}' --output_max_csv '{CSV_DIR}/temp_max.csv' --output_csv '{CSV_DIR}/wall_thickness.csv'", cwd=UKBB_DIR, env=env)

    print('\n=============================================')
    print('  🧠 INICIANDO SEGMENTAÇÃO EIXO LONGO (LA)')
    print('=============================================')
    run_command(f"python3 common/deploy_network.py --seq_name cine_2ch --data_dir '{args.data_dir}' --model_path trained_model/FCN_la_2ch", cwd=UKBB_DIR, env=env)
    run_command(f"python3 common/deploy_network.py --seq_name cine_4ch --data_dir '{args.data_dir}' --model_path trained_model/FCN_la_4ch", cwd=UKBB_DIR, env=env)
    run_command(f"python3 common/deploy_network.py --seq_name cine_4ch --data_dir '{args.data_dir}' --seg4 --model_path trained_model/FCN_la_4ch_seg4", cwd=UKBB_DIR, env=env)

    print('\n📊 Avaliando Volumes Atriais (LA)...')
    run_command(f"python3 long_axis/eval_atrial_volume.py --data_dir '{args.data_dir}' --output_csv '{CSV_DIR}/atrial_volume.csv'", cwd=UKBB_DIR, env=env)

    print('\n🎉 PIPELINE FINALIZADO COM SUCESSO!')
    print(f"Os relatórios clínicos foram salvos em: {CSV_DIR}")