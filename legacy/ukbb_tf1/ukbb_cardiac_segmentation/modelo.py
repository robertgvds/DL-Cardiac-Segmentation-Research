import os
import glob
import shutil
import re

# Caminhos (usando raw strings r"" por causa do Windows)
SRC_DIR = r"./segmentation/models_cv"
DEST_DIR = r"./segmentation/models"

def organizar_checkpoints():
    pastas_com_modelos = []
    
    # 1. Modo Radar: Encontra todas as pastas que realmente contêm os ficheiros .meta, 
    # não importando o quão escondidas estejam.
    for root, dirs, files in os.walk(SRC_DIR):
        if any(".meta" in f for f in files):
            pastas_com_modelos.append(root)
            
    pastas_com_modelos = sorted(pastas_com_modelos)
    
    if not pastas_com_modelos:
        print(f"Nenhum ficheiro .meta encontrado dentro de {SRC_DIR}")
        return
        
    for fold_idx, pasta_origem in enumerate(pastas_com_modelos, start=1):
        pasta_destino = os.path.join(DEST_DIR, str(fold_idx))
        os.makedirs(pasta_destino, exist_ok=True)
        
        print(f"\n[{fold_idx}/{len(pastas_com_modelos)}] Extraindo de: {os.path.basename(pasta_origem)}")
        
        arquivos_copiados = 0
        for nome_arquivo in os.listdir(pasta_origem):
            caminho_completo = os.path.join(pasta_origem, nome_arquivo)
            if not os.path.isfile(caminho_completo):
                continue
                
            # 2. Captura os ficheiros das iterações 1000 e 2000 com as respetivas extensões
            match = re.search(r'\.ckpt-(1000|2000)(\..+)$', nome_arquivo)
            if match:
                iteracao = match.group(1)
                extensao = match.group(2) # .meta, .index, ou .data-00000-of-00001
                
                # 3. Formato enxuto pedido: FCN_sa_2000.meta
                novo_nome = f"FCN_sa_{iteracao}{extensao}"
                destino_final = os.path.join(pasta_destino, novo_nome)
                
                shutil.copy2(caminho_completo, destino_final)
                print(f"  -> Salvo: {novo_nome}")
                arquivos_copiados += 1
                
        if arquivos_copiados == 0:
            print("  -> Nenhum checkpoint 1000 ou 2000 encontrado nesta pasta.")

if __name__ == "__main__":
    organizar_checkpoints()