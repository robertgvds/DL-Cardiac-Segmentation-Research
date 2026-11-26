import os
import json
import glob
import shutil

def triagem_definitiva_mri(pasta_raiz, pasta_destino_base):
    print("===================================================")
    print("🚀 Iniciando Triagem Absoluta: SAX vs LAX")
    print("===================================================")

    pasta_sax = os.path.join(pasta_destino_base, "dataset_SAX")
    pasta_lax = os.path.join(pasta_destino_base, "dataset_LAX")
    pasta_erros = os.path.join(pasta_destino_base, "dataset_NAO_IDENTIFICADO")

    os.makedirs(pasta_sax, exist_ok=True)
    os.makedirs(pasta_lax, exist_ok=True)
    os.makedirs(pasta_erros, exist_ok=True)

    padrao_busca = os.path.join(pasta_raiz, "**", "*.json")
    lista_jsons = glob.glob(padrao_busca, recursive=True)

    if not lista_jsons:
        print(f"❌ Nenhum ficheiro .json encontrado em: {pasta_raiz}")
        return

    contagem = {"SAX": 0, "LAX": 0, "IGNORADO": 0, "ERRO": 0}
    palavras_sequencia = ['lge', 'cine']

    for caminho_json in lista_jsons:
        nome_base = caminho_json.replace('.json', '')
        caminho_nifti = f"{nome_base}.nii.gz"

        if not os.path.exists(caminho_nifti):
            continue

        try:
            with open(caminho_json, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception:
            continue

        nome_arquivo = os.path.basename(caminho_json).lower()
        caminho_completo = caminho_json.lower()
        orientacao = metadata.get("ImageOrientationText", "")
        descricao = str(metadata.get("SeriesDescription", "")).lower()
        protocolo = str(metadata.get("ProtocolName", "")).lower()

        # 1. VERIFICA SE É LGE OU CINE
        eh_lge_cine = any(kw in descricao or kw in protocolo for kw in palavras_sequencia)
        if 'lge' in caminho_completo or 'cine' in caminho_completo:
            eh_lge_cine = True

        if not eh_lge_cine:
            contagem["IGNORADO"] += 1
            continue # Ignora se não for LGE nem CINE

        # ====================================================
        # 2. MOTOR DE DECISÃO: SAX vs LAX (Nome é Rei!)
        # ====================================================
        categoria = None

        # A. Tenta adivinhar pelo NOME DO ARQUIVO ou PASTA (Prioridade Máxima)
        marcadores_sax = ['_sa', 'sa_', 'sax', 'short_axis', 'shortaxis', '/sa/']
        marcadores_lax = ['_la', 'la_', 'lax', 'long_axis', 'longaxis', '/la/', '2ch', '3ch', '4ch']

        if any(m in caminho_completo for m in marcadores_sax):
            categoria = "SAX"
        elif any(m in caminho_completo for m in marcadores_lax):
            categoria = "LAX"
        
        # B. Se o nome não disser nada, confia nos Metadados (Plano B)
        elif orientacao.startswith("Tra"):
            categoria = "SAX"
        elif orientacao.startswith("Cor") or orientacao.startswith("Sag"):
            categoria = "LAX"

        # ====================================================
        # 3. EXECUTAR A CÓPIA
        # ====================================================
        if categoria:
            pasta_alvo = pasta_sax if categoria == "SAX" else pasta_lax
            
            nome_paciente = os.path.basename(os.path.dirname(caminho_json))
            pasta_destino_paciente = os.path.join(pasta_alvo, nome_paciente)
            os.makedirs(pasta_destino_paciente, exist_ok=True)
            
            shutil.copy2(caminho_nifti, os.path.join(pasta_destino_paciente, os.path.basename(caminho_nifti)))
            shutil.copy2(caminho_json, os.path.join(pasta_destino_paciente, os.path.basename(caminho_json)))
            
            print(f"✅ [{categoria}] -> {nome_paciente}/{os.path.basename(caminho_nifti)}")
            contagem[categoria] += 1
        else:
            # Ficheiros bizarros que escaparam a todas as regras
            shutil.copy2(caminho_nifti, os.path.join(pasta_erros, os.path.basename(caminho_nifti)))
            contagem["ERRO"] += 1

    print("\n===================================================")
    print("✅ Triagem Completa Finalizada!")
    print(f"   -> 🟢 {contagem['SAX']} imagens Short Axis (SAX).")
    print(f"   -> 🔵 {contagem['LAX']} imagens Long Axis (LAX).")
    print(f"   -> ⚪ {contagem['IGNORADO']} ignoradas (não eram LGE/CINE).")
    if contagem["ERRO"] > 0:
        print(f"   -> 🔴 {contagem['ERRO']} imagens enviadas para a pasta NAO_IDENTIFICADO.")
    print("===================================================")

if __name__ == "__main__":
    PASTA_RAIZ_NIFTIS = "input/sua_pasta_com_subpastas" 
    PASTA_DESTINO_BASE = "output/dataset_final_separado" 
    
    triagem_definitiva_mri("new_data/niiti", "new_data/separado")