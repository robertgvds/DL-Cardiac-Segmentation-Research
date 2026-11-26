import os
import shutil
import subprocess
import glob
import sys

# =====================================================================
# CONFIGURAÇÕES DE DIRETÓRIOS
# =====================================================================
INPUT_DIR = "new_data/niiti"
OUTPUT_DIR = "output_new_data"

def process_batch():
    project_root = os.path.abspath(os.getcwd())
    input_dir_abs = os.path.abspath(INPUT_DIR)
    output_dir_abs = os.path.abspath(OUTPUT_DIR)
    
    network_output_dir = os.path.join(project_root, "output")

    if not os.path.exists(input_dir_abs):
        print(f"❌ Erro: Diretório de entrada '{input_dir_abs}' não encontrado.")
        return

    for patient_id in sorted(os.listdir(input_dir_abs)):
        patient_path = os.path.join(input_dir_abs, patient_id)
        
        if not os.path.isdir(patient_path):
            continue

        print("\n" + "="*70)
        print(f"🩺 Processando Paciente: {patient_id}")
        print("="*70)
        
        out_cine_dir = os.path.join(output_dir_abs, f"paciente_{patient_id}", "cine")
        out_lge_dir = os.path.join(output_dir_abs, f"paciente_{patient_id}", "lge")
        os.makedirs(out_cine_dir, exist_ok=True)
        os.makedirs(out_lge_dir, exist_ok=True)

        nii_files = glob.glob(os.path.join(patient_path, "*.nii.gz"))
        
        for file_path in sorted(nii_files):
            filename = os.path.basename(file_path)
            
            if filename.startswith("cine_sa"):
                category = "cine"
                out_dir = out_cine_dir
            elif filename.startswith("lge_"):
                category = "lge"
                out_dir = out_lge_dir
            else:
                continue
            
            print(f"\n  -> Encontrado {category.upper()}: {filename}")
            
            # 2. PREPARAÇÃO (Copia imagem pura sem formatar)
            img_basename = f"img_{filename}"
            target_temp_path = os.path.join(out_dir, img_basename)
            shutil.copy2(file_path, target_temp_path)
            
            try:
                # 3. EXECUÇÃO DA SEGMENTAÇÃO (Sem flags de reformatação! Imagem PURA)
                cmd = [sys.executable, "run_seg.py", "-i", target_temp_path]
                subprocess.run(cmd, check=True)
                
                # 4. RESGATE DINÂMICO
                found_any = False
                orig_no_ext = filename.replace(".nii.gz", "")
                
                if os.path.exists(network_output_dir):
                    generated_files = glob.glob(os.path.join(network_output_dir, f"*{orig_no_ext}*.nii.gz"))
                    
                    for gen_file in generated_files:
                        gen_basename = os.path.basename(gen_file)
                        found_any = True
                        
                        is_seg = "seg" in gen_basename.lower()
                        
                        suffix = ""
                        if "_ED" in gen_basename: suffix = "_ED"
                        elif "_ES" in gen_basename: suffix = "_ES"
                        
                        if is_seg:
                            final_name = f"seg_{orig_no_ext}{suffix}.nii.gz"
                        else:
                            final_name = f"img_{orig_no_ext}{suffix}.nii.gz"
                            
                        final_path = os.path.join(out_dir, final_name)
                        shutil.move(gen_file, final_path)
                        print(f"  ✅ Resgatado: {gen_basename} -> {final_name}")
                        
                if not found_any:
                    print(f"  ⚠️ Aviso: Nenhuma segmentação encontrada para {filename}.")
                    
            except subprocess.CalledProcessError:
                print(f"  ❌ Erro ao tentar segmentar {filename}. O ficheiro foi ignorado.")

    print("\n🎉 Processamento em lote finalizado! Imagens puras processadas e organizadas.")

if __name__ == "__main__":
    process_batch()