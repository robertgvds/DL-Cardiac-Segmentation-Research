import argparse
from sa_segmenter import SASegmenter

def main():
    # ==========================================
    # 1. DEFINIÇÃO VIA VARIÁVEIS (CONSTANTES)
    # ==========================================
    DEFAULT_DATA_DIR = "./input/dataset_final" 
    DEFAULT_MODEL_PATH = "./segmentation/models/trained_model/FCN_sa" 
    
    # ==========================================
    # 2. DEFINIÇÃO VIA ARGUMENTOS (TERMINAL)
    # ==========================================
    parser = argparse.ArgumentParser(description="Segmentação de Imagens SA (Short-Axis)")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help="Diretório onde estão as pastas dos pacientes.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help="Caminho para o arquivo do modelo treinado (sem .meta).")
    parser.add_argument('--gpu', type=str, default='0',
                        help="ID da placa de vídeo (GPU).")
    
    args = parser.parse_args()

    # Inicializa o segmentador passando as configurações
    segmenter = SASegmenter(
        data_dir=args.data_dir, 
        model_path=args.model_path, 
        gpu_id=args.gpu,
        image_pattern='*sa*.nii.gz'
    )

    # Roda o processamento
    segmenter.segment_directory()

if __name__ == '__main__':
    main()