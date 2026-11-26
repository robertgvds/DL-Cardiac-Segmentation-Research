import os
import time
import math
import glob
import numpy as np
import nibabel as nib
from image_utils import rescale_intensity

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class SASegmenter:
    """
    Classe Orientada a Objetos para segmentação simplificada 
    de imagens de Ressonância Magnética Cardíaca (Short-Axis CINE e LGE).
    """
    def __init__(self, data_dir, model_path=None, gpu_id='0', image_pattern='*sa*.nii.gz'):
        """
        :param data_dir: Diretório raiz contendo as pastas dos pacientes (ex: 1, 2, 3...)
        :param model_path: Caminho para os pesos do modelo treinado. (Pode ser None se for usar Validação Cruzada)
        :param gpu_id: ID da GPU a ser utilizada
        :param image_pattern: Padrão do nome do arquivo da imagem.
        """
        self.data_dir = data_dir
        self.model_path = model_path
        self.image_pattern = image_pattern
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    def _pad_image(self, image, X, Y):
        X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
        x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
        x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
        
        image_padded = np.pad(
            image, 
            ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 
            'constant'
        )
        return image_padded, x_pre, y_pre

    def segment_directory(self):
        """Usa um modelo único para segmentar todos os pacientes da pasta."""
        if not self.model_path:
            raise ValueError("model_path não foi fornecido na inicialização.")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(f"[*] Carregando o modelo de: {self.model_path} ...")
            saver = tf.train.import_meta_graph(f'{self.model_path}.meta')
            saver.restore(sess, self.model_path)
            print("[*] Modelo carregado com sucesso!\n")

            patients = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
            for patient in patients:
                patient_dir = os.path.join(self.data_dir, patient)
                search_path = os.path.join(patient_dir, self.image_pattern)
                found_images = glob.glob(search_path)

                if not found_images:
                    print(f"[-] Nenhuma imagem '{self.image_pattern}' encontrada na pasta {patient}. Ignorando.")
                    continue

                image_path = found_images[0]
                filename = os.path.basename(image_path)
                name_without_ext = filename.replace('.nii.gz', '').replace('.nii', '')
                output_path = os.path.join(patient_dir, f"{name_without_ext}_seg.nii.gz")

                print(f"[+] Processando paciente '{patient}' | Imagem: {filename} ...")
                self._process_single_image(sess, image_path, output_path)

        print("\n[*] Processamento concluído para todos os pacientes!")

    def run_cv_testing(self, fold_to_patient_dict, models_base_dir, iteration=2000):
        """
        Executa a validação cruzada, subindo um modelo específico para cada paciente de teste.
        :param fold_to_patient_dict: Dicionário ex: {1: "Patient_1", 2: "Patient_2"}
        :param models_base_dir: Pasta raiz onde estão as subpastas 1, 2, 3... dos folds
        :param iteration: Número da iteração alvo (ex: 2000)
        """
        print("\n[*] Iniciando Inferência de Validação Cruzada K-Fold...")
        
        for fold, patient in fold_to_patient_dict.items():
            print(f"\n{'='*50}")
            print(f"[*] FOLD {fold} | Avaliando Paciente: {patient}")
            
            patient_dir = os.path.join(self.data_dir, patient)
            search_path = os.path.join(patient_dir, self.image_pattern)
            found_images = glob.glob(search_path)

            if not found_images:
                print(f"[-] Imagem não encontrada para o paciente {patient}. Pulando.")
                continue

            image_path = found_images[0]
            filename = os.path.basename(image_path)
            name_without_ext = filename.replace('.nii.gz', '').replace('.nii', '')
            output_path = os.path.join(patient_dir, f"{name_without_ext}_pred_fold{fold}.nii.gz")
            
            model_prefix = os.path.join(models_base_dir, str(fold), f"FCN_sa_{iteration}")

            # LIMPA A MEMÓRIA PARA O PRÓXIMO MODELO (Evita colisão de variáveis)
            tf.reset_default_graph()
            
            with tf.Session() as sess:
                print(f"[*] Carregando pesos do modelo: Fold {fold}")
                saver = tf.train.import_meta_graph(f'{model_prefix}.meta')
                saver.restore(sess, model_prefix)
                
                print(f"[+] Segmentando {filename}...")
                self._process_single_image(sess, image_path, output_path)

        print(f"\n{'='*50}")
        print("[*] Validação Cruzada Concluída!")

    def _process_single_image(self, sess, image_path, output_path):
        start_time = time.time()
        nim = nib.load(image_path)
        image = nim.get_fdata()
        is_3d = False
        
        if image.ndim == 3:
            X, Y, Z = image.shape
            T = 1
            image = np.expand_dims(image, axis=-1)
            is_3d = True
        elif image.ndim == 4:
            X, Y, Z, T = image.shape
        else:
            print(f"    [Erro] A imagem possui {image.ndim} dimensões. O esperado é 3D ou 4D.")
            return

        image = rescale_intensity(image, (1, 99))
        pred = np.zeros(image.shape)
        image_padded, x_pre, y_pre = self._pad_image(image, X, Y)

        for t in range(T):
            image_fr = image_padded[:, :, :, t]
            image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
            image_fr = np.expand_dims(image_fr, axis=-1)

            prob_fr, pred_fr = sess.run(
                ['prob:0', 'pred:0'],
                feed_dict={'image:0': image_fr, 'training:0': False}
            )

            pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
            pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
            pred[:, :, :, t] = pred_fr

        if is_3d:
            pred = np.squeeze(pred, axis=-1)
            
        pred = pred.astype(np.int16)
        nim2 = nib.Nifti1Image(pred, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        nib.save(nim2, output_path)

        elapsed = time.time() - start_time
        print(f"    -> Segmentação salva em: {os.path.basename(output_path)} ({elapsed:.2f}s)")