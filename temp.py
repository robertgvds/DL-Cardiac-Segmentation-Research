import os

input_file = "./input/patient.mat"
filename = "patient.mat"
patient_id = "patient"

os.system('CUDA_VISIBLE_DEVICES=0 python3 models/ukbb_cardiac/common/deploy_network.py --seq_name {0} --data_dir {1} '
              '--model_path trained_model/FCN_sa'.format(filename, input_file))