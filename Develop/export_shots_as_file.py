from vhh_stc.STC import STC
import os

config_file = "./config/config_vhh_test.yaml"
stc_instance = STC(config_file)

results_path = "/data/ext/VHH/release_results/release_v1_3_0/vhh_core/results_part1/stc/final_results/"
#results_path = "./debug/"
dst_path = "/data/share/datasets/demo_shots/"

results_file_list = os.listdir(results_path)
results_file_list.sort()
print(results_file_list)

for file in results_file_list[5:6]:
    shots_np = stc_instance.loadStcResults(results_path + file)
    max_recall_id = int(file.split('.')[0])
    print(shots_np)
    stc_instance.export_shots_as_file(shots_np=shots_np, dst_path=dst_path)
