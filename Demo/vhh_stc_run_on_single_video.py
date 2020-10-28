from vhh_stc.STC import STC
import os

config_file = "/caa/Homes01/dhelm/working/vhh/develop/vhh_stc/config/config_vhh_test.yaml"
stc_instance = STC(config_file)

results_path = "/data/share/maxrecall_vhh_mmsi/release/videos/results/sbd/final_results/"
results_file_list = os.listdir(results_path)
print(results_file_list)

for file in results_file_list:
    print(file)
    shots_np = stc_instance.loadSbdResults(results_path + file)
    print(shots_np)
    max_recall_id = int(file.split('.')[0])
    stc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)




