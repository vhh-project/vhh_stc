
from vhh_stc.STC import STC
import numpy as np
import os

config_file = "/home/dhelm/VHH_Develop/installed_pkg/vhh_stc/config/config_vhh_test.yaml"
stc_instance = STC(config_file)


results_path = "/data/share/maxrecall_vhh_mmsi/videos/results/sbd/final_results/"
results_file_list = os.listdir(results_path)
print(results_file_list)
shots_np = stc_instance.loadSbdResults(results_path + results_file_list[0])

print(shots_np)

max_recall_id = 99

stc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id);



