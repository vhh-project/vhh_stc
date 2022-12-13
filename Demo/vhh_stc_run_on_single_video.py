from vhh_stc.STC import STC
import os

config_file = "./config/config_vhh_test.yaml"
stc_instance = STC(config_file)

if(stc_instance.config_instance.debug_flag == True):
    print("DEBUG MODE activated!")
    sbd_results_file = stc_instance.config_instance.sbd_results_path
    shots_np = stc_instance.loadSbdResults(sbd_results_file)
    max_recall_id = int(shots_np[0][0].split('.')[0])
    stc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)
else:
    results_path = "/data/share/fjogl/vhh_stc_data/sbd_results/"

    results_file_list = os.listdir(results_path)
    print(f"Nr of films to run on {len(results_file_list)}")
    for file in results_file_list:
        shots_np = stc_instance.loadSbdResults(results_path + file)
        
        if len(file) > 8:
            max_recall_id = int(file[0:4])
        else:   
            max_recall_id = int(file.split('.')[0])

        print(shots_np, "\n", max_recall_id)
        stc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)





