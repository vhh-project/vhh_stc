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
    results_path = "/data/ext/VHH/release_results/release_v1_3_0/vhh_core/results_part1/core/single/"
    dst_path = "/caa/Projects02/vhh/private/database_nobackup/stc_vhh_mmsi_1_3_0/extracted_frames_part1/"
    video_path = "/data/ext/VHH/release_results/release_v1_3_0/vhh_core/videos_part1/"

    results_file_list = os.listdir(results_path)
    print(results_file_list)

    for file in results_file_list:
        shots_np = stc_instance.loadStcResults(results_path + file)
        stc_instance.extract_frames_per_shot(shots_per_vid_np=shots_np,
                                             number_of_frames=1,
                                             dst_path=dst_path,
                                             video_path=video_path
                                             )

