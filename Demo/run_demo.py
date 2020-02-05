
from stc.STC import STC
import numpy as np

a = STC();

sbd_results_list = "/data/share/results_sbd/final_shots_all.csv"
shots_np = a.loadSbdResults(sbd_results_list)
print(shots_np)

a.runStcClassifier(shots_per_vid_np=shots_np);



