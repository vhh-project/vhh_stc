import os
import numpy as np
import cv2

#####################################################################
##           CONFIGURATION
#####################################################################

classNames = ["CU", "ELS", "I", "LS", "MS", "NA"]  #### --------> FIXME   ### "CU", "ELS", "I", "LS", "MS", "NA"
db_path = "/data/share/datasets/vhh_mmsi_test_db/"
vPath = db_path + "/videos/"
aPath = db_path + "/annotations/stc/final_results/"
dstPath = "/data/share/datasets/vhh_mmsi_test_db/test/"
dim = (960, 720)  #### width, height  (960, 720)
nFrames = 5
#####################################################################

def load_results(filename):
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    data_l = []
    for line in lines[1:]:
        line = line.replace('\n', '')
        line = line.replace('\\', '/')
        #print(line)
        line_split = line.split(';')
        #print(line_split)
        data_l.append([line_split[0], line_split[2], line_split[3], line_split[4]])
    #data_np = np.array(gt_annotation_list)
    return data_l


#####################################################################
##           MAIN PART
#####################################################################

# load gt annotations
gt_file_list = os.listdir(aPath)
gt_file_list.sort()
print(gt_file_list)

all_gt_data_l = []
for file in gt_file_list:
    print(file)
    gt_l = load_results(aPath + file)
    #print(gt_np)
    all_gt_data_l.extend(gt_l)

all_gt_data_np = np.array(all_gt_data_l)
print(all_gt_data_np)

print("create directory structure ... ")
for className in classNames:
    if (not os.path.exists(dstPath + str(className))):
        os.mkdir(dstPath + str(className))

# loop through shots
for i in range(0, len(all_gt_data_np)):
    vidName = str(all_gt_data_np[i][0])
    start_pos = int(all_gt_data_np[i][1])
    end_pos = int(all_gt_data_np[i][2])
    class_name = str(all_gt_data_np[i][3])

    print("#####################")
    print(vidName)
    print(start_pos)
    print(end_pos)
    print(class_name)

    print("line id: " + str(i) + ", movie_name: " + str(vidName) + ", start_position: " + str(start_pos) +
          ", end_position: " + str(end_pos) + ", class_name: " + str(class_name))
    # print(vPath + vidName + ".mp4")

    cap = cv2.VideoCapture(vPath + "/" + vidName)

    for j in range(0, nFrames):
        if (int(start_pos + j) >= end_pos):
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos + j)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (dim[0], dim[1]))
        # print(ret)
        # exit()
        if (ret == True):
            #print("save ...")
            fName = str(vidName.split('.')[0]) + "_" + str(i) + "_" + str(start_pos + j)
            cv2.imwrite(dstPath + "/" + str(class_name) + "/" + fName + ".png", frame)
        # Break the loop
    # else:
    #   continue;

    # When everything done, release the video capture object
    cap.release()