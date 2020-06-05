import os
import numpy as np
import cv2

#####################################################################
##           CONFIGURATION 
#####################################################################

aName = "annotations_train.csv"          #### --------> FIXME
className = "ELS"                          #### --------> FIXME   ### CU, LS, MS, ELS
vPath = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/videos_holocaust/train/"
aPath = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/stc/20191120/jakob/"
dstPath = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/stc/20191120/jakob/train/"
dim = (960, 720);                        #### width, height  (960, 720)
#####################################################################


#####################################################################
##           MAIN PART 
#####################################################################


fp = open(aPath + "/" + aName, 'r');
lines = fp.readlines()
lines = lines[1:]
fp.close();

print("create directory ... ");
if( not os.path.exists(dstPath + str(className))):
    os.mkdir(dstPath + str(className))

for i in range(0, len(lines)):
   tmp_line = lines[i];
   tmp_line = tmp_line.replace('\n', '');
   vidName = str(tmp_line.split(';')[1]);
   vidName = vidName.split('_')
   vidName = '_'.join(vidName[:-1]);
   frm_pos = int(tmp_line.split(';')[2]);
   class_name = str(tmp_line.split(';')[3]);
   
   
   
   if(class_name == className):
      print("entry id: " + str(i+1) + ", movie_name: " + str(vidName) + ", frame_position: " + str(frm_pos) + ", class_name: " + str(class_name))
      #print(vPath + vidName + ".mp4")
      cap = cv2.VideoCapture(vPath + "/" + vidName + ".mp4");
      cap.set(cv2.CAP_PROP_POS_FRAMES, frm_pos)  
      ret, frame = cap.read();
      frame = cv2.resize(frame, (dim[0], dim[1]));
      #print(ret)
      #exit()
      if(ret == True ): 
         print("save ...")
         fName = str(vidName) + "_" + str(frm_pos);
         cv2.imwrite(dstPath + "/" + str(className) + "/" + fName + ".png", frame)
         print("save ...")
         # Break the loop
      #else: 
      #   continue;
	  
      # When everything done, release the video capture object
      cap.release()