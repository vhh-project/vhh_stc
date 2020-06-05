import os
import numpy as np
import cv2

#####################################################################
##           CONFIGURATION 
#####################################################################

vidName = "EF-NS_004_OeFM.mp4"          #### --------> FIXME
className = "CU"                        #### --------> FIXME   ### CU, LS, MS, ELS
vPath = "D:\\Praktikanten\\videos_converted_new\\"
aPath = "D:\\Praktikanten\\videos_converted_new\\"
DELAY_TIME = 1000                       ####  delay time for playing movie in milliseconds

#####################################################################


#####################################################################
##           MAIN PART 
#####################################################################

cap = cv2.VideoCapture(vPath + vidName);
 
if (cap.isOpened()== False): 
   print("Error opening video stream or file")
  
frame_rate_orig = cap.get(cv2.CAP_PROP_FPS);
number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT);
length_of_video = round(number_of_frames / frame_rate_orig, 2);


print("---------------------------------------");
print("video name: " + str(vidName));
print("number of frames: " + str(number_of_frames));
print("length of video [sec]: " + str(length_of_video));
print("original framerate of video: " + str(frame_rate_orig))


fp = open(aPath + "/" + str(vidName.split('.')[0]) + "_" + str(className) + ".csv", 'r');
lines = fp.readlines()
lines = lines[1:]
fp.close();

print(lines)

for i in range(0, len(lines)):
   tmp_line = lines[i];
   tmp_line = tmp_line.replace('\n', '');
   frm_pos = int(tmp_line.split(';')[2]);
   class_name = str(tmp_line.split(';')[3]);
   print("entry id: " + str(i+1) + ", movie_name: " + str(frm_pos) + ", frame_position: " + str(frm_pos) + ", class_name: " + str(class_name))
   cap.set(cv2.CAP_PROP_POS_FRAMES, frm_pos)
   ret, frame = cap.read();

   if(ret == True ): 
      # Display the resulting frame
      cv2.imshow('Frame',frame)
      # Break the loop
   else: 
      break
 
   # Press q on keyboard to exit
   if cv2.waitKey(DELAY_TIME) & 0xFF == ord('q'):
      break   

	
  
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()