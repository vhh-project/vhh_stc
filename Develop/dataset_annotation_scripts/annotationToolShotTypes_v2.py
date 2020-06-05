import os
import numpy as np
import cv2
import glob

#####################################################################
##           CONFIGURATION 
#####################################################################
dbSet = "val"
framePath = "P:\\private\\database_nobackup\\VHH_datasets\\generated\\stc\\20191203\\extractedFrames\\" + str(dbSet) + "\\"
aPath = "P:\\private\\database_nobackup\\VHH_datasets\\generated\\stc\\20191203\\annotations\\"
aName = "annotations_" + str(dbSet) + ".csv"
dim = (704, 480);                        #### width, height  (960, 720)(480, 360) (704, 480) (352, 240)

#####################################################################


#####################################################################
##           METHODS 
#####################################################################

def openCSVFile(filepath):
   print("open or create csv ...");
   print("path: " + str(filepath))
   
   status = os.path.exists(filepath);
   if(status == False):
      print("File does not exist. Create new one.")
      print(filepath)
      fp = open(filepath, 'w');
      header = "aID" + ";" + "vidName" + ";" + "frame_pos" + ";" + "class_name";
      fp.write(header + '\n')
      fp.close();
   elif(status == True):
      print("File already exist.")
	
	   
def addToCSV(filepath, line):
   print("add annotation to csv ... ")
   fp = open(filepath, 'a+');
   fp.write(line + '\n')
   fp.close();

def getClassDistribution(filepath):
   print("get class distribution ... ")
   fp = open(filepath, 'r');
   lines = fp.readlines()
   fp.close();
   
   #print(lines)
   
   if(len(lines) == 1):
      print("There is no entry in this file.");
      return 0, 0;
	  
   class_list = []
   for i in range(1, len(lines)):	  
      tmp = lines[i].replace('\n', '');
      tmp_split = tmp.split(';');
      class_name = tmp_split[3]
      class_list.append(class_name)
   classes_np = np.array(class_list)
   names, cnts = np.unique(classes_np, return_counts=True)
   print("Class Distribution")
   print("------------------")
   for i in range(0, len(cnts)):
      print(names[i] + str(": ") + str(cnts[i]))

   
def readIdOfLastEntryFromCSV(filepath):
   print("read last entries of csv ... ")
   fp = open(filepath, 'r');
   lines = fp.readlines()
   fp.close();
   print(len(lines))
   #print(lines[-1])
   if(len(lines) == 1):
      print("There is no entry in this file.");
      return 0, 0;
	  
   tmp = lines[-1].replace('\n', '');
   tmp_split = tmp.split(';');
   aID_tmp = int(tmp_split[0]);
   frame_name = tmp_split[1];
   lines.append(tmp);
   return aID_tmp, frame_name;

def checkDuplicateEntries(filepath, newEntry):
   fp = open(filepath, 'r');
   lines = fp.readlines()
   fp.close();
   print(newEntry)
   #print(lines[-1])
   if(len(lines) == 1):
      print("There is no entry in this file.");
      return False;
   
   lines_new = [];
   for i in range(0, len(lines)):	  
      tmp = lines[i].replace('\n', '');
      tmp_split = tmp.split(';');
	  
      #aID_tmp = int(tmp_split[0]);
      vidName_tmp = str(tmp_split[1]);
      frm_pos_tmp = tmp_split[2];
      className_tmp = str(tmp_split[3]);
      lines_new.append([vidName_tmp, frm_pos_tmp, className_tmp]);
	  
   lines_new_np = np.array(lines_new);
   newEntry_np = np.array([newEntry.split(';')[1], newEntry.split(';')[2], newEntry.split(';')[3]])
   
   idx1 = np.where(newEntry_np[0] == lines_new_np[:, :1])[0];
   idx2 = np.where(newEntry_np[1] == lines_new_np[:, 1:2])[0];
   print(lines_new_np[idx1])
   print(newEntry_np)
   print(len(idx1))
   print(len(idx2))
   if(len(idx1) > 0 and len(idx2) > 0):
      return True;
   else:
      return False;
   
   #return True;

def getFrame(cap, sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    ret, frame = cap.read();
    return ret, frame;

def overlapMask(frame):
    alpha = 1.0
    overlay = frame.copy()
    output = frame.copy()
	
    offset_x = int(frame.shape[0] / 5) # 360 --> 72
    offset_y = int(frame.shape[1] / 5) # 480 --> 96
    #print(offset_x)
    #print(offset_y)
    mask = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), np.uint8)
    #mask[90:(720-90), 120:(960-120), :] = 255;
    mask[0:frame.shape[0], offset_x:(frame.shape[1]-offset_x), :] = 255;
	#cv2.imwrite("P:\\private\\database_nobackup\\VHH_datasets\\scripts\\mask.png", mask)
    dst = cv2.bitwise_and(frame, mask)
    output = cv2.addWeighted(dst, 0.5, output, 0.5, 0)
    res = output
    return res;

#####################################################################
##           MAIN PART 
#####################################################################

openCSVFile(aPath + "\\" + aName);
aID_last, frame_name = readIdOfLastEntryFromCSV(aPath + "\\" + aName);
print(aID_last)
print(frame_name)

# load all frames from folder
os.chdir(framePath)
filename_list = glob.glob("*.png")
filename_list.sort()
print("number_of_frames: " + str(len(filename_list)))

class_name = "-1"
number_of_frames = len(filename_list);
step_cnt = aID_last;
filename = str(filename_list[step_cnt]);
filename = filename.split('.')[0]
frame = cv2.imread(framePath + "\\" + filename + ".png");
frame = cv2.resize(frame, dim);
print(frame.shape)
print(step_cnt)

alock = False;
aID = aID_last + 1;

print("---------------------------------------");
print("video name: " + str(filename));
print("number of frames: " + str(number_of_frames));

while(True):
    k = cv2.waitKey(25)

    # 13 enter
    # 32 space
    # 8 backspace
    frame = overlapMask(frame)
    cv2.imshow('Frame', frame)
	
    # Press q on keyboard to exit
    if k & 0xFF == ord('q'):
       break
  
    # Press space on keyboard to step frame by frame backward
    if(k == 32): 
       if(step_cnt >= number_of_frames):
          step_cnt = number_of_frames;
       else:
          step_cnt = step_cnt + 1;
		  
       filename = str(filename_list[step_cnt]);
       #print(filename)
       filename = filename.split('.')[0]
       #print(filename)
       frame_number = filename.split('.')[0].split('_')[-1];
       frame = cv2.imread(framePath + "\\" + filename + ".png");
       #print(frame.shape)
       frame = cv2.resize(frame, (dim[0], dim[1]));
       #print(frame.shape)
       #print(step_cnt)
       print("---------------------------------------");
       print("video name: " + str(filename));
       print("step_cnt: " + str(step_cnt));
       #print("last entry ID: " + str(aID_last))
       #print("last frame position: " + str(frm_pos_last))
       getClassDistribution(aPath + "\\" + aName)
       
    if(k == 8): 
       if(step_cnt <= 0):
          step_cnt = 0;
       else:
          step_cnt = step_cnt - 1;
		  
       filename = str(filename_list[step_cnt]);
       filename = filename.split('.')[0]
       frame_number = filename.split('.')[0].split('_')[-1];
       frame = cv2.imread(framePath + "\\" + filename + ".png");
       frame = cv2.resize(frame, (dim[0], dim[1]));
       #print(frame.shape)
       #print(step_cnt)
       print("---------------------------------------");
       print("video name: " + str(filename));
       print("step_cnt: " + str(step_cnt));
       #print("number of frames: " + str(number_of_frames));
       #print("last entry ID: " + str(aID_last))
       #print("last frame position: " + str(frm_pos_last))
       getClassDistribution(aPath + "\\" + aName)

    # Press enter on keyboard to save annotation to csv
    if(k & 0xFF == ord('1')):
        print("switch to class: ELS")
        class_name = "ELS"
    if(k & 0xFF == ord('2')):
        print("switch to class: LS")
        class_name = "LS"
    if(k & 0xFF == ord('3')):
        print("switch to class: MS")
        class_name = "MS"
    if(k & 0xFF == ord('4')):
        print("switch to class: CU")
        class_name = "CU"
    if(k & 0xFF == ord('5')):
        print("switch to class: NONE")
        class_name = "NONE"
       
    # Press enter on keyboard to save annotation to csv
    if(k == 13):
       if(class_name != "-1"):
          filename = str(filename_list[step_cnt]);
          filename = filename.split('.')[0]
          frame_number = filename.split('.')[0].split('_')[-1];
          aID = step_cnt;
          entry = str(aID) + ";" + filename + ";" + str(frame_number) + ";" + str(class_name);
          alock = checkDuplicateEntries(aPath + "\\" + aName, entry);
          #print(alock)
          if(alock == False):
             aID = aID + 1;
             #entry = str(aID) + ";" + aName + ";" + str(frm_pos) + ";" + str(className);
             addToCSV(aPath + "\\" + aName, entry);
          else:
             print("Annotation already exist.")
          # reset class 
          class_name = "-1";
       else:
          print("Select valid classname first!");	   