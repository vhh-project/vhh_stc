import numpy as np


class Shot:

    def __init__(self, sid, movie_name, start_pos, end_pos):
        #print("create instance of shot ...");
        self.sid = sid;
        self.movie_name = movie_name;
        self.start_pos = start_pos;
        self.end_pos = end_pos;

    def convert2String(self):
        tmp_str = str(self.sid) + ";" + str(self.movie_name) + ";" + str(self.start_pos) + ";" + str(self.end_pos);
        return tmp_str;

    def printShotInfo(self):
        print("------------------------")
        print("shot id: " + str(self.sid));
        print("movie name: " + str(self.movie_name));
        print("start frame: " + str(self.start_pos));
        print("end frame: " + str(self.end_pos));