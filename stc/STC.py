import numpy as np
from stc.Video import Video
import os
from stc.Models import loadModel
from stc.CustomTransforms import ToGrayScale
import torch
from PIL import Image
from torchvision import transforms

class STC():
    def __init__(self):
        print("create instance of stc ... ")
        self.classes = ['CU', 'ELS', 'LS', 'MS']
        self.vPath = "/data/share/videos/test/"
        self.mPath = "/data/share/pretrained_models/20191226_BasicCNN_Resnet_Imagenet_ExpNum_6/best_model.pth"
        self.DEBUG = True;
        self.results_stc = ""

        if (self.DEBUG == True):
            print("DEBUG MODE activated!")
            self.sbd_results_list = "/data/share/results_sbd/final_shots_all.csv"
            self.debug_results = "/data/share/results_debug/"

    def runStcClassifier(self, shots_per_vid_np=None):
        print("run stc classifier ... ")

        if (type(shots_per_vid_np) == None):
            print("ERROR: you have to set the parameter shots_per_vid_np!")
            exit()

        if(self.DEBUG == True):
            # load shot list from result file
            shots_np = self.loadSbdResults(self.sbd_results_list)
        else:
            shots_np = shots_per_vid_np

        if (len(shots_np) == 0):
            print("ERROR: there must be at least one shot in the list!")
            exit()

        # prepare transformation for cnn model
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((720, 960)),
            transforms.CenterCrop((720, 720)),
            transforms.Resize((128, 128)),
            ToGrayScale(),
            transforms.ToTensor(),
            transforms.Normalize((94.05657 / 255.0, 94.05657 / 255.0, 94.05657 / 255.0),
                                 (57.99793 / 255.0, 57.99793 / 255.0, 57.99793 / 255.0))
        ])

        model = self.loadStcModel(self.mPath, classes=self.classes);

        if (self.DEBUG == True):
            nShots = 3;
        else:
            nShots = len(shots_np)

        results_stc_l = []
        for idx in range(0, nShots):
            #print(shots_np[idx])
            shot_id = int(shots_np[idx][0])
            vid_name = str(shots_np[idx][1])
            start = int(shots_np[idx][2])
            stop = int(shots_np[idx][3]) + 1

            vid_instance = self.loadSingleVideo(os.path.join(self.vPath, vid_name))

            frame_l = []
            for idx in range(start, stop):
                frame = vid_instance.getFrame(idx)
                frame = preprocess(frame)

                if (self.DEBUG == True):
                    tmp_trans = transforms.ToPILImage();
                    frame_pil = tmp_trans(frame)
                    #frame_pil.save(self.debug_results + "/" + str(vid_name) + "_" + str(start) + "_" + str(stop) + "" + str(idx) + ".png")

                frame_l.append(frame)
            tensor_l = torch.stack(frame_l)

            # run classifier
            class_name, nHits = self.runModel(model, tensor_l)

            # prepare results
            print(str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name))
            results_stc_l.append([str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name)])

        results_stc_np = np.array(results_stc_l)

        # export results
        self.exportStcResults(results_stc_np)

    def loadStcModel(self, mPath, classes):
        print("load pre trained model")
        model = loadModel(model_arch="Resnet", classes=classes, pre_trained_path=mPath);
        return model;

    def loadSingleVideo(self, vPath):
        vid_instance = Video()
        vid_instance.load(vPath)
        return vid_instance

    def runModel(self, model, tensor_l):
        input_batch = tensor_l

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        model.eval()
        with torch.no_grad():
            output = model(input_batch)
            preds = output.argmax(1, keepdim=True)
            distr = torch.unique(preds, return_counts=True)
            idx = distr[1].argmax(0, keepdim=True).item()

            class_name = self.classes[distr[0][idx].item()]
            nHits = distr[1][idx].item()

            return class_name, nHits;

    def loadSbdResults(self, sbd_results_path):
        # open sbd results
        fp = open(sbd_results_path, 'r')
        lines = fp.readlines()
        lines = lines[1:]

        lines_n = []
        for i in range(0, len(lines)):
            line = lines[i].replace('\n', '')
            line_split = line.split(';')
            lines_n.append([line_split[0], os.path.join(line_split[1]), line_split[2], line_split[3]])
        lines_np = np.array(lines_n)
        #print(lines_np.shape)

        return lines_np

    def exportStcResults(self, stc_results_np: np.ndarray):
        print("export results to csv!")

        if(len(stc_results_np) == 0):
            print("ERROR: numpy is empty")
            exit()

        # open stc resutls file
        if (self.DEBUG == True):
            fp = open(self.debug_results + "/" + "results_stc.csv", 'w')
        else:
            fp = open(self.results_stc + "/" + "results_stc.csv", 'w')
        header = "vid_name;shot_id;start;end;stc"
        fp.write(header + "\n")

        for i in range(0, len(stc_results_np)):
            tmp_line = str(stc_results_np[i][0])
            for c in range(1, len(stc_results_np[i])):
                tmp_line = tmp_line + ";" + stc_results_np[i][c]
            fp.write(tmp_line + "\n")





