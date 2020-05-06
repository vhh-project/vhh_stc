import numpy as np
from stc.Video import Video
import os
from stc.Models import loadModel
from stc.CustomTransforms import ToGrayScale
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from stc.utils import *
from stc.Configuration import Configuration
import cv2
import json


class STC():
    """
        A class for retrieving values from config files.

    """

    def __init__(self, config_file: str):
        """

        :param config_file:
        """
        print("create instance of stc ... ")

        if (config_file == ""):
            printCustom("No configuration file specified!", STDOUT_TYPE.ERROR)
            exit()

        self.config_instance = Configuration(config_file);
        self.config_instance.loadConfig();

        if (self.config_instance.debug_flag == True):
            print("DEBUG MODE activated!")
            self.debug_results = "/data/share/maxrecall_vhh_mmsi/videos/results/stc/develop/"

    def runOnSingleVideo(self, shots_per_vid_np=None, max_recall_id=-1):
        """
        Get a value from the config file.

        :param section: the section that contains the entry
        :param var: the variable that holds the value
        :return: the retrieved value
        """

        print("run stc classifier on single video ... ")

        if (type(shots_per_vid_np) == None):
            print("ERROR: you have to set the parameter shots_per_vid_np!")
            exit()

        if (max_recall_id == -1 or max_recall_id == 0):
            print("ERROR: you have to set a valid max_recall_id [1-n]!")
            exit()

        if(self.config_instance.debug_flag == True):
            # load shot list from result file
            shots_np = self.loadSbdResults(self.config_instance.sbd_results_path)
        else:
            shots_np = shots_per_vid_np

        if (len(shots_np) == 0):
            print("ERROR: there must be at least one shot in the list!")
            exit()

        with open(self.config_instance.path_pre_trained_model + "/" + "/experiment_notes.json", 'r') as json_file:
            param_dict = json.load(json_file)

        model_arch = param_dict['expNet']
        classes = param_dict['classes']
        pre_trained_weights = self.config_instance.path_pre_trained_model + "/best_model.pth"

        #model = self.loadStcModel(self.config_instance.path_pre_trained_model, classes=self.config_instance.class_names)
        model = loadModel(model_arch=model_arch, classes=classes, pre_trained_path=pre_trained_weights)

        if (self.config_instance.debug_flag == True):
            num_shots = 3
        else:
            num_shots = len(shots_np)

        vid_name = shots_np[0][1]
        vid_instance = self.loadSingleVideo(os.path.join(self.config_instance.path_videos, vid_name))

        # prepare transformation for cnn model
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(vid_instance.height), vid_instance.width)),
            transforms.CenterCrop((int(vid_instance.height), int(vid_instance.height))),
            transforms.Resize(self.config_instance.resize_dim),
            ToGrayScale(),
            transforms.ToTensor(),
            transforms.Normalize((self.config_instance.mean_values[0] / 255.0,
                                  self.config_instance.mean_values[1] / 255.0,
                                  self.config_instance.mean_values[2] / 255.0),
                                 (self.config_instance.std_dev[0] / 255.0,
                                  self.config_instance.std_dev[1] / 255.0,
                                  self.config_instance.std_dev[2] / 255.0))
        ])

        # read all frames of video
        cap = cv2.VideoCapture(self.config_instance.path_videos + "/" + vid_name)
        frame_l = []
        cnt = 0
        while (True):
            cnt = cnt + 1
            ret, frame = cap.read()
            # print(cnt)
            # print(ret)
            # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (ret == True):
                frame = preprocess(frame)
                frame_l.append(frame)
            else:
                break;
        # exit()

        all_tensors_l = torch.stack(frame_l)
        frame_cnt = 0
        results_stc_l = []
        for idx in range(0, num_shots):
            #print(shots_np[idx])
            shot_id = int(shots_np[idx][0])
            vid_name = str(shots_np[idx][1])
            start = int(shots_np[idx][2])
            stop = int(shots_np[idx][3])

            shot_tensors = all_tensors_l[start:stop+1, :, :, :]

            # run classifier
            class_name, nHits, all_preds_np = self.runModel(model, shot_tensors)

            if(self.config_instance.save_raw_results == 1):
                #print("save intermediate/raw results ... ")

                # prepare folder structure
                video_raw_result_path = os.path.join(self.config_instance.path_raw_results, str(vid_name.split('.')[0]))
                createFolder(video_raw_result_path)
                shot_result_path = os.path.join(video_raw_result_path, str(shot_id))
                createFolder(shot_result_path)

                # save class distribution
                indices, distr = np.unique(all_preds_np, return_counts=True)
                class_distr_per_shot = np.zeros(len(self.config_instance.class_names)).astype('uint8')
                class_distr_per_shot[indices] = distr
                #names = self.config_instance.class_names[indices]
                #print(names)
                plotClassDistribution(file_name=shot_result_path + "/class_distr_" + str(shot_id),
                                      file_extension="pdf",
                                      class_distr_name=self.config_instance.class_names,
                                      class_distr_data=class_distr_per_shot)

                # save predicition per frame for whole video
                raw_results_per_frame_csv = "results_per_frame.csv"
                for r in range(0, len(all_preds_np)):
                    frame_cnt = frame_cnt + 1
                    entries_l = [frame_cnt, all_preds_np[r], self.config_instance.class_names[all_preds_np[r]]]
                    csvWriter(dst_folder=video_raw_result_path, name=raw_results_per_frame_csv, entries_list=entries_l)

                # save n frames of each shot in separate folders
                n = self.config_instance.number_of_frames_per_shot
                shot_frames_np = np.array(shot_tensors)
                number_of_frames = len(shot_frames_np.shape)
                center_idx = int(number_of_frames / 2)
                a = center_idx - n
                if(a <= 0): a = 0
                if(a >= len(shot_frames_np)): a = len(shot_frames_np)
                b = center_idx + n
                if (b <= 0): b = 0
                if (b >= len(shot_frames_np)): b = len(shot_frames_np)

                for j in range(a, b):
                    #print(shot_frames_np[j].transpose(1, 2, 0).shape)
                    cv2.imwrite(shot_result_path + "/" + str(shot_id) + "_ " + str(j) + ".png",
                                shot_frames_np[j].transpose(1, 2, 0))

            # prepare results
            print(str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name))
            results_stc_l.append([str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name)])

        results_stc_np = np.array(results_stc_l)

        # export results
        self.exportStcResults(str(max_recall_id), results_stc_np)

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

        # prepare pytorch dataloader
        dataset = data.TensorDataset(input_batch)  # create your datset

        inference_dataloader = data.DataLoader(dataset, batch_size=self.config_instance.batch_size)  # create your dataloader

        preds_l = []
        for i, inputs in enumerate(inference_dataloader):
            input_batch = inputs[0]
            input_batch = Variable(input_batch)

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            model.eval()
            with torch.no_grad():
                output = model(input_batch)
                preds = output.argmax(1, keepdim=True)
                preds_l.extend(preds.detach().cpu().numpy().flatten())

        preds_np = np.array(preds_l).flatten()
        indices, distr = np.unique(preds_np, return_counts=True)

        idx = distr.argmax(0)
        #print(idx)

        class_name = self.config_instance.class_names[idx]
        nHits = distr[idx]
        #print(class_name)
        #print(nHits)

        return class_name, nHits, preds_np

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

    def exportStcResults(self, fName, stc_results_np: np.ndarray):
        print("export results to csv!")

        if(len(stc_results_np) == 0):
            print("ERROR: numpy is empty")
            exit()

        # open stc resutls file
        if (self.config_instance.debug_flag == True):
            fp = open(self.debug_results + "/" + fName + ".csv", 'w')
        else:
            fp = open(self.config_instance.path_final_results + "/" + fName + ".csv", 'w')
        header = "vid_name;shot_id;start;end;stc"
        fp.write(header + "\n")

        for i in range(0, len(stc_results_np)):
            tmp_line = str(stc_results_np[i][0])
            for c in range(1, len(stc_results_np[i])):
                tmp_line = tmp_line + ";" + stc_results_np[i][c]
            fp.write(tmp_line + "\n")





