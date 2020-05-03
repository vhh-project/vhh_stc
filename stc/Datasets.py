import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import h5py as hf
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import cv2

class ToGrayScale(object):
    def __call__(self, frame):
        frame = np.asarray(frame)
        # print(type(frame))
        # print(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray,(224, 224))
        frame_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # hist_cl1 = cv2.calcHist([cl1], [0], None, [256], [0, 256])

        frame_gray = Image.fromarray(frame_gray)
        return frame_gray

    def __repr__(self):
        return self.__class__.__name__ + 'toGrayscale'


def loadDatasetFromFolder(path="", batch_size=64):
    if (path == "" or path == None):
        print("ERROR: you must specifiy a valid dataset path!")
        exit();

    # Datasets from folders
    traindir = path + "/train/"
    validdir = path + "/val/"
    testdir = path + "/test/"

    # Number of subprocesses to use for data loading
    num_workers = 1

    # Percentage of training set to use as validation
    n_valid = 0.2

    # Convert data to a normalized torch.FloatTensor
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((720, 960)),
        transforms.CenterCrop((720, 720)),
        transforms.Resize((128, 128)),
        ToGrayScale(),
        transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        transforms.RandomVerticalFlip(),  # randomly flip and rotate
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize((94.05657 / 255.0, 94.05657 / 255.0, 94.05657 / 255.0),
                             (57.99793 / 255.0, 57.99793 / 255.0, 57.99793 / 255.0))
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((720, 960)),
        transforms.CenterCrop((720, 720)),
        transforms.Resize((128, 128)),
        ToGrayScale(),
        transforms.ToTensor(),
        transforms.Normalize((94.05657 / 255.0, 94.05657 / 255.0, 94.05657 / 255.0),
                             (57.99793 / 255.0, 57.99793 / 255.0, 57.99793 / 255.0))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((720, 960)),
        transforms.CenterCrop((720, 720)),
        transforms.Resize((128, 128)),
        # ClaHe(),
        ToGrayScale(),
        transforms.ToTensor(),
        transforms.Normalize((94.05657 / 255.0, 94.05657 / 255.0, 94.05657 / 255.0),
                             (57.99793 / 255.0, 57.99793 / 255.0, 57.99793 / 255.0))
    ])

    train_data = datasets.ImageFolder(root=traindir, transform=transform_train)
    valid_data = datasets.ImageFolder(root=validdir, transform=transform_valid)
    test_data = datasets.ImageFolder(root=testdir, transform=transform_test)

    # Dataloader iterators, make sure to shuffle
    trainloader = DataLoader(train_data,
                             batch_size=batch_size,
                             # sampler=train_sampler,
                             shuffle=True,
                             num_workers=num_workers
                             );

    # print(np.array(trainloader.dataset).shape)

    validloader = DataLoader(valid_data,
                             batch_size=batch_size,
                             # sampler=valid_sampler,
                             shuffle=False,
                             num_workers=num_workers
                             );

    testloader = DataLoader(test_data,
                            batch_size=batch_size,
                            num_workers=num_workers
                            )

    print("train samples: " + str(len(train_data)))
    print("valid samples: " + str(len(valid_data)))
    print("test samples: " + str(len(test_data)))

    return trainloader, len(train_data), validloader, len(valid_data), testloader;