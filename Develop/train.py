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
import torch.nn as nn  # Add on classifier
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import cv2
from torch.autograd import Variable
from datetime import datetime
import json
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

from vhh_stc.Models import *
from vhh_stc.Datasets import *


def plot_confusion_matrix(cm=None,
                          target_names=[],
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, path=""):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(path)

#######################
## train model
#######################

def train(param_dict):
    dst_path = param_dict['dst_path'];
    expTimeStamp = param_dict['expTimeStamp']
    expType = param_dict['expType']
    expNet = param_dict['expNet']
    pre_trained_weights = param_dict['expWeights']
    expNum = param_dict['expNum']
    db_path = param_dict['db_path']
    n_epochs = param_dict['n_epochs']
    batch_size = param_dict['batch_size']
    lRate = param_dict['lRate']
    wDecay = param_dict['wDecay']
    classes = param_dict['classes']
    early_stopping_threshold = param_dict['early_stopping_threshold']

    ####################
    ## create experiment
    ####################
    expName = str(expTimeStamp) + "_" + str(expType) + "_" + str(expNet) + "_ExpNum_" + str(expNum);
    expFolder = dst_path + "/" + expName

    if not os.path.isdir(dst_path + "/" + expName):
        os.mkdir(expFolder)

    with open(expFolder + "/experiment_notes.json", 'w') as json_file:
        json.dump(param_dict, json_file)

    writer = SummaryWriter(log_dir="./runs/" + expName)

    ################
    # load dataset
    ################
    trainloader, nSamples_train, validloader, nSamples_valid, testloader = loadDatasetFromFolder(db_path, batch_size);

    # Whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False;
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    ################
    # define model
    ################
    model = loadModel(model_arch=expNet, classes=classes, pre_trained_path=pre_trained_weights, expType=expType);
    #if(pre_trained_path != None):
    #    model_dict_state = torch.load(pre_trained_path)
        #model.load_state_dict(model_dict_state['net'])

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    ################
    # Specify the Loss function
    ################
    criterion = nn.CrossEntropyLoss()

    ################
    # Specify the optimizer
    ################
    optimizer = optim.SGD(model.parameters(), lr=lRate, momentum=0.9, nesterov=True, weight_decay=wDecay)

    # print("[Creating Learning rate scheduler...]")
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)

    # Define the lists to store the results of loss and accuracy
    best_acc = 0.0;
    best_loss = 10.0;
    early_stopping_cnt = 0;

    for epoch in range(0, n_epochs):
        tLoss_sum = 0;
        tAcc_sum = 0;
        vLoss_sum = 0;
        vAcc_sum = 0;
        ###################
        # train the model #
        ###################
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):

            # Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # run forward pass
            outputs = model(inputs)
            tLoss = criterion(outputs, labels)
            tLoss_sum += tLoss.item()

            # run backward pass
            optimizer.zero_grad()
            tLoss.backward()
            optimizer.step()

            preds = outputs.argmax(1, keepdim=True)
            correct = preds.eq(labels.view_as(preds)).sum()
            acc = correct.float() / preds.shape[0]
            tAcc_sum += acc.item()

            # if (i + 1) % 10 == 0:
            #    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, n_epochs, i + 1, total_step_train, tLoss.item(), (correct / total) * 100))

        ###################
        # validate the model #
        ###################
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(validloader):
                ## Convert torch tensor to Variable
                inputs = Variable(inputs)
                labels = Variable(labels)

                # If we have GPU, shift the data to GPU
                CUDA = torch.cuda.is_available()
                if CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                vLoss = criterion(outputs, labels)
                vLoss_sum += vLoss.item()

                preds = outputs.argmax(1, keepdim=True)
                correct = preds.eq(labels.view_as(preds)).sum()
                acc = correct.float() / preds.shape[0]
                vAcc_sum += acc.item()

                # if (i + 1) % 5 == 0:
                #    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, n_epochs, i + 1, total_step_val, vLoss.item(), (correct / total) * 100))

        print('Epoch [{:d}/{:d}]: train_loss: {:.3f}, train_acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}'.format(
            epoch + 1, n_epochs, tLoss_sum / len(trainloader), tAcc_sum / len(trainloader),
            vLoss_sum / len(validloader), vAcc_sum / len(validloader)))

        ###############################
        # write results to tensorboard
        ###############################
        writer.add_scalar('train_loss', tLoss_sum / len(trainloader), epoch)
        writer.add_scalar('valid_loss', vLoss_sum / len(validloader), epoch)
        writer.add_scalar('train_acc', tAcc_sum / len(trainloader), epoch)
        writer.add_scalar('valid_acc', vAcc_sum / len(validloader), epoch)

        ###############################
        # Save checkpoint.
        ###############################
        acc_curr = 100. * (vAcc_sum / len(validloader));
        vloss_curr = vLoss_sum / len(validloader)
        if acc_curr > best_acc:
            print('Saving...')
            state = {
                'net': model.state_dict(),
                'acc': acc_curr,
                'loss': vloss_curr,
                'epoch': epoch,
            }
            # if not os.path.isdir('checkpoint'):
            #    os.mkdir('checkpoint')
            torch.save(state,
                       expFolder + "/" "best_model" + ".pth")  # + str(round(acc_curr, 4)) + "_" + str(round(vloss_curr, 4))
            best_acc = acc_curr
            # best_loss = vloss_curr
            early_stopping_cnt = 0;
        # scheduler.step()

        ###############################
        # early stopping.
        ###############################
        if (acc_curr <= best_acc):
            early_stopping_cnt = early_stopping_cnt + 1;
        if (early_stopping_cnt >= early_stopping_threshold):
            print('Early stopping active --> stop training ...')
            break;

    writer.close()


def run():
    lRate_list = [0.001, 0.0001, 0.00001]
    #lRate_list = [0.0001]
    wDecay_list = [0.0, 0.002]
    #wDecay_list = [0.02]
    ##batch_size_list = [64, 128]
    #expNet_list = ["Resnet", "Densenet", "Vgg"]
    expNet_list = ["Resnet", "Vgg"]

    # lRate_list = [0.01, 0.001, 0.0001]
    # lRate_list = [0.001]
    # wDecay_list = [0.008]
    # batch_size_list = [64, 128]
    # expNet_list = ["Vgg"]

    for k in range(0, len(expNet_list)):
        for i in range(0, len(lRate_list)):
            for j in range(0, len(wDecay_list)):
                ####################
                # experiment config
                ####################
                exp_config = {'lRate': lRate_list[i],
                              'batch_size': 64,
                              'n_epochs': 200,
                              'wDecay': wDecay_list[j],
                              'classes': ["CU", "ELS", "LS", "MS"],
                              'early_stopping_threshold': 30,
                              'db_path': "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/vhh_stc/20191203/db_v8/",
                              'dst_path': "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/pytorch_exp_db_v8/",
                              'expTimeStamp': datetime.now().strftime("%Y%m%d"),
                              'expType': "BasicCNN_db_v8",
                              'expNet': expNet_list[k],
                              'expWeights': "",
                              'expNum': str(j) + str(i) + str(k)
                              }

                train(exp_config)


def runSingleExp():
    lRate_list = 0.0001
    wDecay_list = 0.0002
    expNet_list = "sota_rahul"

    ####################
    # experiment config
    ####################
    exp_config = {'lRate': lRate_list,
                  'batch_size': 64,
                  'n_epochs': 200,
                  'wDecay': wDecay_list,
                  'classes': ["CU", "ELS", "LS", "MS"],
                  'early_stopping_threshold': 20,
                  'db_path': "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/vhh_stc/20191203/db_v8/",
                  'dst_path': "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/sota/rahul/",
                  'expTimeStamp': datetime.now().strftime("%Y%m%d"),
                  'expType': "all",
                  'expNet': expNet_list,
                  'expWeights': "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/sota/rahul/20200130_all_sota_rahul_ExpNum_1/",
                  'expNum': 999
                  }

    train(exp_config) #, pre_trained_path)


#########################
# test section
#########################


def test():
    path = "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/pytorch_exp_db_v8/"
    exp_list = [a for a in os.listdir(path) if os.path.isdir(path + "/" + a)]
    print(exp_list)

    ## create results_summary file
    fp = open(path + "/results_summary.csv", 'w');
    header = "expFolder;accuracy;precision;recall;f1_score;kappa"
    fp.write(header + "\n");

    for folder in exp_list:

        with open(path + "/" + folder + "/experiment_notes.json", 'r') as json_file:
            param_dict = json.load(json_file)

        # dst_path = param_dict['dst_path'];
        dst_path = path
        expTimeStamp = param_dict['expTimeStamp']
        expType = param_dict['expType']
        expNet = param_dict['expNet']
        expWeights = param_dict['expWeights']
        expNum = param_dict['expNum']
        # db_path = param_dict['db_path']
        db_path = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/vhh_stc/20191203/db_v8/"
        n_epochs = param_dict['n_epochs']
        batch_size = param_dict['batch_size']
        lRate = param_dict['lRate']
        wDecay = param_dict['wDecay']
        classes = param_dict['classes']
        early_stopping_threshold = param_dict['early_stopping_threshold']

        expName = str(expTimeStamp) + "_" + str(expType) + "_" + str(expNet) + "_" + str(expWeights) + "ExpNum_" + str(
            expNum);
        expFolder = dst_path + "/" + expName

        ################
        # load dataset
        ################
        trainloader, nSamples_train, validloader, nSamples_valid, testloader = loadDatasetFromFolder(db_path,
                                                                                                     batch_size);
        print(classes)
        # Whether to train on a gpu
        train_on_gpu = torch.cuda.is_available()
        print("Train on gpu: " + str(train_on_gpu))

        # Number of gpus
        multi_gpu = False;
        if train_on_gpu:
            gpu_count = torch.cuda.device_count()
            print("gpu_count: " + str(gpu_count))
            if gpu_count > 1:
                multi_gpu = True
            else:
                multi_gpu = False

        model = loadModel(expNet, classes)
        model_dict_state = torch.load(expFolder + "/best_model.pth")
        # exit()
        model.load_state_dict(model_dict_state['net'])
        # print(model)

        print("model performance:")
        print("------------------")
        print("validation accuracy: " + str(model_dict_state['acc']))
        print("validation loss: " + str(model_dict_state['loss']))
        print("epoch: " + str(model_dict_state['epoch']))

        vAcc_sum = 0;

        test_predictions_all = []
        test_labels_all = []

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                ## Convert torch tensor to Variable
                inputs = Variable(inputs)
                labels = Variable(labels)

                # If we have GPU, shift the data to GPU
                # CUDA = torch.cuda.is_available()
                # if CUDA:
                #    inputs = inputs.cuda()
                #    labels = labels.cuda()


                ACITVATE_ENSEMBLE = False;
                if(ACITVATE_ENSEMBLE == True):
                    outputs_sum = model(inputs)
                    for i in range(0, 10):
                        outputs = model(inputs)
                        outputs_sum = outputs_sum + outputs
                        outputs = outputs_sum / 10.0
                else:
                    outputs = model(inputs)

                # print(outputs.detach().cpu().numpy().shape)
                # print(labels.detach().cpu().numpy().shape)
                test_predictions_all.extend(outputs.detach().cpu().numpy())
                test_labels_all.extend(labels.detach().cpu().numpy())

                preds = outputs.argmax(1, keepdim=True)
                correct = preds.eq(labels.view_as(preds)).sum()
                acc = correct.float() / preds.shape[0]
                vAcc_sum += acc.item()
                # print(len(validloader))


                # if (i + 1) % 5 == 0:
                #    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, n_epochs, i + 1, total_step_val, vLoss.item(), (correct / total) * 100))

            print('Test Accuracy of the model on the testset: {} %'.format((vAcc_sum / len(testloader)) * 100))

        test_predictions_all_np = np.array(test_predictions_all)
        test_labels_all_np = np.array(test_labels_all)

        # y_score = np.rint(scores).astype('int')
        y_score = np.argmax(test_predictions_all_np, axis=1)
        y_test = test_labels_all_np

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(y_test, y_score)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y_test, y_score, average='weighted')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_test, y_score, average='weighted')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test, y_score, average='weighted')
        print('F1 score: %f' % f1)

        # kappa
        kappa = cohen_kappa_score(y_test, y_score)
        print('Cohens kappa: %f' % kappa)
        # confusion matrix
        matrix = confusion_matrix(y_test, y_score, labels=[0, 1, 2, 3])
        print(matrix)

        print("save confusion matrix ...")
        plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=True,
                              path=expFolder + "/confusion_matrix_normalize.png")
        plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=False,
                              path=expFolder + "/confusion_matrix.png")

        print("save results to summary file ...")
        line = str(expFolder) + ";" + str(accuracy) + ";" + str(precision) + ";" + str(recall) + ";" + str(
            f1) + ";" + str(kappa)
        fp.write(line + "\n");

    fp.close();


def testSingleExp():
    #expFolder1 = "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/20200130_test_SOTA_Imagenet_ExpNum_85/"
    #expFolder1 = "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/sota/rahul/20200130_all_sota_rahul_ExpNum_1/"
    #expFolder1 = "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/sota/rahul/20200130_last_sota_rahul_ExpNum_1/"
    #expFolder1 = "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/20200202_own_deeplab_resnet_test_ExpNum_202/"
    #expFolder1 = "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/20200127_test_Resnet_Imagenet_ExpNum_75/"
    expFolder = "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/20200206_own_depthmap_resnet_test_ExpNum_303/"
    #expFolder = "/caa/Projects02/vhh/private/Results/ShotTypeClassification/20191118/pytorch_exp_db_v8/20200206_BasicCNN_db_v8_Resnet_ExpNum_100/"

    # db_path = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/vhh_stc/20191203/db_v7/"

    with open(expFolder + "/experiment_notes.json", 'r') as json_file:
        param_dict = json.load(json_file)

    # dst_path = param_dict['dst_path'];
    #dst_path = path
    expNet = param_dict['expNet']
    db_path = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/vhh_stc/20191203/db_v8/"
    batch_size = param_dict['batch_size']
    classes = param_dict['classes']

    trainloader, nSamples_train, validloader, nSamples_valid, testloader = loadDatasetFromFolder(db_path, batch_size);
    print(classes)
    # Whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False;
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    outputs_list = []

    model = loadModel(expNet, classes, expFolder + "/best_model.pth")

    vAcc_sum = 0;

    test_predictions_all = []
    test_labels_all = []

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = Variable(inputs)
            labels = Variable(labels)

            #model.conv1.register_forward_hook(get_activation('conv1'))
            outputs1 = model(inputs)
            '''
            act = activation['conv1'].squeeze()

            fig, axarr = plt.subplots(8, 8)
            print(act.size())
            a = 0;
            b = 0;
            for j in range(0, act.size(0)):
                b = b + 1
                if(act.size(0) % 8):
                    a = a + 1
                    b = 0;
                print("-----------")
                print(b)
                print(a)
                axarr[b, a].plot(act[j])
                #plt.imsave("./test_" + str(idx) +".png", act[idx])
            fig.savefig("./test.png");

            #fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            #axs[0, 0].plot(x)

            #plt.show()
            '''
            test_predictions_all.extend(outputs1.detach().cpu().numpy())
            test_labels_all.extend(labels.detach().cpu().numpy())

            preds = outputs1.argmax(1, keepdim=True)
            correct = preds.eq(labels.view_as(preds)).sum()
            acc = correct.float() / preds.shape[0]
            vAcc_sum += acc.item()

        print('Test Accuracy of the model on the testset: {} %'.format((vAcc_sum / len(testloader)) * 100))

    test_predictions_all_np = np.array(test_predictions_all)
    test_labels_all_np = np.array(test_labels_all)

    # y_score = np.rint(scores).astype('int')
    y_score = np.argmax(test_predictions_all_np, axis=1)
    y_test = test_labels_all_np

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, y_score)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_score, average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_score, average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_score, average='weighted')
    print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(y_test, y_score)
    print('Cohens kappa: %f' % kappa)
    # confusion matrix
    matrix = confusion_matrix(y_test, y_score, labels=[0, 1, 2, 3])
    print(matrix)

    print("save confusion matrix ...")
    plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=True,
                          path=expFolder + "/confusion_matrix_normalize.png")
    plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=False,
                          path=expFolder + "/confusion_matrix.png")