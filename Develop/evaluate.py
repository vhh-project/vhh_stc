from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import os
import numpy as np

##########################################################
###                  CONFIGURATION                     ###
##########################################################
dst_path = "/caa/Projects02/vhh/private/Results/comparison_crop_uncrop_nobackup/results/stc/"
pred_path = dst_path + "/final_results/uncropped/"
gt_path = "/data/share/datasets/vhh_mmsi_test_db_v3/annotations/stc/"
##########################################################
##########################################################

def calculate_metrics(dst_path, y_score, y_test, labels):
    """
    This method is used to calculate the metrics: precision, recall, f1score.
    Furthermore, the confusion matrix is generated and stored as figure on a specified location.

    :param y_score: This parameter must hold a valid numpy array with the class prediction per shot .
    :param y_test: This parameter must hold a valid numpy array with the groundtruth labels per shot.
    """
    #print(len(y_score))
    #print(len(y_test))

    #y_score = [s.lower() for s in y_score]
    #y_test = [s.lower() for s in y_test]
    #labels = [s.lower() for s in labels]

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

    # confusion matrix
    #print(y_test)
    #print(y_score)
    matrix = confusion_matrix(y_test, y_score, labels=labels)
    print(matrix)
    print(classification_report(y_test, y_score))


    print("save confusion matrix ...")
    plot_confusion_matrix(cm=matrix,
                               target_names=labels,
                               title='Confusion matrix',
                               cmap=None,
                               normalize=False,
                               path=dst_path + "/confusion_matrix_shot_based.pdf")
    plot_confusion_matrix(cm=matrix,
                               target_names=labels,
                               title='Confusion matrix',
                               cmap=None,
                               normalize=True,
                               path=dst_path + "/confusion_matrix_normalized_shot_based.pdf")

def plot_confusion_matrix(cm=None,
                          target_names=[],
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          path=""):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    from matplotlib import pyplot as plt
    plt.rc('pdf', fonttype=42)


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure() #figsize=(8, 6)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
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
    plt.ylabel('Ground truth label')
    plt.xlabel('Predicted label - accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.savefig(path)
    plt.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.4)
    plt.savefig(path, dpi=500)


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

def predicions_per_class(all_pred_data_np, all_gt_data_np, class_name):
    print("")
    print("#####################################")
    print("Results for class: " + str(class_name))

    pred_np = np.squeeze(all_pred_data_np[:, 3:])
    gt_np = np.squeeze(all_gt_data_np[:, 3:])
    gt_tilt_np = gt_np.copy()
    pred_tilt_np = pred_np.copy()

    idx_gt = np.where(gt_tilt_np != class_name)[0]
    idx_pred = np.where(pred_tilt_np != class_name)[0]

    gt_tilt_np[idx_gt] = "NA"
    pred_tilt_np[idx_pred] = "NA"
    #print(gt_tilt_np)
    #print(pred_tilt_np)

    y_score = np.squeeze(pred_tilt_np).tolist()
    y_test = np.squeeze(gt_tilt_np).tolist()
    calculate_metrics(y_score, y_test)


# load samples
pred_file_list = os.listdir(pred_path)
pred_file_list.sort()
print(pred_file_list)

all_pred_data_l = []
for file in pred_file_list:
    print(file)
    pred_l = load_results(pred_path + file)
    #print(gt_np)
    all_pred_data_l.extend(pred_l)

all_pred_data_np = np.array(all_pred_data_l)
print(all_pred_data_np)

# load groundtruth labels
gt_file_list = os.listdir(gt_path)
gt_file_list.sort()

print(gt_file_list)
#exit()
all_gt_data_l = []
for file in gt_file_list:
    #print(file)
    gt_l = load_results(gt_path + file)
    #print(gt_np)
    all_gt_data_l.extend(gt_l)

all_gt_data_np = np.array(all_gt_data_l)
#print(all_gt_data_np)
labels = (np.unique(all_pred_data_np[:, 3:]))
#print(labels)

#print(np.unique(all_pred_data_np[:, 3:]))
#print(np.unique(all_gt_data_np[:, 3:]))



# all classes
y_score = np.squeeze(all_pred_data_np[:, 3:]).tolist()
y_test = np.squeeze(all_gt_data_np[:, 3:]).tolist()

#print(np.array(np.unique(y_score, return_counts=True)))
#print(np.array(np.unique(y_test, return_counts=True)))

labels = ["CU", "ELS", "I", "LS", "MS", "NA"]
calculate_metrics(dst_path, y_score, y_test, labels)


# only pan tilt and NA
#predicions_per_class(all_pred_data_np, all_gt_data_np, "TILT")
#predicions_per_class(all_pred_data_np, all_gt_data_np, "PAN")
exit()
