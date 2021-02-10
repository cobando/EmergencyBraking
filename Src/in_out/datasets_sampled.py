import os
import numpy as np
import torch

from torch.utils.data import TensorDataset
from sklearn.utils import resample

def load_dataset(fold,window_size):

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/sampled_collision_KL'))  #
    # This doesn't work because I'm using the same base event subject to generate data in the train,val and test sets
    # So I created another dataset where data sampled from a subject is ONLY in EITHER test,val OR TEST.

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/sampled_collision_KL_trianvaltest'))  #

    path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                 '../../data/post_processed/sampled_persubject_KL/subject_12_collision_KL_trianvaltest'))  #


    files = sorted([elt for elt in os.listdir(path_to_data) if elt[-4:] == '.npy'],
                   key=(lambda x: int(x.split('_')[3])))
    np.random.RandomState(seed=0).shuffle(files)

    # CAREFUL TO NOT MIXING SUBJECTS
    # DIVIDE IN FOLDS

    datapoints_train = []
    datapoints_val = []
    datapoints_test = []

    labels_train = []
    labels_val = []
    labels_test = []

    for file in files:
        datapoint = np.load(os.path.join(path_to_data, file))
        # Standarized mean 0, std 1
        datapoint = (datapoint - datapoint.mean(axis=1).reshape(59, 1)) / datapoint.std(axis=1).reshape(59, 1)

        # # Select window_size ## Comment for modes
        # datapoint = datapoint[:, int(t_event - window_size*w_b): int(t_event + window_size*w_a)]

        datapoint = torch.from_numpy(datapoint).float()
        datapoint_type = file.split('_')[-3]

        # Binary LEBEL classification: event; noevent
        label = {'event': 1.0, 'noevent': 0.0}[file.split('_')[-1].split('.')[0]]
        label = torch.from_numpy(np.array(label)).float()

        # # Test NN fit to random labels
        # label = torch.from_numpy(np.array(np.random.randint(2))).float()
        if datapoint_type == 'train':
            datapoints_train.append(datapoint)
            labels_train.append(label)

        elif datapoint_type == 'val':
            datapoints_val.append(datapoint)
            labels_val.append(label)

        else:
            datapoints_test.append(datapoint)
            labels_test.append(label)


    datapoints_train = torch.stack(datapoints_train).unsqueeze(1)
    labels_train = torch.stack(labels_train).unsqueeze(1)

    datapoints_val = torch.stack(datapoints_val).unsqueeze(1)
    labels_val = torch.stack(labels_val).unsqueeze(1)

    datapoints_test = torch.stack(datapoints_test).unsqueeze(1)
    labels_test = torch.stack(labels_test).unsqueeze(1)

    n_train = len(datapoints_train)
    n_val   = len(datapoints_val)
    n_test  = len(datapoints_test)

    print('>> n_train = %d' % n_train)
    print('>> n_val = %d' % n_val)
    print('>> n_test = %d' % n_test)


    return (TensorDataset(datapoints_train, labels_train),
            TensorDataset(datapoints_val, labels_val),
            TensorDataset(datapoints_test, labels_test))
