import os
import numpy as np
import torch

from torch.utils.data import TensorDataset

def load_dataset(fold):

    #path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/All_noevnets'))
    path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/Balanced_01'))

    files = sorted([elt for elt in os.listdir(path_to_data) if elt[-4:] == '.npy'],
                   key=(lambda x: int(x.split('_')[2])))
    np.random.RandomState(seed=0).shuffle(files)

    # CAREFUL NOT MIXING SUBJECTS
    # DIVIDE IN FOLDS

    datapoints = []
    labels = []
    for file in files:
        datapoint = np.load(os.path.join(path_to_data, file))
        datapoint = torch.from_numpy(datapoint).float()
        datapoints.append(datapoint)

        label = {'event': 1.0, 'noevent': 0.0}[file.split('_')[-1].split('.')[0]]
        label = torch.from_numpy(np.array(label)).float()
        # label = torch.from_numpy(np.array(np.random.randint(2))).float()
        labels.append(label)

    datapoints = torch.stack(datapoints).unsqueeze(1)
    labels = torch.stack(labels).unsqueeze(1)

    n_train = len(datapoints) // 3
    n_val = len(datapoints) // 3
    n_test = len(datapoints) - n_train - n_val

    print('>> n_train = %d' % n_train)
    print('>> n_val = %d' % n_val)
    print('>> n_test = %d' % n_test)

    datapoints_train = datapoints[:n_train]
    datapoints_val = datapoints[n_train:(n_train + n_val)]
    datapoints_test = datapoints[-n_test:]

    labels_train = labels[:n_train]
    labels_val = labels[n_train:(n_train + n_val)]
    labels_test = labels[-n_test:]

    return (TensorDataset(datapoints_train, labels_train),
            TensorDataset(datapoints_val, labels_val),
            TensorDataset(datapoints_test, labels_test))
