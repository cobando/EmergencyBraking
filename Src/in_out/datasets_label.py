import os
import numpy as np
import torch

from torch.utils.data import TensorDataset
from sklearn.utils import resample

def load_dataset(fold,window_size):

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/VPja_all'))
    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/Balanced_01')) # Only for one subject
    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/AllSubjects_events_collision'))  #
    #
    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/AllSubjects_balanced_collison'))  #

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/AllSubjects_filter_1_8_braking_collison'))  #

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/AllSubjects_balanced_filter_1_8_braking_collison'))  #

    path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                 '../../data/post_processed/nModes500'))  #

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/AllSubjects_erpchnls_balanced_filter_1_8_braking_collison'))  #

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/AllSubjects_erpchnls_eog_balanced_filter_1_8_braking_collison'))  #

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/sampled_collision_KL'))  #

    path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                 '../../data/post_processed/persubject_epochs_collision/subject_12_collision_trianvaltest'))  #

    path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                 '../../data/post_processed/AllSubjects_events_noevents_balanced'))  #



    files = sorted([elt for elt in os.listdir(path_to_data) if elt[-4:] == '.npy'],
                   key=(lambda x: int(x.split('_')[2])))
    np.random.RandomState(seed=0).shuffle(files)

    # CAREFUL TO NOT MIXING SUBJECTS
    # DIVIDE IN FOLDS

    datapoints = []
    labels = []
    w_b = 60/320 # This is fix, what changes is window_size
    w_a = 260/320
    t_event = 60
    for file in files:
        datapoint = np.load(os.path.join(path_to_data, file))

        # Select window_size ## Comment for modes
        #datapoint = datapoint[:, int(t_event - window_size*w_b): int(t_event + window_size*w_a)] # keep propotional window

        datapoint = datapoint[:, : window_size] # keep starting point (before the event) and reduce the window

        datapoint = torch.from_numpy(datapoint).float()
        datapoints.append(datapoint)

        # Binary LEBEL classification: event; noevent
        label = {'event': 1.0, 'noevent': 0.0}[file.split('_')[-1].split('.')[0]]
        label = torch.from_numpy(np.array(label)).float()

        # # Test NN fit to random labels
        # label = torch.from_numpy(np.array(np.random.randint(2))).float()

        labels.append(label)


    datapoints = torch.stack(datapoints).unsqueeze(1)
    labels = torch.stack(labels).unsqueeze(1)

    n_train = int(len(datapoints)*0.8*0.8) #len(datapoints) // 3  # int(len(datapoints)*0.4)
    n_test  = int(len(datapoints)*0.2) #len(datapoints) - n_train - n_val
    n_val = len(datapoints) - n_train - n_test #len(datapoints) // 3  # int(len(datapoints)*0.2) #

    # n_train = len(datapoints) // 3  # int(len(datapoints)*0.4)
    # n_val = len(datapoints) // 3  #
    # n_test = len(datapoints) - n_train - n_val



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
