import os
import numpy as np
import torch

from torch.utils.data import TensorDataset

def load_dataset(fold):

    #path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/All_noevnets'))
    #path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/Balanced_01'))
    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/Only_events')) # Contains only event-segments
    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/nModes320'))  # Contains nMoodes
    # path_to_data_perf = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/Performance'))

    path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                 '../../data/post_processed/AllSubjects_events'))  #

    # path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                              '../../data/post_processed/AllSubjects_events_EMG'))  #


    # path_to_data_perf = os.path.normpath(
    #     os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_performance'))

    path_to_data_perf = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_reactio_time'))

    # path_to_data_perf = os.path.normpath(
    #     os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_sumEEG'))

    # path_to_data_perf = os.path.normpath(
    #     os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_meanEEG'))


    files = sorted([elt for elt in os.listdir(path_to_data) if elt[-4:] == '.npy'],
                   key=(lambda x: int(x.split('_')[2])))
    np.random.RandomState(seed=0).shuffle(files)

    # CAREFUL NOT MIXING SUBJECTS
    # DIVIDE IN FOLDS

    datapoints = []
    # labels = []
    performances = []
    for file in files:
        datapoint = np.load(os.path.join(path_to_data, file))
        datapoint = torch.from_numpy(datapoint).float()
        datapoints.append(datapoint)

        # Binary LEBEL classification: event; noevent
        # label = {'event': 1.0, 'noevent': 0.0}[file.split('_')[-1].split('.')[0]]
        # label = torch.from_numpy(np.array(label)).float()
        # # label = torch.from_numpy(np.array(np.random.randint(2))).float()
        # labels.append(label)

        # Continuous PERFORMANCE prediction
        file_perf = file.rsplit('_',1)[0].split('/')[-1] + '_performance.npy'
        performance = np.load(os.path.join(path_to_data_perf, file_perf))
        performance = torch.from_numpy(performance).float()

        # To compare to random perf
        # #performance = torch.from_numpy(np.array(np.random.randint(100))).float()
        #
        # To compare to random perf according to distribution
        # performance = torch.from_numpy(np.array(np.random.normal(430, 126, 1))).float()

        performances.append(performance)

    datapoints = torch.stack(datapoints).unsqueeze(1)
    # labels = torch.stack(labels).unsqueeze(1)
    performances = torch.stack(performances).unsqueeze(1)

    n_train = len(datapoints) // 3  # int(len(datapoints)*0.4)
    n_val   = len(datapoints) // 3  # int(len(datapoints)*0.2) #
    n_test  = len(datapoints) - n_train - n_val

    print('>> n_train = %d' % n_train)
    print('>> n_val = %d' % n_val)
    print('>> n_test = %d' % n_test)

    datapoints_train = datapoints[:n_train]
    datapoints_val = datapoints[n_train:(n_train + n_val)]
    datapoints_test = datapoints[-n_test:]

    # labels_train = labels[:n_train]
    # labels_val  = labels[n_train:(n_train + n_val)]
    # labels_test = labels[-n_test:]

    performances_train = performances[:n_train]
    performances_val = performances[n_train:(n_train + n_val)]
    performances_test = performances[-n_test:]

    return (TensorDataset(datapoints_train, performances_train),
            TensorDataset(datapoints_val, performances_val),
            TensorDataset(datapoints_test, performances_test))
