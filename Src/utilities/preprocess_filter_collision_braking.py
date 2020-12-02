# Imports
import h5py
import numpy as np
import os
import mne
from mne.channels import make_standard_montage
from sklearn.utils import resample

path_data_mat = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/raw/'))

##### Variables
filter_0 = 1
filter_1 = 8
channels_eeg = np.delete(np.arange(61), [0, 5])
montage_std = make_standard_montage('biosemi64')
event_id = {'braking': 0, 'collision': 1}  #
tmin = -0.3  # ts_i = 60  # Target segment interval intial point -> corresponds to 300ms
tmax = 1.2  # ts_f = 260  # Target segment interval intial point -> corresponds to 1200ms`
fs = 200
baseline_correction = 0.1

# path_data_save = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_filter_' +
#                  str(filter_0) + '_' + str(filter_1) + '_braking_collison/'))

# path_data_save = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_balanced_filter_' +
#                  str(filter_0) + '_' + str(filter_1) + '_braking_collison/'))

path_data_save = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_erpchnls_eog_balanced_filter_' +
                 str(filter_0) + '_' + str(filter_1) + '_braking_collison/')) # Contians only ERP channels (the ones highlighted in the paper)


if not os.path.exists(path_data_save):
    os.makedirs(path_data_save)

# path_plot_save = os.path.normpath(
#     os.path.join(os.path.dirname(__file__), '../../plots/epochs_driving_collision_filter_' +
#                  str(filter_0) + '_' + str(filter_1)))
# if not os.path.exists(path_plot_save):
#     os.makedirs(path_plot_save)


###### Import DATA
file_names = sorted(os.listdir(path=path_data_mat))
file_names = [x for x in file_names if not x.startswith('.')]

for file_name in file_names:
    # Extract data cnt - contains data of different channels:
    # EEG,'EMGf', 'lead_gas', 'lead_brake', 'dist_to_lead', 'wheel_X', 'wheel_Y', 'gas', 'brake'
    with h5py.File(os.path.join(path_data_mat, file_name), 'r') as f:
        cnt = f.get('cnt')
        x = np.array(cnt.get('x'))  # x is the continuous multivariate data
        test = f['cnt/clab']
        name_elec = list()
        for j in range(len(test)):
            st = test[j][0]
            obj = f[st]
            str1 = ''.join(chr(i) for i in obj[:])
            name_elec.append(str1)

    # Extract data mrk - contains information on the EVENTS
    with h5py.File(os.path.join(path_data_mat, file_name), 'r') as f:
        mrk = f.get('mrk')
        time_mrk = np.array(mrk.get('time'))  # mrk.time is the timestamp for each event in milliseconds
        y = np.array(mrk.get('y'))  # mrk.y is a binary matrix telling which of the five types each event is

    ###### Events
    car_brake = time_mrk.T * y[:, 1]  # contains the time in ms of events car_brake
    car_collision = time_mrk.T * y[:, 3]

    events = np.array(car_brake[car_brake > 0] / 5).astype(int)

    ### Collision Events
    # Find events with collision when comparing car_brake, car_collision, no need to convert with /5
    car_brake_y = car_brake[car_brake > 0]
    car_collision_y = car_collision[car_collision > 0]
    collison_label = []
    for i in range(len(car_brake_y)):
        diff = car_collision_y - car_brake_y[i]
        if (len(diff[diff > 0]) == 0):
            collison_label.append(0)
        elif (min(diff[diff > 0]) > 20000):  # car_brake_y[i] is too far from any collision, so label = 0, 25000 is
            collison_label.append(0)
        else:
            collison_label.append(1)

    # To get BALANCED DATA
    event_collison = np.where(np.asarray(collison_label) == 1)[0]
    not_event = np.where(np.asarray(collison_label) == 0)[0]

    # downsample majority
    not_event_downsampled = resample(not_event,
                                     replace=False,  # sample without replacement
                                     n_samples=len(event_collison),  # match minority n
                                     random_state=27)  # reproducible results

    events = np.concatenate((events[not_event_downsampled], events[event_collison]))
    collison_label = np.concatenate(
        (np.asarray(collison_label)[not_event_downsampled], np.asarray(collison_label)[event_collison]))
    # Comment until here




    ##### Matrix of events anotations
    events_collison = np.empty([len(events), 3], dtype=int)
    events_collison[:, 0] = events
    events_collison[:, 1] = int(0)
    events_collison[:, 2] = np.array(collison_label).astype(int)

    ###### EEG data in MNE
    ch_types = 'eeg'
    ch_names = np.array(name_elec)[channels_eeg]

    ###################@ Comment one or the other###################@###################@###################@
    # # All EEG channels
    # eeg_data = x[channels_eeg, :]
    # info = mne.create_info(ch_names=ch_names.tolist(), sfreq=fs, ch_types=ch_types)  # , ch_types = ch_types)


    # Selected EEG channels
    eeg_erp = ['AF3', 'AF4', 'F7', 'Fz', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
               'C5', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P9', 'P10', 'POz']
    eeg_erp = ['EOGv', 'AF3', 'AF4', 'EOGh', 'F7', 'Fz', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
               'C5', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P9', 'P10', 'POz'] ## Inlciudes also EOG
    eeg_erp_indx = [np.where(ch_names == i)[0][0] for i in eeg_erp]
    eeg_data = x[eeg_erp_indx, :]
    info = mne.create_info(ch_names=eeg_erp, sfreq=fs, ch_types=ch_types)  # , ch_types = ch_types)
    ###################@###################@###################@###################@###################@

    info.set_montage(montage_std)
    raw = mne.io.RawArray(eeg_data, info)

    ###### Filter Data
    raw_filt = raw.copy()
    raw_filt.filter(filter_0, filter_1, fir_design='firwin')

    ##### Picks
    picks = mne.pick_types(raw_filt.info, meg=False, eeg=True, eog=True,
                           exclude='bads')

    ##### Epochs
    epochs = mne.Epochs(raw_filt, events_collison, event_id, tmin=tmin, tmax=tmax, proj=True,
                        picks=picks, baseline=(None, baseline_correction), preload=True,
                        reject=None)
    data = epochs._data

    # ##### Plots ONLY ONCE
    # epochs['braking'].average().plot(time_unit='ms', titles="Braking").savefig(path_plot_save + "/epochs_driving"+ '%s_subject.png' % (file_name))
    # epochs['collision'].average().plot(time_unit='ms', titles="Collision").savefig(path_plot_save + "/epochs_braking" + '%s_subject.png' % (file_name))


    #### Save data  -  each epoch in a file
    for event_index in range(len(events)):
        if (collison_label[event_index] == 1):
            path_out = os.path.join(path_data_save, '%s_segment_%d_event.npy' % (file_name, event_index))  # collison
        else:
            path_out = os.path.join(path_data_save,
                                    '%s_segment_%d_noevent.npy' % (file_name, event_index))  # nocollison
        if os.path.isfile(path_out):
            continue
        np.save(path_out, data[event_index, :, :])

    #  End
