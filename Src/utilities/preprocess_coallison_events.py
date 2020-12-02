# Imports
import h5py
import numpy as np
import os
from sklearn.utils import resample

path_data_mat = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/raw/'))
# path_data_save = os.path.normpath(
#     os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_events_collision/'))
path_data_save = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_balanced_collison/'))

file_names = sorted(os.listdir(path=path_data_mat))
file_names = [x for x in file_names if not x.startswith('.')]


for file_name in file_names:
    # Extract data cnt - contains data of different channels:
    # EEG,'EMGf', 'lead_gas', 'lead_brake', 'dist_to_lead', 'wheel_X', 'wheel_Y', 'gas', 'brake'
    with h5py.File(os.path.join(path_data_mat, file_name), 'r') as f:
        cnt = f.get('cnt')
        x = np.array(cnt.get('x'))  # x is the continuous multivariate data

    # Extract data mrk - contains information on the EVENTS
    with h5py.File(os.path.join(path_data_mat, file_name), 'r') as f:
        mrk = f.get('mrk')
        time_mrk = np.array(mrk.get('time'))  # mrk.time is the timestamp for each event in milliseconds
        y = np.array(mrk.get('y'))  # mrk.y is a binary matrix telling which of the five types each event is

    car_brake = time_mrk.T * y[:, 1]  # contains the time in ms of events car_brake
    car_collision = time_mrk.T * y[:, 3]

    events = np.array(car_brake[car_brake > 0] / 5).astype(int)

    noevents_int = np.append(0, events)
    noevents_len = np.append(events[0], np.diff(events))

    # Construct matrix for all target segments for channel chn_name
    ts_i = 60  # Target segment interval intial point -> corresponds to 300ms
    ts_f = 260  # Target segment interval intial point -> corresponds to 1200ms

    # Find events with collision
    car_brake_y = car_brake[car_brake > 0]
    car_collision_y = car_collision[car_collision > 0]
    collison_label = []
    for i in range(len(car_brake_y)):
        diff = car_collision_y - car_brake_y[i]
        if (len(diff[diff > 0]) == 0):
            collison_label.append(0)
        elif (min(diff[diff > 0]) > 25000):  # car_brake_y[i] is too far from any collision, so label = 0, 25000 is
            collison_label.append(0)
        else:
            collison_label.append(1)  # There is a collison

    # To get BALANCED DATA
    event_collison = np.where(np.asarray(collison_label) == 1)[0]
    not_event = np.where(np.asarray(collison_label) == 0)[0]

    # downsample majority
    not_event_downsampled = resample(not_event,
                                     replace=False,  # sample without replacement
                                     n_samples=len(event_collison),  # match minority n
                                     random_state=27)  # reproducible results

    events = np.concatenate((events[not_event_downsampled], events[event_collison]))
    collison_label = np.concatenate((np.asarray(collison_label)[not_event_downsampled], np.asarray(collison_label)[event_collison]))
    # Comment until here





    channels_eeg = np.delete(np.arange(61), [0, 5])
    for event_index in range(len(events)):
        event_time = events[event_index]
        A = x[channels_eeg, event_time - ts_i:event_time + ts_f]
        # A_norm = np.subtract(A, np.array(A[:, :20].mean(1)).reshape(A.shape[0], 1))
        A_norm = A
        if(collison_label[event_index] == 1):
            path_out = os.path.join(path_data_save, '%s_segment_%d_event.npy' % (file_name, event_index)) # collison
        else:
            path_out = os.path.join(path_data_save, '%s_segment_%d_noevent.npy' % (file_name, event_index)) #nocollison
        if os.path.isfile(path_out):
            continue
        np.save(path_out, A_norm)

    # ### UNCOMMENT to get no_events
    #
    # gap_sml = 600  # Non-targets data blocks (1500 ms duration, 500 ms equidistant offset) that were at least 3000 ms apart from any stimulus.
    # nts_offset = 100  # 500 ms equidistant offset equidistant offset
    #
    #
    # gap_sml = 600  # 3000 ms apart from any stimulus
    # nts_offset = 100  # 500 ms
    # count = 0
    # for event_index in range(len(noevents_len)):
    #     noevents_seg = int((noevents_len[event_index] - gap_sml * 2 - ts_i + ts_f) / nts_offset)
    #     if (noevents_seg > 0):
    #         for noevent_ix in range(noevents_seg):
    #             ip = noevents_int[event_index] + gap_sml + nts_offset * noevent_ix
    #             fp = ip + ts_i + ts_f
    #             A = x[channels_eeg, ip:fp]
    #             A_norm = np.subtract(A, np.array(A[:, :20].mean(1)).reshape(A.shape[0], 1))
    #             np.save(os.path.join(path_data_save, '%s_segment_%d_noevent.npy' % (file_name, count)), A_norm)
    #             count += 1
