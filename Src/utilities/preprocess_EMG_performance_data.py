# Imports
import scipy.io
import h5py
import time
from h5py import File  # Package used for loading data from the input h5 file
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir

path_data_mat = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/raw/'))
path_data_save1 = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_events_EMG/'))
#path_data_save2 = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_performance/'))
path_data_save3 = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_reactio_time/'))

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

    car_brake_y = time_mrk.T * y[:, 1]  # contains the time in ms of events car_brake
    react_emg_y = time_mrk.T*y[:, 4]
    car_collision_y = time_mrk.T*y[:, 3]

    events_y = np.array(car_brake_y[car_brake_y > 0] / 5).astype(int)
    noevents_int = np.append(0, events_y)
    noevents_len = np.append(events_y[0], np.diff(events_y))

    car_brake_y_no0 = car_brake_y[car_brake_y > 0]
    react_emg_y_no0 = react_emg_y[react_emg_y > 0]
    react_time = []
    event_noreact = []
    for i in range(len(car_brake_y_no0)):  #
        diff = react_emg_y_no0 - car_brake_y_no0[i]
        if (len(diff[diff > 0]) == 0):
            event_noreact.append(i)
        elif (min(diff[diff > 0]) > 10000):
            event_noreact.append(i)
        else:
            react_time.append(min(diff[diff > 0]))


    events_react_y = np.delete(events_y,event_noreact)
    perf = react_time / np.power(x[64, events_react_y], 2) #  x[64, events_react_y] could be 0
    perf_react_time = react_time

    # Construct matrix for all target segments for channel chn_name
    ts_i = 60  # Target segment interval intial point -> corresponds to 300ms
    ts_f = 260  # Target segment interval intial point -> corresponds to 1200ms
    gap_sml = 600  # Non-targets data blocks (1500 ms duration, 500 ms equidistant offset) that were at least 3000 ms apart from any stimulus.
    nts_offset = 100  # 500 ms equidistant offset equidistant offset

    channels_eeg = np.arange(62)
    channels_eeg = np.delete(channels_eeg, [0, 5])
    # channels_eeg = 62 # To only get EMG data
    for event_index in range(len(events_react_y)):
        event_time = events_react_y[event_index]
        A_norm = x[channels_eeg, event_time - ts_i:event_time + ts_f]
        #A_norm = np.subtract(A, np.array(A[:, :20].mean(1)).reshape(A.shape[0], 1))
        path_out = os.path.join(path_data_save1, '%s_segment_%d_event.npy' % (file_name, event_index))
        #path_out_perf = os.path.join(path_data_save2, '%s_segment_%d_performance.npy' % (file_name, event_index))
        path_out_perf_reaction_time = os.path.join(path_data_save3, '%s_segment_%d_performance.npy' % (file_name, event_index))

        # if os.path.isfile(path_out):
        #     continue


        # if np.isfinite(perf[event_index]):
        #     np.save(path_out, A_norm)
        #     np.save(path_out_perf, perf[event_index])

        if np.isfinite(perf_react_time[event_index]): # or if perf_react_time[event_index] <100, don't put it
            np.save(path_out, A_norm)
            np.save(path_out_perf_reaction_time, perf_react_time[event_index])
