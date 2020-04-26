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
path_data_save = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/post_processed/'))

file_names = sorted(os.listdir(path=path_data_mat))
file_names = [x for x in file_names if not x.startswith('.')]

file_name = "VPja.mat"  # VPja.mat ; VPbax.mat


# Extract data cnt - contains data of different channels:
# EEG,'EMGf', 'lead_gas', 'lead_brake', 'dist_to_lead', 'wheel_X', 'wheel_Y', 'gas', 'brake'
with h5py.File(os.path.join(path_data_mat, file_name), 'r') as f:
    cnt = f.get('cnt')
    #T_cnt = np.array(cnt.get('T'))
    #clab = np.array(cnt.get('clab'))  # clab is the electrode names
    #file = np.array(cnt.get('file'))
    #fs = np.array(cnt.get('fs'))  # fs is the sampling rate
    #title = np.array(cnt.get('title'))
    x = np.array(cnt.get('x'))  # x is the continuous multivariate data

# Extract data mrk - contains information on the EVENTS
with h5py.File(os.path.join(path_data_mat, file_name), 'r') as f:
    mrk = f.get('mrk')
    #className = np.array(mrk.get('className'))  # are the names of the different types of events
    #event = mrk.get('event')  #
    #react = np.array(event.get('react'))
    time_mrk = np.array(mrk.get('time'))  # mrk.time is the timestamp for each event in milliseconds
    y = np.array(mrk.get('y'))  # mrk.y is a binary matrix telling which of the five types each event is

car_brake_y = time_mrk.T*y[:, 1]  # contains the time in ms of events car_brake
events_y = np.array(car_brake_y[car_brake_y > 0]/5).astype(int)
noevents_int = np.append(0, events_y)
noevents_len = np.append(events_y[0], np.diff(events_y))



# # Construct matrix for all target segments for channel chn_name
# ts_i = 60  # Target segment interval intial point -> corresponds to 300ms
# ts_f = 240  # Target segment interval intial point -> corresponds to 1200ms
# gap_sml = 600  # Non-targets data blocks (1500 ms duration, 500 ms equidistant offset) that were at least 3000 ms apart from any stimulus.
# nts_offset = 100  # 500 ms equidistant offset equidistant offset
#
#
# channels_eeg = np.arange(61)
# channels_eeg = np.delete(channels_eeg,[0,5])
# for event_index in range(len(events_y)):
#     event_time = events_y[event_index]
#     A = x[channels_eeg, event_time - ts_i:event_time + ts_f]
#     A_norm = np.subtract(A, np.array(A[:, :20].mean(1)).reshape(A.shape[0], 1))
#     if(event_time < int(x.shape[1]/2)):
#         np.save(os.path.join(path_data_save, "train", '%s_segment_%d_event.npy' % (file_name, event_index), A_norm))
#         # np.save(os.path.join(path_data_save, "train", file_name + '_segment_' + str(event_index) + "_event.npy"), _norm)
#     else:
#         np.save(os.path.join(path_data_save, "test", '%s_segment_%d_event.npy' % (file_name, event_index), A_norm))
#         # np.save(path_data_save + "test/" + file_name + '_segment_' + str(event_index) + "_event.npy", A_norm)
#
#     noevents_seg = int((noevents_len[event_index] - gap_sml*2 - ts_i + nts_offset) / ts_)
#     for noevent_ix in range(noevents_seg):
#         noevents_int[event_index] + gap_sml
#         A = x[channels_eeg, event_time - ts_i:event_time + ts_f]












