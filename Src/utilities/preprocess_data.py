# Imports
import h5py
import numpy as np
import os
from sklearn.utils import resample

path_data_mat = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data/raw/'))
# path_data_save = os.path.normpath(
#     os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_events/'))

# path_data_save = os.path.normpath(
#     os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_events_noevents/'))

path_data_save = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../../data/post_processed/AllSubjects_events_noevents_balanced/'))

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
    events_y = np.array(car_brake_y[car_brake_y > 0] / 5).astype(int)
    noevents_int = np.append(0, events_y)
    noevents_len = np.append(events_y[0], np.diff(events_y))

    # Construct matrix for all target segments for channel chn_name
    ts_i = 60  # Target segment interval intial point -> corresponds to 300ms
    ts_f = 260  # Target segment interval intial point -> corresponds to 1200ms
    gap_sml = 600  # Non-targets data blocks (1500 ms duration, 500 ms equidistant offset) that were at least 3000 ms apart from any stimulus.
    nts_offset = 100  # 500 ms equidistant offset equidistant offset

    channels_eeg = np.arange(61)
    channels_eeg = np.delete(channels_eeg, [0, 5])
    for event_index in range(len(events_y)):
        event_time = events_y[event_index]
        A = x[channels_eeg, event_time - ts_i:event_time + ts_f]
        A_norm = np.subtract(A, np.array(A[:, :20].mean(1)).reshape(A.shape[0], 1))
        path_out = os.path.join(path_data_save, '%s_segment_%d_event.npy' % (file_name, event_index))
        if os.path.isfile(path_out):
            continue
        np.save(path_out, A_norm)

    ### UNCOMMENT to get no_events
    gap_sml = 600  # 3000 ms apart from any stimulus
    nts_offset = 100  # 500 ms

    non_events = []
    i = 0
    while i < len(events_y):
        non_event_temp = np.random.randint(ts_i, x.shape[1]-ts_f)
        if (len(np.where(abs(events_y - non_event_temp) * 5 <= 4200)[0]) == 0): # 4200 corresponds to: 300ms + 1200ms
            non_events.append(non_event_temp)
            i += 1

    for noevent_index in range(len(non_events)):
        event_time = non_events[noevent_index]
        A = x[channels_eeg, event_time - ts_i:event_time + ts_f]
        A_norm = np.subtract(A, np.array(A[:, :20].mean(1)).reshape(A.shape[0], 1))
        np.save(os.path.join(path_data_save, '%s_segment_%d_noevent.npy' % (file_name, noevent_index)), A_norm)




# This it to find ALL noevents that satify the condition of being gap_sml appart from any stimulus
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
