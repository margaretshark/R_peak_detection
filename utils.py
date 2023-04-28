import matplotlib.pyplot as plt
import wfdb
import numpy as np
from wfdb import processing
from functools import partial
import torch.nn as nn
import os

def plot_signal_with_r_peaks(ecg, rpeaks, rec_name):
    '''
    Plot the ECG signal with R-peaks
    --------------------------------
    ecg: recodrd from x-laid 
    rpeaks: positions of peaks with types:
            'A' - Atrial premature beat
            '*' or 'N' - Normal beat
            'V' - Premature ventricular contraction 
    rec_name: record naming from data folder
    '''
    
    plt.figure(figsize=(20,10))
    
    plt.plot(ecg)
    plt.plot(rpeaks, ecg[rpeaks],'rx')
    plt.xlabel('Sample number')
    plt.ylabel('ECG signal')
    plt.title(f'ECG signal for record {rec_name} from MIT-BIH Arrhythmia Database')
    
    
def load_patient(data_dir, rec_name):
    
    print('Loading Data for Patient : {}'.format(rec_name))

    # Loading ecg signal for given patient
    record = wfdb.rdrecord(f'{data_dir}/{rec_name}')
    
    ecg = np.asarray(record.p_signal[:,0], dtype=np.float64)

    # Load R peak annotations stored as Sample number
    annotation = wfdb.rdann(f'{data_dir}/{rec_name}', 'atr')
    Rpeaks_pos = annotation.sample[np.in1d(annotation.symbol, 
             ['+', 'A', 'N', 'V'])]
    Rpeaks_pos = np.asarray(Rpeaks_pos)
    

    print('Total Beats : ',str(len(Rpeaks_pos)))

    return ecg, Rpeaks_pos

def extract_test_windows(signal,win_size,stride):
    
    normalize = partial(processing.normalize_bound, lb=-1, ub=1)

    signal = np.squeeze(signal)

    
    pad_sig = np.pad(signal,
                     (win_size - stride, win_size),
                     mode='edge')
    # Lists of data windows and corresponding indices
    data_windows = []
    win_idx = []

    # Indices for padded signal
    pad_id = np.arange(pad_sig.shape[0])


    # Split into windows and save corresponding padded indices
    for win_id in range(0, len(pad_sig), stride):
        if win_id + win_size < len(pad_sig):
            
            window = pad_sig[win_id:win_id+win_size]
            if window.any():
                window = np.squeeze(np.apply_along_axis(normalize, 0, window))

            data_windows.append(window)
            win_idx.append(pad_id[win_id:win_id+win_size])


    data_windows = np.asarray(data_windows)
    data_windows = data_windows.reshape(data_windows.shape[0],
                                        data_windows.shape[1], 1)
    win_idx = np.asarray(win_idx)
    win_idx = win_idx.reshape(win_idx.shape[0]*win_idx.shape[1])

    return win_idx, data_windows

def mean_preds(win_idx, preds, orig_len, win_size, stride):
        """
        Calculate mean of overlapping predictions.
        Function takes window indices and corresponding predictions as
        input and then calculates mean for predictions. One mean value
        is calculated for every index of the original padded signal. At
        the end padding is removed so that just the predictions for
        every sample of the original signal remain.
        Parameters
        ----------
        win_idx : array
            Array of padded signal indices before splitting.
        preds : array
            Array that contain predictions for every data window.
        orig_len : int
            Lenght of the signal that was used to extract data windows.
        Returns
        -------
        pred_mean : int
            Predictions for every point for the original signal. Average
            prediction is calculated from overlapping predictions.
        """
        # flatten predictions from different windows into one vector
        preds = preds.reshape(preds.shape[0]*preds.shape[1])
        assert(preds.shape == win_idx.shape)

        pred_mean = calculate_means(indices=win_idx, values=preds)

        # Remove paddig
        pred_mean = pred_mean[int(win_size-stride):
                              (win_size-stride)+orig_len]

        return pred_mean
    
def mean(win_idx, preds, orig_len, win_size, stride):

        preds = preds.reshape(preds.shape[0]*preds.shape[1])

        assert(preds.shape == win_idx.shape)


        # Combine indices with predictions
        comb = np.column_stack((win_idx, preds))


        comb = comb[comb[:, 0].argsort()]
        split_on = np.where(np.diff(comb[:, 0]) != 0)[0]+1


        mean_values = [arr[:, 1].mean() for arr in np.split(comb, split_on)]

        mean_values = np.array(mean_values)    

        

        # Remove paddig
        pred_mean = mean_values[int(win_size-stride):
                              (win_size-stride)+orig_len]

        return pred_mean