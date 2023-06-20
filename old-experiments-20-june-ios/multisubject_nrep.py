import sys
import time as tic
# %matplotlib qt
from scipy.io import loadmat
import numpy as np
import pandas as pd

import inspect #path_fun = inspect.getfile(StratifiedKFold)
import matplotlib.pyplot as plt
import DecToolbox as dt
import mne
import os
import random

from statsmodels.tsa.stattools import coint

path_utils = '/decoding_toolbox_py/helper_funcs' 
sys.path.append(path_utils)

''' VARIABLES '''

dataset = 'eeg'

amount_of_subjects = 20 # Change the range so the process is faster

number_of_repetition = 0

numC = 8
angles = [i * 180./numC for i in range(numC)]
x_labels = np.array(angles)

resample = False # speeds up the procees but showing worse results overall
if resample: resample_frequency = 20 # in Hz, original freq is 500Hz

cfg_stim = dict()
cfg_stim['kappa'] = 4
cfg_stim['NumC'] = numC
cfg_stim['Tuning'] = 'vonmises'
# cfg_stim['Tuning'] = 'halfRectCos'
cfg_stim['offset'] = 0

cfg_train = dict()
cfg_train['gamma'] = 0.1
cfg_train['demean'] = True
cfg_train['returnPattern'] = True

cfg_test = dict()
cfg_test['demean'] = 'traindata'

'''EEG Dataset'''

if amount_of_subjects > 26: amount_of_subjects = 26
subjs_list = ['s{:02d}'.format(i) for i in range(1, amount_of_subjects+1) if i != 6 ] 
path = 'Cond_CJ_EEG'

epochs = []
all_epochs = []
all_rawdata = []
all_st_epochs = []
all_st_rawdata = []
for subject_id in subjs_list:
    preproc_path = os.path.join(path, subject_id)
    
    epoch = mne.read_epochs(os.path.join(preproc_path, 'main_epo.fif'), verbose=False)
    epochs.append(epoch.average())
    all_epochs.append(epoch)
    all_rawdata.append({'epoch_dat': epoch.get_data(), 'metadata': epoch.metadata})
    
    st_epoch = mne.read_epochs(os.path.join(preproc_path, 'mainstim_epo.fif'), verbose=False)
    # print(st_epoch.info['sfreq'])
    if resample: 
        print('Frequency before:', st_epoch.info['sfreq'])
        st_epoch = st_epoch.resample(resample_frequency)
        print('Frequency after:' ,st_epoch.info['sfreq'])
        
    all_st_epochs.append(st_epoch)
    all_st_rawdata.append(
        {
        'epoch_dat': st_epoch.get_data()[st_epoch.metadata['nrep'] == number_of_repetition,:,:] ,
        'metadata': st_epoch.metadata[st_epoch.metadata['nrep'] == number_of_repetition]
        }
        )
gvaverage = mne.grand_average(epochs[:])

from DecToolbox import CV_encoder, CreateFolds

nSubj = len(subjs_list)
preds = [None] * nSubj
G = [None] * nSubj

for subj in range(nSubj):
    time = all_st_epochs[subj].times
    label = all_st_epochs[subj].ch_names
    Y = all_st_rawdata[subj]['epoch_dat']
    Y = np.einsum('kji->jik', Y)
    Y = np.delete(Y, 25, axis=0)

    X = np.array(all_st_rawdata[subj]['metadata'].orient)
    X = np.digitize(X, bins = np.array(angles))-1.
    phi = X * (180./numC)
    numF, numT, numN = Y.shape

    G[subj] = X.copy() 
    
    CONDS = np.unique(G[subj])
    nConds = CONDS.size
    nfold = 5
    FoldsIdx = dt.CreateFolds(G[subj], Y, nfold)
    
    design, sortedesign = dt.stim_features(phi, cfg_stim)
    
    Xhat = np.zeros([numC,numN, numT])
    for it in range(numT):
        cfg = dict()
        cfg['cfgE'] = {'gamma': 0.01, 'demean' : True, 'returnPattern' : True}
        cfg['cfgD'] = {'demean' : 'traindata'}
        Xhat[:,:,it] = CV_encoder(design, Y, it, cfg, FoldsIdx)
    
    preds[subj] = Xhat   

m_centered = np.zeros((numC,numC, numT, nSubj))
for ival, isubj in enumerate(subjs_list):
    Xhat = preds[ival]
    Xhat_centered = 0*Xhat.copy()
    
    for ic in range(numC): # here trials that match similar label orientation are shifted together x positions (np.roll)
        Xhat_centered[:, G[ival] == ic,:] = np.roll(Xhat[:,G[ival] == ic,:], -ic, axis = 0)
        m_centered[:,ic, :, ival] =  np.mean( Xhat_centered[:,  G[ival] == ic, :], axis = 1)

Xhat_centeredmean = np.mean( m_centered, axis = 1)
Xhat_centeredmean = np.mean( Xhat_centeredmean, axis = 2)

max_abs = np.max(np.abs(Xhat_centeredmean))

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
plt.imshow(Xhat_centeredmean,aspect='auto',  vmin=-max_abs, vmax=max_abs )
plt.colorbar()
plt.show()

