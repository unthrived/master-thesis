'''EEG Dataset'''
import os
import mne
import numpy as np
import decoding_toolbox_py.Helper_funcs.DecToolbox as dt



'''VARIBLES'''

amount_of_subjects = 26 # Change the range so the process is faster
if amount_of_subjects > 26: amount_of_subjects = 26
subjs_list = ['s{:02d}'.format(i) for i in range(1, amount_of_subjects+1) if i != 6 ] 
nSubj = len(subjs_list)

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
def read_data(
        repetitions=False,
        repetition_index=1,
        resample=False,
        resample_frequency = 20,
        amount_of_subjects = 1,
        task = 'main'
        ):
    if amount_of_subjects < 6:
        subjs_list = ['s{:02d}'.format(i) for i in range(1, amount_of_subjects+1)] 
    elif amount_of_subjects > 26:
        amount_of_subjects = 26
        subjs_list_5 = ['s{:02d}'.format(i) for i in range(1, 6)]
        subjs_list = ['s{:02d}'.format(i) for i in range(7, amount_of_subjects+2)]
        subjs_list = subjs_list_5 + subjs_list
    else:
        subjs_list_5 = ['s{:02d}'.format(i) for i in range(1, 6)]
        subjs_list = ['s{:02d}'.format(i) for i in range(7, amount_of_subjects+2)]
        subjs_list = subjs_list_5 + subjs_list

    path = 'Cond_CJ_EEG'

    epochs = []
    all_epochs = []
    all_rawdata = []
    all_st_epochs = []
    all_st_rawdata = []
    for subject_id in subjs_list:
        preproc_path = os.path.join(path, subject_id)

        if task == 'main':
            epoch = mne.read_epochs(os.path.join(preproc_path, 'main_epo.fif'), verbose=False)
            
            if resample: 
                print('Frequency before:', epoch.info['sfreq'])
                epoch = epoch.resample(resample_frequency)
                print('Frequency after:' ,epoch.info['sfreq'])

            # epochs.append(epoch.average())
            all_epochs.append(epoch)
            all_rawdata.append({
                'epoch_dat': epoch.get_data(), 
                'metadata': epoch.metadata
                })
            
        if task == 'stim':
        
            st_epoch = mne.read_epochs(os.path.join(preproc_path, 'mainstim_epo.fif'), verbose=False)
            # print(st_epoch.info['sfreq'])
            if resample: 
                print('Frequency before:', st_epoch.info['sfreq'])
                st_epoch = st_epoch.resample(resample_frequency)
                print('Frequency after:' ,st_epoch.info['sfreq'])
                
            all_st_epochs.append(st_epoch)
            if repetitions:
                all_st_rawdata.append({
                    'epoch_dat': st_epoch.get_data()[st_epoch.metadata['nrep'] == repetition_index,:,:] ,
                    'metadata': st_epoch.metadata[st_epoch.metadata['nrep'] == repetition_index]
                    })
            else:
                all_st_rawdata.append({'epoch_dat': st_epoch.get_data(), 'metadata': st_epoch.metadata})
    if task == 'main':
        return all_epochs, all_rawdata
    else:
        return all_st_epochs, all_st_rawdata

def train_stim_ori(all_st_rawdata, raw_predicts = False):
    '''Train procedure used for forward encoding model'''
    nSubj = len(all_st_rawdata)
    preds = [None] * nSubj
    G = [None] * nSubj

    for subj in range(nSubj):
        X = all_st_rawdata[subj]['epoch_dat']
        if subj == 1: print(X.shape)
        X = np.einsum('kji->jik', X)        
        if subj == 1: print(X.shape)
        
        X = np.delete(X, 25, axis=0)

        y = np.array(all_st_rawdata[subj]['metadata'].orient)
        y = np.digitize(y, bins = np.array(angles))-1.
        phi = y * (180./numC)
        numF, numT, numN = X.shape

        G[subj] = y.copy() 
        
        CONDS = np.unique(G[subj])
        nConds = CONDS.size
        nfold = 5
        FoldsIdx = dt.CreateFolds(G[subj], X, nfold)
        
        design, sortedesign = dt.stim_features(phi, cfg_stim)
        
        Xhat = np.zeros([numC,numN, numT])
        for it in range(numT):
            cfg = dict()
            cfg['cfgE'] = {'gamma': 0.01, 'demean' : True, 'returnPattern' : True}
            cfg['cfgD'] = {'demean' : 'traindata'}
            Xhat[:,:,it] = dt.CV_encoder(design, X, it, cfg, FoldsIdx)
        preds[subj] = Xhat
    if raw_predicts:
        return preds
    else:
        m_centered = np.zeros((numC,numC, numT, nSubj))
        for ival, isubj in enumerate(subjs_list):
            Xhat = preds[ival]
            Xhat_centered = 0*Xhat.copy()
            
            for ic in range(numC): # here trials that match similar label orientation are shifted together y positions (np.roll)
                Xhat_centered[:, G[ival] == ic,:] = np.roll(Xhat[:,G[ival] == ic,:], -ic, axis = 0)
                m_centered[:,ic, :, ival] =  np.mean( Xhat_centered[:,  G[ival] == ic, :], axis = 1)

        Xhat_centeredmean = np.mean( m_centered, axis = 1)
        Xhat_centeredmean = np.mean( Xhat_centeredmean, axis = 2)
        
        return Xhat_centeredmean
    
def train_condv(all_rawdata, raw_predicts = False):
    nSubj = len(all_rawdata)
    preds = [None] * nSubj
    G = [None] * nSubj

    for subj in range(nSubj):
        
        X = all_rawdata[subj]['epoch_dat']
        if subj == 1: print(X.shape)
        X = np.einsum('kji->jik', X)        
        if subj == 1: print(X.shape)
        
        X = np.delete(X, 25, axis=0)
        all_rawdata[subj]['metadata']['condv'] = 'C'
        all_rawdata[subj]['metadata'].loc[all_rawdata[subj]['metadata']['cond']== 1, 'condv'] =  'D'

        y = np.array(all_rawdata[subj]['metadata'].condv)
        phi = y
        numF, numT, numN = X.shape

        G[subj] = y.copy() 
        
        CONDS = np.unique(G[subj])
        nConds = CONDS.size
        nfold = 5
        FoldsIdx = dt.CreateFolds(G[subj], X, nfold)
        
        design, sortedesign = dt.stim_features(phi, cfg_stim)
        
        Xhat = np.zeros([numC,numN, numT])
        for it in range(numT):
            cfg = dict()
            cfg['cfgE'] = {'gamma': 0.01, 'demean' : True, 'returnPattern' : True}
            cfg['cfgD'] = {'demean' : 'traindata'}
            Xhat[:,:,it] = dt.CV_encoder(design, X, it, cfg, FoldsIdx)
        preds[subj] = Xhat
    if raw_predicts:
        return preds
    else:
        m_centered = np.zeros((numC,numC, numT, nSubj))
        for ival, isubj in enumerate(subjs_list):
            Xhat = preds[ival]
            Xhat_centered = 0*Xhat.copy()
            
            for ic in range(numC): # here trials that match similar label orientation are shifted together y positions (np.roll)
                Xhat_centered[:, G[ival] == ic,:] = np.roll(Xhat[:,G[ival] == ic,:], -ic, axis = 0)
                m_centered[:,ic, :, ival] =  np.mean( Xhat_centered[:,  G[ival] == ic, :], axis = 1)

        Xhat_centeredmean = np.mean( m_centered, axis = 1)
        Xhat_centeredmean = np.mean( Xhat_centeredmean, axis = 2)
        
        return Xhat_centeredmean
    
def train_main_ori(all_rawdata, raw_predicts = False, use_orientation = 0):
    nSubj = len(all_rawdata)
    preds = [None] * nSubj
    G = [None] * nSubj
    orientations = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6']
    for subj in range(nSubj):
        
        X = all_rawdata[subj]['epoch_dat']
        #print(X.shape)
        X = np.einsum('kji->jik', X)        
        #print(X.shape)
        
        X = np.delete(X, 25, axis=0)

        y = np.array(all_rawdata[subj]['metadata'][orientations[use_orientation]])
        y = y*180./np.pi
        y = np.digitize(y, bins = np.array(angles))-1.
        phi = y * (180./numC)
        numF, numT, numN = X.shape

        G[subj] = y.copy() 
        
        CONDS = np.unique(G[subj])
        nConds = CONDS.size
        nfold = 5
        FoldsIdx = dt.CreateFolds(G[subj], X, nfold)
        
        design, sortedesign = dt.stim_features(phi, cfg_stim)
        
        Xhat = np.zeros([numC,numN, numT])
        for it in range(numT):
            cfg = dict()
            cfg['cfgE'] = {'gamma': 0.01, 'demean' : True, 'returnPattern' : True}
            cfg['cfgD'] = {'demean' : 'traindata'}
            Xhat[:,:,it] = dt.CV_encoder(design, X, it, cfg, FoldsIdx)
        preds[subj] = Xhat
    if raw_predicts:
        return preds
    else:
        m_centered = np.zeros((numC,numC, numT, nSubj))
        for ival, isubj in enumerate(subjs_list):
            Xhat = preds[ival]
            Xhat_centered = 0*Xhat.copy()
            
            for ic in range(numC): # here trials that match similar label orientation are shifted together y positions (np.roll)
                Xhat_centered[:, G[ival] == ic,:] = np.roll(Xhat[:,G[ival] == ic,:], -ic, axis = 0)
                m_centered[:,ic, :, ival] =  np.mean( Xhat_centered[:,  G[ival] == ic, :], axis = 1)

        Xhat_centeredmean = np.mean( m_centered, axis = 1)
        Xhat_centeredmean = np.mean( Xhat_centeredmean, axis = 2)
        
        return Xhat_centeredmean
