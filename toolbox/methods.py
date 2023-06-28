'''Python methods done by Junjie Ji 2023'''
'''Require EEG Dataset'''

import os
import mne
import numpy as np
import random
import decoding_toolbox_py.Helper_funcs.DecToolbox as dt
from sklearn.svm import SVC
# from pyrcn.echo_state_network import ESNClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


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


''' Variables for the decoding toolbox '''

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
    if amount_of_subjects < 6: # we have to do this so the cardinality matches because theres no subject 6
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
                all_st_rawdata.append({
                    'epoch_dat': st_epoch.get_data() , 
                    'metadata': st_epoch.metadata
                    })
    if task == 'main':
        return all_epochs, all_rawdata
        # return all_rawdata
    if task == 'stim':
        return all_st_epochs, all_st_rawdata
        # return all_st_rawdata

def time_labels (
        task = 'main',
        resample = False,
        resample_frequency = 20,
        ):
    path = 'Cond_CJ_EEG'
    subject_id = 's01'
    preproc_path = os.path.join(path, subject_id)
    if task == 'main':
        epoch = mne.read_epochs(os.path.join(preproc_path, 'main_epo.fif'), verbose=False)
        if resample: 
            epoch = epoch.resample(resample_frequency)
        time_labels = epoch.times
    if task == 'stim':
        st_epoch = mne.read_epochs(os.path.join(preproc_path, 'mainstim_epo.fif'), verbose=False)
        if resample: 
            st_epoch = st_epoch.resample(resample_frequency)
        time_labels = st_epoch.times
    return time_labels

def read_data_repetitions(
        repetitions=True,
        repetition_index=1,
        resample=False,
        resample_frequency = 20,
        amount_of_subjects = 1,
        task = 'main'
        ):
    '''Reading data based on repetitions'''
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
            if repetitions:
                if repetition_index == 0:
                    repetition_values = [0]  # Use repetition 0
                else:
                    repetition_values = [1, 2]  # Use repetitions 1 and 2
                all_rawdata.append({
                    'epoch_dat': epoch.get_data()[np.isin(epoch.metadata['nrep'], repetition_values), :, :],
                    'metadata': epoch.metadata[np.isin(epoch.metadata['nrep'], repetition_values)]
                })
            else:
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
                if repetition_index == 0:
                    repetition_values = [0]  # Use repetition 0
                else:
                    repetition_values = [1, 2]  # Use repetitions 1 and 2
                all_st_rawdata.append({
                    'epoch_dat': st_epoch.get_data()[np.isin(st_epoch.metadata['nrep'], repetition_values), :, :],
                    'metadata': st_epoch.metadata[np.isin(st_epoch.metadata['nrep'], repetition_values)]
                })
            else:
                all_st_rawdata.append({'epoch_dat': st_epoch.get_data(), 'metadata': st_epoch.metadata})
    if task == 'main':
        return all_rawdata
    if task == 'stim':
        return all_st_rawdata


def read_data_repetitions_decision(
        repetitions=True,
        repetition_index=1,
        decision_index=0,
        resample=False,
        resample_frequency = 20,
        amount_of_subjects = 1,
        task = 'main'
        ):
    '''Reading data based on repetitions and last decision'''
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
            if repetitions:
                if repetition_index == 0:
                    repetition_values = [0]  # Use repetition 0
                else:
                    repetition_values = [1, 2]  # Use repetitions 1 and 2
                decision_values = [decision_index]
                all_rawdata.append({
                    'epoch_dat': epoch.get_data()[np.isin(epoch.metadata['nrep'], repetition_values)&np.isin(epoch.metadata['deci-1'], decision_values), :, :],
                    'metadata': epoch.metadata[np.isin(epoch.metadata['nrep'], repetition_values)&np.isin(epoch.metadata['deci-1'], decision_values)]
                })
            else:
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
                if repetition_index == 0:
                    repetition_values = [0]  # Use repetition 0
                else:
                    repetition_values = [1, 2]  # Use repetitions 1 and 2
                decision_values = [decision_index]
                all_st_rawdata.append({
                    'epoch_dat': st_epoch.get_data()[np.isin(st_epoch.metadata['nrep'], repetition_values)&np.isin(st_epoch.metadata['deci-1'], decision_values), :, :],
                    'metadata': st_epoch.metadata[np.isin(st_epoch.metadata['nrep'], repetition_values)&np.isin(st_epoch.metadata['deci-1'], decision_values)]
                })
            else: 
                all_st_rawdata.append({'epoch_dat': st_epoch.get_data(), 'metadata': st_epoch.metadata})
    if task == 'main':
        return all_rawdata
    if task == 'stim':
        return all_st_rawdata


def train_stim_ori(all_st_rawdata, raw_predicts = False):
    '''Train procedure used for forward encoding model with st data'''
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

    
def train_main_ori(all_rawdata, raw_predicts = False, use_orientation = 0):
    '''Forward encoding model for the main task (6 orientations)'''
    nSubj = len(all_rawdata)
    preds = [None] * nSubj
    G = [None] * nSubj
    orientations = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6']
    for subj in range(nSubj):
        
        X = all_rawdata[subj]['epoch_dat']
        #print(X.shape)
        X = np.einsum('kji->jik', X)        
        #print(X.shape)
        
        # Important line to delete the eye channel
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
        for subj in range(nSubj):
            Xhat = preds[subj]
            Xhat_centered = 0*Xhat.copy()
            
            for ic in range(numC): # here trials that match similar label orientation are shifted together y positions (np.roll)
                Xhat_centered[:, G[subj] == ic,:] = np.roll(Xhat[:,G[subj] == ic,:], -ic, axis = 0)
                m_centered[:,ic, :, subj] =  np.mean( Xhat_centered[:,  G[subj] == ic, :], axis = 1)

        Xhat_centeredmean = np.mean( m_centered, axis = 1)
        Xhat_centeredmean = np.mean( Xhat_centeredmean, axis = 2)
        
        return Xhat_centeredmean


def train_main_ori_shuffled(all_rawdata, raw_predicts = False, use_orientation = 0):
    '''Forward encoding model for the main task (6 orientations)'''
    nSubj = len(all_rawdata)
    preds = [None] * nSubj
    G = [None] * nSubj
    orientations = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6']
    for subj in range(nSubj):
        
        X = all_rawdata[subj]['epoch_dat']
        #print(X.shape)
        X = np.einsum('kji->jik', X)        
        #print(X.shape)
        
        # Important line to delete the eye channel
        X = np.delete(X, 25, axis=0)

        y = np.array(all_rawdata[subj]['metadata'][orientations[use_orientation]])

        random.shuffle(y)

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
        for subj in range(nSubj):
            Xhat = preds[subj]
            Xhat_centered = 0*Xhat.copy()
            
            for ic in range(numC): # here trials that match similar label orientation are shifted together y positions (np.roll)
                Xhat_centered[:, G[subj] == ic,:] = np.roll(Xhat[:,G[subj] == ic,:], -ic, axis = 0)
                m_centered[:,ic, :, subj] =  np.mean( Xhat_centered[:,  G[subj] == ic, :], axis = 1)

        Xhat_centeredmean = np.mean( m_centered, axis = 1)
        Xhat_centeredmean = np.mean( Xhat_centeredmean, axis = 2)
        
        return Xhat_centeredmean


def train_timepoints(X, y, verbose=False, display_roc=False, acc_only = True):
    '''We use standard random forest classifier to train the data'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    minority = np.unique(y_train,return_counts=True)[1][0]
    mayority = np.unique(y_train,return_counts=True)[1][1]
    # print(mayority/minority)
    class_weight = {
        0: 1.0,  
        1: mayority/minority
    }
    sample_weights = np.array([class_weight[label] for label in y_train])
    
    # clf = CatBoostClassifier(task_type = 'GPU')
    # clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf = RandomForestClassifier(n_jobs=-1)
    # clf = LinearSVC(random_state=0, loss="hinge") # Faster than Random Forest
    clf.fit(X_train, y_train
            ,sample_weight=sample_weights
            )

    y_pred = clf.predict(X_test)    
    if verbose:
        print(classification_report(y_test, y_pred))
        print(np.unique(y_test, return_counts=True))
        print(np.unique(y_pred, return_counts=True))
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    # features = clf.feature_importances_
    if display_roc:
        # I think this is not working
        from sklearn.metrics import roc_curve
        from sklearn.metrics import RocCurveDisplay
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=clf.classes_[1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        return roc_display
    # if np.unique(y_pred, return_counts=True)[1].shape[0] == 1:
    #     unique_pred_0 = 0
    #     unique_pred_1 = np.unique(y_pred, return_counts=True)[1][0]  
    # else:
    #     unique_pred_0 = np.unique(y_pred, return_counts=True)[1][0]
    #     unique_pred_1 = np.unique(y_pred, return_counts=True)[1][1]
    if acc_only:
        return accuracy
    return accuracy, f1, roc #, features

def train_timepoints_svc(X, y, verbose=False, display_roc=False, acc_only = True):
    '''Same as train_timepoints but using SVC instead of Random Forest, faster but worse results'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    minority = np.unique(y_train,return_counts=True)[1][0]
    mayority = np.unique(y_train,return_counts=True)[1][1]
    # print(mayority/minority)
    class_weight = {
        0: 1.0,  
        1: mayority/minority
    }
    sample_weights = np.array([class_weight[label] for label in y_train])
    
    # clf = CatBoostClassifier(task_type = 'GPU')
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf = RandomForestClassifier(n_jobs=-1)
    clf = SVC(kernel='linear', probability=True, random_state=0)
    # clf = LinearSVC(random_state=0, loss="hinge") # Faster than Random Forest
    clf.fit(X_train, y_train
            # ,sample_weight=sample_weights
            )

    y_pred = clf.predict(X_test)    
    if verbose:
        print(classification_report(y_test, y_pred))
        print(np.unique(y_test, return_counts=True))
        print(np.unique(y_pred, return_counts=True))
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    # features = clf.feature_importances_
    if display_roc:
        # I think this is not working
        from sklearn.metrics import roc_curve
        from sklearn.metrics import RocCurveDisplay
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=clf.classes_[1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        return roc_display
    # if np.unique(y_pred, return_counts=True)[1].shape[0] == 1:
    #     unique_pred_0 = 0
    #     unique_pred_1 = np.unique(y_pred, return_counts=True)[1][0]  
    # else:
    #     unique_pred_0 = np.unique(y_pred, return_counts=True)[1][0]
    #     unique_pred_1 = np.unique(y_pred, return_counts=True)[1][1]
    if acc_only:
        return accuracy
    return accuracy, f1, roc #, features

def train_condv(all_rawdata, raw_predicts = False):
    '''not working'''
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