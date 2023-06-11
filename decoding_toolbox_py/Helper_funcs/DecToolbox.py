#    Created by Alexis Pérez Bellido, 2022
import numpy as np


def CreateFolds(X,Y,nfold):
# Creating folds containing test and train indexes. Created by Alexis Pérez Bellido, 2022

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=nfold,shuffle=False)
    Folds = [None] * nfold
    i = 0
    numN = X.shape[0]
    for train_index, test_index  in  skf.split(X = np.zeros(numN), y = X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Folds[i] = { 'train_index': train_index, 'test_index': test_index}
        i += 1

    return Folds



def stim_features(phi, cfg):
#   [design] = stim_features(cfg, phi)
#    Returns hypothetical channel responses given a presented orientation, cf. Brouwer & Heeger.
#
#    phi         Array of length N, where N is the number of trials, that specifies the presented
#                orientation on each trial. Orientation is specified in degrees and is expected to
#                have a range of 0-180.
#    cfg         Configuration dictionary that can possess the following fields:
#                ['NumC']                               The number of hypothetical channels C to use. The
#                                                    channels are equally distributed across the circle,
#                                                    starting at 0.
#                ['Tuning']                        The tuning curve according to which the responses
#                                                    are calculated.
#                ['Tuning'] = 'vonMises'           Von Mises curve. Kappa: concentration parameter.
#                ['Tuning'] = 'halfRectCos'        Half-wave rectified cosine. Kappa: power.
#                ['Tuning'] = [function_handle]    User-specified function that can take a matrix as input,
#                                                    containing angles in radians with a range of 0-pi.
#                ['kappa']                              Parameter(s) to be passed on to the tuning curve.
#                ['offset']                            The orientation of the first channel. (default = 0)
#           
#    design      The design matrix, of size C x N, containing the hypothetical channel responses.
#    sortedesign A sorted version of the design matrix, sorted by the presented orientation to improve model visualization.
#    Created by Pim Mostert, 2016 in Matlab. Exported to Python by Alexis Pérez Bellido, 2022

    kappa = cfg['kappa']
    NumC = cfg['NumC']
    if 'offset' not in cfg:
        offset = 0
    else:
         offset = cfg['offset']

    sort_idx = phi.argsort(axis = 0)

    NumN = phi.size
    phi = phi - offset
    design = np.arange(0,NumC)[np.newaxis,:].T * np.ones([1, NumN]) * 180/NumC
    design = design - np.ones([NumC,1])*phi.T
    design = design * (np.pi/180) # transforming to radians

    if cfg['Tuning'] == 'halfRectCos':
        fn = lambda x: np.abs(np.cos(x))**kappa
    if cfg['Tuning'] == 'vonmises':
        mu = 0
        fn = lambda x: np.exp(kappa*np.cos(2*x-mu))/(2*np.pi*np.i0(kappa)) 
    
    design = fn(design)
    
    sortedesign = design[:, sort_idx] #np.take_along_axis(design, sort_idx, axis = 1).copy()
    
    return [design , sortedesign]



def train_encoder(X, Y, cfg):
#   [decoder] = train_encoder(X, Y, cfg)
#    Trains a linear decoder "beamformer style" to optimally recover the latent components as 
#    prescribed in X. Several decoders may be trained indepedently, corresponding to several
#    latent components.
#
#    X           Vector or matrix of size C x N, where C is the number of components and N is
#                the number of trials, that contains the expected/prescribed component activity
#                in the training data.
#    Y           Matrix of size F x N, where F is the number of features, that contains the
#                training data.
#    cfg         Configuration struct that can possess the following fields:
#                ['gamma'] = [scalar]                Shrinkage regularization parameter, with range [0 1]. 
#                                                 No default given.
#                ['returnPattern'] = 'yes' or 'no'   Whether the spatial patterns of the components should
#                                                 be returned. Default = 'no';
#                ['demean'] = 'yes' or 'no'          Whether to demean the data first (per feature, over
#                                                 trials). Default = 'yes';.
#
#    decoder     The (set of) decoder(s), that may be passed on to an appropriate decoding function,
#                e.g. decode_beamformer. It may contain a field .pattern of size C x F
#
#    See also DECODE_BEAMFORMER.
#    Created by Pim Mostert, 2016 in Matlab. Exported to Python by Alexis Pérez Bellido, 2022 
    Y = Y.T
    X = X.T
    numC = X.shape[1]
    numF = Y.shape[1]
    decoder = dict()
    gamma = cfg['gamma']

    # demean activity in each trial
    if 'demean' not in cfg:
        cfg['demean'] = True

    if 'returnPattern' not in cfg:
        cfg['returnPattern'] = False

    if  cfg['demean']:
        Ym = Y.mean(axis=0, keepdims=True)
        Y = Y - np.repeat(Ym, axis = 0, repeats = Y.shape[0])
        decoder['dmY'] = Ym.T # save demeaned Y for posterior inspection
    
    if cfg['returnPattern']:
        decoder['pattern'] = np.zeros([numF, numC]) 

    decoder['W'] = np.zeros([numC, numF]) # weights empty matrix


    for ic in range(numC):
        # Estimate leadfield for current channel
        l = ((X[:,ic].T @ X[:,ic])**-1) * (X[:,[ic]].T @ Y) 
        if cfg['returnPattern']:
            decoder['pattern'][:, ic] = l
        # Estimate noise (what is not explained by the regressors coefficients)
        N = Y - X[:,[ic]] * l
        # Estimate noise covariance
        S = np.cov(N,rowvar=False).copy() # rowvar is necessary to get the correct covariance matrix shape
        #  Regularize
        S = (1-gamma)*S + gamma*np.eye(numF) * np.trace(S)/numF #% [w,d] = eig(S);  eigenvalues -> pdiag(d)
        decoder['W'][ic, :] = np.dot(l,np.linalg.inv(S))
    return decoder


def test_encoder( decoder, Y, cfg):

#    [Xhat] = decode_beamformer(cfg, decoder, Y)
#    Estimate the activity of latent components using a linear decoder, obtained from an
#    appropriate training function. Several components may be estimated independently.
#
#    decoder     The linear decoder obtained from e.g. train_beamformer.
#    Y           Matrix of size F x N, where F is the number of features and the N the number of trials,
#                that contains the data that is to be decoded.
#    cfg         Configuration struct that can possess the following fields:
#                .demean                          Whether the data should be demeaned (per feature,
#                                                 over trials) prior to decoding. The mean can be
#                                                 specified in the following ways:
#                        = 'trainData'            The mean of the training data (default).
#                        = 'testData'             The mean of the testing data.
#                        = [F x 1] vector         Manually specified mean, where F is the number of
#                                                 features (e.g. sensors).
#                        = 'no'                   No demeaning.
#
#    Xhat        Vector or matrix of size C x N, where C is the number of components, containing
#                the decoded data.
#
#    See also TRAIN_ENCODER.
#    Created by Pim Mostert, 2016 in Matlab. Exported to Python by Alexis Pérez Bellido, 2022 


    NumN = Y.shape[1]
    NumC = decoder['W'].shape[0]
    NumF = Y.shape[0]

    # Demean or not demean
    if cfg['demean'] == 'traindata': # demean test data with mean of training data
        Y = Y - np.repeat(decoder['dmY'], axis = 1, repeats = NumN)
    if cfg['demean'] == 'testdata':
        Ym = Y.mean(axis = 0, keepdims=True)
        Y = Y - np.repeat(Ym, axis = 1, repeats = NumN)

    # Decode
    # Inverting the calculated weights for each channel to decode the stimuli

    decoder['iW'] = 0*decoder['W'].copy() # empty array to store the inverted weights

    for ic in range(NumC):
        W = decoder['W'][ic,:]
        decoder['iW'][ic,:] = W / (W @ W.T) # inverting the weights

    Xhat = decoder['iW'] @ Y # Predicting stim based on activity multiplying by the inverted decoder -> Y = WX + N, doing this is like calculating X = Y/N
    return Xhat


# CV_encoder(design, Y, cfg, FoldsIdx) 
# Function to perform cross-validation on the decoder using all the data.
def CV_encoder(design, Y, sel_t , cfg, FoldsIdx):

    numC = design.shape[0]
    numN = Y.shape[2]

    Xhat = np.zeros([numC,numN])
    Xhat_centered = 0*Xhat.copy()

    nfold = np.size(FoldsIdx)

    for ifold in range(nfold):
        # Output matrix
        dat = dict()
        dat['Y_train'] = np.squeeze(Y[:,sel_t, FoldsIdx[ifold]['train_index']])
        idesign = design[:,FoldsIdx[ifold]['train_index']]
        dat['Y_test'] = np.squeeze(Y[:,sel_t, FoldsIdx[ifold]['test_index']])

        if 'cfgE' not in cfg:
            cfgE = {'gamma': 0.01, 'demean' : True, 'returnPattern' : True}
        else:
            cfgE = cfg['cfgE']

        decoder = train_encoder(idesign, dat['Y_train'], cfgE)

        if 'cfgD' not in cfg:
            cfgD = {'demean' : 'traindata'}
        else:
            cfgD = cfg['cfgD']


        Xhat[:,FoldsIdx[ifold]['test_index']] = test_encoder( decoder, dat['Y_test'], cfgD) 
    
    return Xhat

# Prepared to perform cross-validation and temporal generalization of the decoder
def CV_TG_encoder(design, Y, sel_t, cfg, FoldsIdx):

    numC = design.shape[0]
    numT = Y.shape[1]
    numN = Y.shape[2]

    # decode numT X numT
    Xhat = np.zeros([numC,numN, numT])
    Xhat_centered = 0*Xhat.copy()

    nfold = np.size(FoldsIdx)

    for ifold in range(nfold):
        # Training encoder
        dat = dict()
        dat['Y_train'] = np.squeeze(Y[:,sel_t, FoldsIdx[ifold]['train_index']])
        idesign = design[:,FoldsIdx[ifold]['train_index']]
        dat['Y_test'] = np.squeeze(Y[:,sel_t, FoldsIdx[ifold]['test_index']])

        if 'cfgE' not in cfg:
            cfgE = {'gamma': 0.01, 'demean' : True, 'returnPattern' : True}
        else:
            cfgE = cfg['cfgE']

        decoder = train_encoder(idesign, dat['Y_train'], cfgE)

        if 'cfgD' not in cfg:
            cfgD = {'demean' : 'traindata'}
        else:
            cfgD = cfg['cfgD']
        # Testing encoding model on all the time points of test data
        for it in range(numT):
            dat['Y_test'] = np.squeeze(Y[:,it, FoldsIdx[ifold]['test_index']])
            Xhat[:,FoldsIdx[ifold]['test_index'],it] = test_encoder( decoder, dat['Y_test'], cfgD) 
    
    return Xhat