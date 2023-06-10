import numpy as np
from utils import product_matrix_vector


def fast_wilcoxon(X, y=None, zero_method='wilcox', correction=False,
                  n_jobs=-1):
    from mne.parallel import parallel_func

    if y is not None:
        X -= y
    dims = X.shape
    X = X.reshape(len(X), -1)
    parallel, p_time_gen, n_jobs = parallel_func(_loop_wilcoxon, n_jobs)
    n_chunks = np.min([n_jobs, X.shape[1]])
    out = parallel(p_time_gen(X[..., chunk],
                              zero_method=zero_method, correction=correction)
                   for chunk in np.array_split(range(X.shape[1]), n_chunks))
    stats, p_val = map(list, zip(*out))
    stats = np.hstack(stats).reshape(dims[1:])
    p_val = np.hstack(p_val).reshape(dims[1:])
    return stats, p_val


def _loop_wilcoxon(X, zero_method, correction):
    from scipy.stats import wilcoxon
    p_val = np.ones(X.shape[1])
    stats = np.ones(X.shape[1])
    for ii, x in enumerate(X.T):
        stats[ii], p_val[ii] = wilcoxon(x)
    return stats, p_val


def corr_linear_circular(X, alpha):
    # Authors:  Jean-Remi King <jeanremi.king@gmail.com>
    #           Niccolo Pescetelli <niccolo.pescetelli@gmail.com>
    #
    # Licence : BSD-simplified
    """
    Parameters
    ----------
        X : numpy.array, shape (n_angles, n_dims)
            The linear data
        alpha : numpy.array, shape (n_angles,)
            The angular data (if n_dims == 1, repeated across all x dimensions)
    Returns
    -------
        R : numpy.array, shape (n_dims)
            R values
        R2 : numpy.array, shape (n_dims)
            R square values
        p_val : numpy.array, shape (n_dims)
            P values
    Adapted from:
        Circular Statistics Toolbox for Matlab
        By Philipp Berens, 2009
        berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
        Equantion 27.47
    """

    from scipy.stats import chi2
    import numpy as np

    # computes correlation for sin and cos separately
    rxs = repeated_corr(X, np.sin(alpha))
    rxc = repeated_corr(X, np.cos(alpha))
    rcs = repeated_corr(np.sin(alpha), np.cos(alpha))

    # tile alpha across multiple dimension without requiring memory
    if X.ndim > 1 and alpha.ndim == 1:
        rcs = rcs[:, np.newaxis]

    # Adapted from equation 27.47
    R = (rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2)

    # JR adhoc way of having a sign....
    R = np.sign(rxs) * np.sign(rxc) * R
    R2 = np.sqrt(R ** 2)

    # Get degrees of freedom
    n = len(alpha)
    pval = 1 - chi2.cdf(n * R2, 2)

    return R, R2, pval


def corr_circular_linear(alpha, X):
    # Authors:  Jean-Remi King <jeanremi.king@gmail.com>
    #
    # Licence : BSD-simplified
    """
    Parameters
    ----------
        alpha : numpy.array, shape (n_angles,)
            The angular data (if n_dims == 1, repeated across all x dimensions)
        X : numpy.array, shape (n_angles, n_dims)
            The linear data
    Returns
    -------
        R : numpy.array, shape (n_dims)
            R values
        R2 : numpy.array, shape (n_dims)
            R square values
        p_val : numpy.array, shape (n_dims)
            P values
    Adapted from:
        Circular Statistics Toolbox for Matlab
        By Philipp Berens, 2009
        berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
        Equantion 27.47
    """

    from scipy.stats import chi2
    from jr.utils import pairwise
    import numpy as np

    # computes correlation for sin and cos separately
    # WIP Applies repeated correlation if X is vector
    # TODO: deals with non repeated correlations (X * ALPHA)
    if alpha.ndim > 1:
        rxs = repeated_corr(np.sin(alpha), X)
        rxc = repeated_corr(np.cos(alpha), X)
        rcs = np.zeros_like(alpha[0, :])
        rcs = pairwise(np.sin(alpha), np.cos(alpha), func=_loop_corr,
                       n_jobs=-1)
    else:
        # WIP Applies repeated correlation if alpha is vector
        rxs = repeated_corr(X, np.sin(alpha))
        rxc = repeated_corr(X, np.cos(alpha))
        rcs = repeated_corr(np.sin(alpha), np.cos(alpha))

    # Adapted from equation 27.47
    R = (rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2)

    # JR adhoc way of having a sign....
    R = np.sign(rxs) * np.sign(rxc) * R
    R2 = np.sqrt(R ** 2)

    # Get degrees of freedom
    n = len(X)
    pval = 1 - chi2.cdf(n * R2, 2)

    return R, R2, pval


def _loop_corr(X, Y):
    R = np.zeros(X.shape[1])
    for ii, (x, y) in enumerate(zip(X.T, Y.T)):
        R[ii] = repeated_corr(x, y)
    return R


def repeated_corr(X, y, dtype=float):
    """Computes pearson correlations between a vector and a matrix.
    Adapted from Jona-Sassenhagen's PR #L1772 on mne-python.
    Parameters
    ----------
        X : np.array, shape (n_samples, n_measures)
            Data matrix onto which the vector is correlated.
        y : np.array, shape (n_samples)
            Data vector.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.
    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim == 1:
        X = X[:, None]
    shape = X.shape
    X = np.reshape(X, [shape[0], -1])
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')
    if X.ndim == 1:
        X = X[:, None]
    ym = np.array(y.mean(0), dtype=dtype)
    Xm = np.array(X.mean(0), dtype=dtype)
    y -= ym
    X -= Xm
    y_sd = y.std(0, ddof=1)
    X_sd = X.std(0, ddof=1)[:, None if y.shape == X.shape else Ellipsis]
    R = (np.dot(y.T, X) / float(len(y) - 1)) / (y_sd * X_sd)
    R = np.reshape(R, shape[1:])
    # cleanup variable changed in place
    y += ym
    X += Xm
    return R


def repeated_spearman(X, y, dtype=None):
    """Computes spearman correlations between a vector and a matrix.
    Parameters
    ----------
        X : np.array, shape (n_samples, n_measures ...)
            Data matrix onto which the vector is correlated.
        y : np.array, shape (n_samples)
            Data vector.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.
    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    from scipy.stats import rankdata
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim == 1:
        X = X[:, None]
    shape = X.shape
    X = np.reshape(X, [shape[0], -1])
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')

    # Rank
    X = np.apply_along_axis(rankdata, 0, X)
    y = np.apply_along_axis(rankdata, 0, y)
    # Double rank to ensure that normalization step of compute_corr
    # (X -= mean(X)) remains an integer.
    X *= 2
    y *= 2
    X = np.array(X, dtype=dtype)
    y = np.array(y, dtype=dtype)
    R = repeated_corr(X, y, dtype=type(y[0]))
    R = np.reshape(R, shape[1:])
    return R


def corr_circular(ALPHA1, alpha2, axis=0):
    """ Circular correlation coefficient for two circular random variables.
    Input:
    ------
    ALPHA1 : np.array, shape[axis] = n
        The matrix
    alpha2 : np.array, shape (n), or shape == ALPHA1.shape
        Vector or matrix
    axis : int
        The axis used to estimate correlation
    Returns
    -------
    Y : np.array, shape == X.shape
    Adapted from pycircstat by Jean-Remi King :
    1. Less memory consuming than original
    2. supports ALPHA1 as matrix and alpha2 as vector
    https://github.com/circstat/pycircstat
    References: [Jammalamadaka2001]_
    """

    # center data on circular mean
    def sin_center(alpha):
        m = np.arctan2(np.mean(np.sin(alpha), axis=axis),
                       np.mean(np.cos(alpha), axis=axis))
        return np.sin((alpha - m) % (2 * np.pi))

    sin_alpha1 = sin_center(ALPHA1)
    sin_alpha2 = sin_center(alpha2)

    # compute correlation coeffcient from p. 176
    if sin_alpha1.ndim == sin_alpha2.ndim:
        num = np.sum(sin_alpha1 * sin_alpha2, axis=axis)
        den = np.sqrt(np.sum(sin_alpha1 ** 2, axis=axis) *
                      np.sum(sin_alpha2 ** 2, axis=axis))
    else:
        num = np.sum(product_matrix_vector(sin_alpha1, sin_alpha2, axis=axis))
        den = np.sqrt(np.sum(sin_alpha1 ** 2, axis=axis) *
                      np.sum(sin_alpha2 ** 2))
    return num / den


def robust_mean(X, axis=None, percentile=[5, 95]):
    X = np.array(X)
    axis_ = axis
    # force axis to be 0 for facilitation
    if axis is not None and axis != 0:
        X = np.transpose(X, [axis] + range(0, axis) + range(axis+1, X.ndim))
        axis_ = 0
    mM = np.percentile(X, percentile, axis=axis_)
    indices_min = np.where((X - mM[0][np.newaxis, ...]) < 0)
    indices_max = np.where((X - mM[1][np.newaxis, ...]) > 0)
    X[indices_min] = np.nan
    X[indices_max] = np.nan
    m = np.nanmean(X, axis=axis_)
    return m


def fast_mannwhitneyu(Y, X, use_continuity=True, n_jobs=-1):
    from mne.parallel import parallel_func
    X = np.array(X)
    Y = np.array(Y)
    nx, ny = len(X), len(Y)
    dims = X.shape
    X = np.reshape(X, [nx, -1])
    Y = np.reshape(Y, [ny, -1])
    parallel, p_time_gen, n_jobs = parallel_func(_loop_mannwhitneyu, n_jobs)
    n_chunks = np.min([n_jobs, X.shape[1]])
    chunks = np.array_split(range(X.shape[1]), n_chunks)
    out = parallel(p_time_gen(X[..., chunk],
                              Y[..., chunk], use_continuity=use_continuity)
                   for chunk in chunks)
    # Unpack estimators into time slices X folds list of lists.
    U, p_value = map(list, zip(*out))
    U = np.hstack(U).reshape(dims[1:])
    p_value = np.hstack(p_value).reshape(dims[1:])
    AUC = U / (nx * ny)
    # XXX FIXME this introduces a bug
    # # correct directionality of U stats imposed by mannwhitneyu
    # if nx > ny:
    #     AUC = 1 - AUC
    return U, p_value, AUC


def _loop_mannwhitneyu(X, Y, use_continuity=True):
    n_col = X.shape[1]
    U, P = np.zeros(n_col), np.zeros(n_col)
    for ii in range(n_col):
        try:
            U[ii], P[ii] = mannwhitneyu(X[:, ii], Y[:, ii], use_continuity)
        except ValueError as e:
            if e.message == 'All numbers are identical in amannwhitneyu':
                U[ii], P[ii] = .5 * len(X) * len(Y), 1.
            else:
                raise ValueError(e.message)
    return U, P


def dPrime(hits, misses, fas, crs):
    from scipy.stats import norm
    from math import exp, sqrt
    Z = norm.ppf
    hits, misses, fas, crs = float(hits), float(misses), float(fas), float(crs)
    # From Jonas Kristoffer Lindelov : lindeloev.net/?p=29
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5 / (hits + misses)
    halfFa = 0.5 / (fas + crs)

    # Calculate hitrate and avoid d' infinity
    hitRate = hits / (hits + misses)
    if hitRate == 1:
        hitRate = 1 - halfHit
    if hitRate == 0:
        hitRate = halfHit

    # Calculate false alarm rate and avoid d' infinity
    faRate = fas/(fas+crs)
    if faRate == 1:
        faRate = 1 - halfFa
    if faRate == 0:
        faRate = halfFa

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hitRate) - Z(faRate)
    out['beta'] = exp(Z(faRate)**2 - Z(hitRate)**2)/2
    out['c'] = -(Z(hitRate) + Z(faRate))/2
    out['Ad'] = norm.cdf(out['d']/sqrt(2))
    return out


def mannwhitneyu(x, y, use_continuity=True):
    """Adapated from scipy.stats.mannwhitneyu but includes direction of U"""
    from scipy.stats import rankdata, tiecorrect
    from scipy.stats import distributions
    from numpy import asarray
    x = asarray(x)
    y = asarray(y)
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1*n2 + (n1*(n1+1))/2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1*n2 - u1  # remainder is U for y
    T = tiecorrect(ranked)
    if T == 0:
        raise ValueError('All numbers are identical in amannwhitneyu')
    sd = np.sqrt(T * n1 * n2 * (n1+n2+1) / 12.0)

    if use_continuity:
        # normal approximation for prob calc with continuity correction
        z = abs((u1 - 0.5 - n1*n2/2.0) / sd)
    else:
        z = abs((u1 - n1*n2/2.0) / sd)  # normal approximation for prob calc

    return u2, distributions.norm.sf(z)


def nested_analysis(X, df, condition, function=None, query=None,
                    single_trial=False, y=None, n_jobs=-1):
    """ Apply a nested set of analyses.
    Parameters
    ----------
    X : np.array, shape(n_samples, ...)
        Data array.
    df : pandas.DataFrame
        Condition DataFrame
    condition : str | list
        If string, get the samples for each unique value of df[condition]
        If list, nested call nested_analysis.
    query : str | None, optional
        To select a subset of trial using pandas.DataFrame.query()
    function : function
        Computes across list of evoked. Must be of the form:
        function(X[:], y[:])
    y : np.array, shape(n_conditions)
    n_jobs : int
        Number of core to compute the function. Defaults to -1.
    Returns
    -------
    scores : np.array, shape(...)
        The results of the function
    sub : dict()
        Contains results of sub levels.
    """
    import numpy as np
    from jr.utils import pairwise
    if isinstance(condition, str):
        # Subselect data using pandas.DataFrame queries
        sel = range(len(X)) if query is None else df.query(query).index
        X = X.take(sel, axis=0)
        y = np.array(df[condition][sel])
        # Find unique conditions
        values = list()
        for ii in np.unique(y):
            if (ii is not None) and (ii not in [np.nan]):
                try:
                    if np.isnan(ii):
                        continue
                    else:
                        values.append(ii)
                except TypeError:
                    values.append(ii)
        # Subsubselect for each unique condition
        y_sel = [np.where(y == value)[0] for value in values]
        # Mean condition:
        X_mean = np.zeros(np.hstack((len(y_sel), X.shape[1:])))
        y_mean = np.zeros(len(y_sel))
        for ii, sel_ in enumerate(y_sel):
            X_mean[ii, ...] = np.mean(X[sel_, ...], axis=0)
            if isinstance(y[sel_[0]], str):
                y_mean[ii] = ii
            else:
                y_mean[ii] = y[sel_[0]]
        if single_trial:
            X = X.take(np.hstack(y_sel), axis=0)  # ERROR COME FROM HERE
            y = y.take(np.hstack(y_sel), axis=0)
        else:
            X = X_mean
            y = y_mean
        # Store values to keep track
        sub_list = dict(X=X_mean, y=y_mean, sel=sel, query=query,
                        condition=condition, values=values,
                        single_trial=single_trial)
    elif isinstance(condition, list):
        # If condition is a list, we must recall the function to gather
        # the results of the lower levels
        sub_list = list()
        X_list = list()  # FIXME use numpy array
        for subcondition in condition:
            scores, sub = nested_analysis(
                X, df, subcondition['condition'], n_jobs=n_jobs,
                function=subcondition.get('function', None),
                query=subcondition.get('query', None))
            X_list.append(scores)
            sub_list.append(sub)
        X = np.array(X_list)
        if y is None:
            y = np.arange(len(condition))
        if len(y) != len(X):
            raise ValueError('X and y must be of identical shape: ' +
                             '%s <> %s') % (len(X), len(y))
        sub_list = dict(X=X, y=y, sub=sub_list, condition=condition)

    # Default function
    function = _default_analysis if function is None else function

    scores = pairwise(X, y, function, n_jobs=n_jobs)
    return scores, sub_list


def _default_analysis(X, y):
    # from sklearn.metrics import roc_auc_score
    from jr.stats import fast_mannwhitneyu
    # Binary contrast
    unique_y = np.unique(y)
    # if two condition, can only return contrast
    if len(y) == 2:
        y = np.where(y == unique_y[0], 1, -1)
        return np.mean(X * y[:, np.newaxis], axis=0)
    elif len(unique_y) == 2:
        # if two conditions but multiple trials, can return AUC
        # auc = np.zeros_like(X[0])
        _, _, auc = fast_mannwhitneyu(X[y == unique_y[0], ...],
                                      X[y == unique_y[1], ...], n_jobs=1)
        # for ii, x in enumerate(X.T):
        #     auc[ii] = roc_auc_score(y, np.copy(x))
        return auc
    # Linear regression:
    elif len(unique_y) > 2:
        return repeated_spearman(X, y)
    else:
        raise RuntimeError('Please specify a function for this kind of data')


def median_abs_deviation(x, axis=None):
    """median absolute deviation"""
    x = np.asarray(x)
    # transpose selected axis in front
    shape = x.shape
    n_dim = len(shape)
    axis_ = None
    if axis is not None:
        dim_order = np.hstack((axis, np.delete(np.arange(n_dim), axis)))
        x = np.transpose(x, dim_order)
        axis_ = 0

    # compute median
    center = np.median(x, axis=axis_, keepdims=False)
    if len(center):
        center = center[np.newaxis, ...]

    # compute median absolute deviation from median
    mad = np.median(np.abs(x - center), axis=axis_)

    return mad


def cross_correlation_fft(a, b, mode='valid'):
    """Cross correlation between two 1D signals. Similar to np.correlate, but
    faster.
    Parameters
    ----------
    a : np.array, shape(n)
    b : np.array, shape(m)
        If len(b) > len(a), a, b = b, a
    Output
    ------
    r : np.array
        Correlation coefficients. Shape depends on mode.
    """
    from scipy import signal
    a = np.asarray(a)
    b = np.asarray(b)
    if np.prod(a.ndim) > 1 or np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    if len(b) > len(a):
        a, b = b, a
    n = len(a)
    # Pad vector
    c = np.hstack((np.zeros(n/2), b, np.zeros(n/2 + len(a) - len(b) + 1)))
    # Convolution of reverse signal:
    return signal.fftconvolve(c, a[::-1], mode=mode)


def align_signals(a, b):
    """Finds optimal delay to align two 1D signals
    maximizes hstack((zeros(shift), b)) = a
    Parameters
    ----------
    a : np.array, shape(n)
    b : np.array, shape(m)
    Output
    ------
    shift : int
        Integer that maximizes hstack((zeros(shift), b)) - a = 0
    """
    # check inputs
    a = np.asarray(a)
    b = np.asarray(b)
    if np.prod(a.ndim) > 1 or np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    # longest first
    sign = 1
    if len(b) > len(a):
        sign = -1
        a, b = b, a
    r = cross_correlation_fft(a, b)
    shift = np.argmax(r) - len(a) + len(a) / 2
    # deal with odd / even lengths (b doubles in size by cross_correlation_fft)
    if len(a) % 2 and len(b) % 2:
        shift += 1
    if len(a) > len(b) and len(a) % 2 and not(len(b) % 2):
        shift += 1
    return sign * shift


def cross_correlation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.
    `x` and `y` must be one-dimensional numpy arrays with the same length.
    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]
    The return vaue has length 2*maxlag + 1.
    Author: http://stackoverflow.com/questions/30677241
            Warren Weckesser
    """
    from numpy.lib.stride_tricks import as_strided

    def _check_arg(x, xname):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError('%s must be one-dimensional.' % xname)
        return x

    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)


class ScoringAUC():
    """Score AUC for multiclass problems.
    Average of one against all.
    """
    def __call__(self, clf, X, y, **kwargs):
        from sklearn.metrics import roc_auc_score

        # Generate predictions
        if hasattr(clf, 'decision_function'):
            y_pred = clf.decision_function(X)
        elif hasattr(clf, 'predict_proba'):
            y_pred = clf.predict_proba(X)
        else:
            y_pred = clf.predict(X)

        # score
        classes = set(y)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        _score = list()
        for ii, this_class in enumerate(classes):
            _score.append(roc_auc_score(y == this_class,
                                        y_pred[:, ii]))
            if (ii == 0) and (len(classes) == 2):
                _score[0] = 1. - _score[0]
                break
        return np.mean(_score, axis=0)


if __name__ == '__main__':
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import LabelBinarizer
    x = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    # test equal results to sklearn
    for clf in (LogisticRegression, RidgeClassifier, LinearSVC):
        clf = clf(random_state=0)
        cv = KFold(3, random_state=0)
        score_sklearn = cross_val_score(clf, x, y, scoring='roc_auc', cv=cv)
        score_me = cross_val_score(clf, x, y, scoring=ScoringAUC(), cv=cv)
        np.testing.assert_array_almost_equal(score_sklearn, score_me)
    # test works with multiclass
    y = np.random.randint(0, 3, 100)
    score = cross_val_score(clf, x, y, scoring=ScoringAUC(), cv=cv)

    X = LabelBinarizer().fit_transform(y)
    score = cross_val_score(clf, X, y, scoring=ScoringAUC(), cv=cv)
    assert(score.mean(0) == 1.)