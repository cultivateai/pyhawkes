import numpy as np


def convert_discrete_to_continuous(S, dt):
    # Convert S to continuous time
    from pybasicbayes.util.general import ibincount
    T = S.shape[0] * dt
    S_ct = dt * np.concatenate([ibincount(Sk) for Sk in S.T]).astype(float)
    S_ct += dt * np.random.rand(*S_ct.shape)
    assert np.all(S_ct < T)
    C_ct = np.concatenate([k*np.ones(Sk.sum()) for k, Sk in enumerate(S.T)]).astype(int)

    # Sort the data
    perm = np.argsort(S_ct)
    S_ct = S_ct[perm]
    C_ct = C_ct[perm]
    return S_ct, C_ct, T


def convert_continuous_to_discrete(S, C, dt, T_min, T_max):
    bins = np.arange(T_min, T_max, dt)
    if bins[-1] != T_max:
        bins = np.hstack((bins, [T_max]))
    T = bins.size - 1

    K = C.max()+1
    S_dt = np.zeros((T, K))
    for k in range(K):
        S_dt[:, k] = np.histogram(S[C == k], bins)[0]

    assert S_dt.sum() == len(S)
    return S_dt


def logistic(x, lam_max=1.0):
    return lam_max*1.0/(1.0+np.exp(-x))


def logit(x, lam_max=1.0):
    return np.log(x/lam_max)-np.log(1-(x/lam_max))


def sample_nig(mu0, lmbda0, alpha0, beta0):
    mu0, lmbda0, alpha0, beta0 = np.broadcast_arrays(mu0, lmbda0, alpha0, beta0)
    shp = mu0.shape
    assert lmbda0.shape == alpha0.shape == beta0.shape == shp
    tau = np.array(np.random.gamma(alpha0, 1./beta0)).reshape(shp)
    mu = np.array(np.random.normal(mu0, np.sqrt(1./(lmbda0 * tau)))).reshape(shp)
    return mu, tau
