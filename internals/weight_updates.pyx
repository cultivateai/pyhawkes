# Cythonized updates for the parent variables
#
# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=False
# cython: language_level=3


import numpy as np
cimport numpy as np
cimport cython
# cimport scipy.linalg.cython_blas as cblas

from libc.math cimport log, exp
from libc.stdlib cimport malloc, free

from cython.parallel import prange


cpdef inline double process_log_likelihood(int K, double[::1] A_col, double lambda0_k,
                                    double[::1] Wk, double[:,::1] lambda_ir_k,
                                    double[::1] Ns, double T):
    cdef double ll = 0

    # - \int lambda_k(t) dt
    ll -= lambda0_k * T
    ll -= np.multiply(A_col,Wk).dot(Ns)

    # + \sum_n log(lambda(s_n))
    ll += np.log(lambda0_k + np.sum(np.multiply(A_col,lambda_ir_k), axis=1)).sum()

    return ll


cpdef inline double process_log_likelihood_nogil(int K, double[::1] A_col, double lambda0_k,
                                    double[::1] Wk, double[:,::1] lambda_ir_k,
                                    double[::1] Ns, double T) nogil:
    cdef int num_ir = lambda_ir_k.shape[0]
    cdef double *denom = <double *> malloc(num_ir * cython.sizeof(double))
    cdef double ll = 0
    for i in xrange(num_ir):
        denom[i] = lambda0_k

    ll -= lambda0_k * T

    # ll -= (A_col * Wk).dot(Ns)
    # ll += np.log(lambda0_k + np.sum(A_col * lmbda_ir[C==k,:], axis=1)).sum()
    for k in range(K):
        if A_col[k] == 0:
            continue
        ll -= Wk[k] * Ns[k]
        for ir in range(num_ir):
            denom[ir] += lambda_ir_k[ir,k]
    for ir in range(num_ir):
        ll += log(denom[ir])

    free(denom)

    return ll


cpdef expected_A_col(double[::1] pk, int K, double[::1] A_col_expected,
                     double[::1] A_mask, double[::1] Wk,
                     double lambda0_k, double[:,::1] lambda_ir_k,
                     double[::1] Ns, double T):
    cdef double ll0, ll1, lp0, lp1, Z
    cdef int num_ir = lambda_ir_k.shape[0]

    for k1 in range(K):

        # Edge not allowed
        if A_mask[k1] == 0:
            A_col_expected[k1] = 0.
            continue

        # Will be no variation in log likelihoods
        if Ns[k1] == 0 and num_ir == 0:
            A_col_expected[k1] = 0.5
            continue

        # Compute the log likelihood of the events given W and A=0 or A=1
        A_mask[k1] = 0
        ll0 = process_log_likelihood(K, A_mask, lambda0_k, Wk, lambda_ir_k, Ns, T)
        A_mask[k1] = 1
        ll1 = process_log_likelihood(K, A_mask, lambda0_k, Wk, lambda_ir_k, Ns, T)

        # Sample A given conditional probability
        lp0 = ll0 + log(1.0 - pk[k1])
        lp1 = ll1 + log(pk[k1])
        Z = np.logaddexp(lp0,lp1)
        # Z = log(exp(lp0) + exp(lp1))

        A_col_expected[k1] = exp(lp1 - Z)


cpdef resample_A_col(double[::1] pk, int K, double[::1] A_col, double[::1] u_k,
                     double[::1] Amask, double[::1] Wk,
                     double lambda0_k, double[:,::1] lambda_ir_k,
                     double[::1] Ns, double T):
    cdef double ll0, ll1, lp0, lp1, Z
    cdef int num_ir = lambda_ir_k.shape[0]

    for k1 in range(K):

        if Amask[k1] == 0:
            A_col[k1] = 0
            continue

        if pk[k1] == 0:
            A_col[k1] = 0
            continue

        if pk[k1] == 1:
            A_col[k1] = 1
            continue

        # Will be no variation in log likelihoods
        if Ns[k1] == 0 and num_ir == 0:
            A_col[k1] = u_k[k1] < 0.5
            continue

        # Compute the log likelihood of the events given W and A=0 or A=1
        A_col[k1] = 0
        ll0 = process_log_likelihood(K, A_col, lambda0_k, Wk, lambda_ir_k, Ns, T)
        A_col[k1] = 1
        ll1 = process_log_likelihood(K, A_col, lambda0_k, Wk, lambda_ir_k, Ns, T)

        # Sample A given conditional probability
        lp0 = ll0 + log(1.0 - pk[k1])
        lp1 = ll1 + log(pk[k1])
        Z = np.logaddexp(lp0,lp1)
        # Z = log(exp(lp0) + exp(lp1))

        A_col[k1] = log(u_k[k1]) < lp1 - Z
