import numpy as np
from scipy.special import psi, gammaln, multigammaln, digamma
import scipy
from visualize import *


class Dirichlet(object):
    def __init__(self, a):
        self.a = a


class Wishart(object):
    def __init__(self, w, v):
        self.w = w
        self.v = v


class Normal(object):
    def __init__(self, m, b):
        self.m = m
        self.b = b


class Params(object):
    def __init__(self, a, w, v, m, b):
        self.a = a
        self.w = w
        self.v = v
        self.m = m
        self.b = b


def cal_xbar(r, x):

    num, nk = r.shape
    num, ndim = x.shape

    xbar = np.zeros((nk, ndim))

    for k in range(nk):
        for n in range(num):
            xbar[k] += r[n, k] * x[n]

    return xbar


def cal_s(r, x, xbar, N):

    num, nk = r.shape
    num, ndim = x.shape

    s = [np.zeros((ndim, ndim)) for _ in range(nk)]

    for k in range(nk):
        for n in range(num):
            s[k] += r[n, k] * np.outer(x[n]-xbar[k], x[n]-xbar[k])/N[k]

            #print('sk:', s[k])
            #print('xbar:', xbar[k])
            #print('xn:', x[n])
            #print('rnk:', r[n, k])

    return s



def cal_w_inv(wo, s, bo, N, xbar, mo):

    nk, ndim = xbar.shape
    wo_inv = scipy.linalg.inv(wo)

    w_inv = []#np.zeros(nk)

    for k in range(nk):
        Ns = N[k] * s[k]
        b = bo*N[k]/(bo+N[k])
        w_inv.append(wo_inv + Ns + b * np.outer(xbar[k]-mo, xbar[k]-mo))

        #print('wo-1:', wo_inv)
        #print('xbar:', xbar[k])
        #print('mo:', mo)
        #print('winv-1:', w_inv[-1])

    return w_inv


def cal_log_pi(a):

    return psi(a)/psi(np.sum(a))


def cal_log_lambda(v, w, D):

    lmbda = np.zeros(len(w))

    for k in range(len(lmbda)):
        temp = 0
        for i in range(D):
            temp += psi((v[k] - i)/2)
        lmbda[k] = temp + D*np.log(2) + np.log(scipy.linalg.det(w[k]))

        #print('lambdak:', lmbda[k])

    return lmbda


def quad(A, x):

    return np.dot(np.matmul(A, x), x)



def cal_quad(b, x, m, w, v):

    num, ndim = x.shape
    D = ndim
    nk, ndim = m.shape

    E_nw = np.zeros((num, nk))
    for k in range(nk):
        for n in range(num):
            E_nw[n, k] = (D/b[k]) + v[k]*quad(w[k], x[n]-m[k])

        #print('E_nw_k:', E_nw[k])

    return E_nw


def cal_rho(x, E_log_lambda, E_quad, E_log_pi):

    num, D = x.shape
    nk = len(E_log_pi)
    rho = np.zeros((num, nk))
    for n in range(num):
        for k in range(nk):
            rho[n, k] = E_log_pi[k] + (1/2) * E_log_lambda[k] - (D/2) * np.log(2*np.pi) - (1/2) * E_quad[n, k]
            #print('E_log_pi_k;', E_log_pi[k])
            #print('E_log_lambda_k;', E_log_lambda[k])
            #print('E_quad_k;', E_quad[k])
            #print(rho[n, k])

        #print('rho_n:', rho[n,:])

    return rho


def inference(x, Dirichlet, Wishart, Normal, n_classes=7, num_iter=18):


    #m_sequence = []
    #s_sequence = []
    #w_sequence = []
    #v_sequence = []
    #b_sequence = []
    params_sequence = []


    num, ndim = x.shape
    ao = Dirichlet.a
    wo = Wishart.w
    vo = Wishart.v
    bo = Normal.b
    mo = Normal.m

    #E step
    rho = np.random.rand(num, n_classes)
    r = rho / np.sum(rho, 1, keepdims=True)

    for iter in range(num_iter):
        #M step
        N = np.sum(r, 0)# + 1e-100
        a = ao + N
        b = bo + N

        xbar = cal_xbar(r, x)/np.expand_dims(N, axis=1)
        m = bo*mo + xbar*np.expand_dims(N, axis=1)
        m = m / np.expand_dims(b, axis=1)
        s = cal_s(r, x, xbar, N)/np.expand_dims(np.expand_dims(N, axis=1), axis=2)
        w_inv = cal_w_inv(wo, s, bo, N, xbar, mo)
        w = np.linalg.inv(w_inv)
        v = vo + N + 1

        p = Params(a=None, w=None, v=None, m=None, b=None)
        p.a = a
        p.b = b
        p.m = m
        p.w = w
        p.v = v

        params_sequence.append(p)

        #E step
        D = ndim
        E_log_pi = cal_log_pi(a)
        E_log_lambda = cal_log_lambda(v, w, D)
        E_quad = cal_quad(b, x, m, w, v)
        log_rho = cal_rho(x, E_log_lambda, E_quad, E_log_pi)
        rho = np.exp(log_rho)
        r = rho / np.sum(rho, 1, keepdims=True)

    print(w)
    return  params_sequence

if __name__ == '__main__':
    np.random.seed(0)

    n_classes = 7
    Diri = Dirichlet(a = np.ones(n_classes)*1e-5)
    Wish = Wishart(w = np.eye(2), v = 1.0)
    Norm = Normal(m = np.zeros(2), b = 1.0e-3)
    x = np.vstack((np.random.multivariate_normal([-10, -10], np.array([[1.0, 0.0],[0.0, 1.0]]), 20),
                   np.random.multivariate_normal([0, 0], np.array([[1.0, 0.0],[0.0, 1.0]]), 20),
                   np.random.multivariate_normal([10, 10], np.array([[1.0, 0.0],[0.0, 1.0]]), 20)))
    #print(x)
    params = inference(x, Diri, Wish, Norm)
    plot_q_it(params, x, xlim=None, ylim=None, n=None, s=10, ncols=3)
