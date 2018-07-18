import numpy as np
from scipy.special import psi, gammaln, multigammaln, digamma
import copy
import scipy


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


def cal_xbar(r, x):

    num, nk = r.shape
    num, ndim = x.shape

    xbar = np.zeros((nk, ndim))

    for k in range(nk):
        for n in range(num):
            xbar[k] += r[n,k] * x[n]

    return xbar


def cal_s(r, x, xbar):

    num, nk = r.shape
    num, ndim = x.shape

    s = np.zeros(nk)

    for k in range(nk):
        for n in range(num):
            s[k] += r[n,k] * np.sum((x[n]-xbar[k])**2)

    return s



def cal_w_inv(wo, s, bo, N, xbar, mo):

    nk, ndim = xbar.shape
    wo_inv = scipy.linalg.inv(wo)

    w = np.zeros(nk)

    for k in range(nk):
        Ns = N[k] * s[k]
        b = bo*N[k]/(bo+N[k])
        print(wo_inv)
        print(np.outer(xbar[k]-mo, xbar[k]-mo))
        print(N[k])
        print(s[k])
        w[k] = wo_inv + Ns + b * np.outer(xbar[k]-mo, xbar[k]-mo)

    return w


def cal_log_pi(a):

    return psi(a)/psi(np.sum(a))


def cal_log_lambda(v, w, D):

    lmbda = np.zeros(len(w))

    for k in range(len(lmbda)):
        temp = 0
        for i in range(D):
            temp += psi((v[k] + 1 - i)/2)
        lmbda[k] = temp + D*np.log(2)/np.log(np.e) + np.log(scipy.linalg.det(w[k]))/np.log(np.e)

    return lmbda


def quad(A,x):

    return np.dot(np.matmul(A,x),x)



def cal_quad(b, x, m, w, v):

    num, ndim = x.shape
    D = ndim
    nk, ndim = m.shape

    E_nw = np.zeros(nk)
    for k in range(nk):
        temp = 0
        for n in range(num):
            temp += D*np.linalg.inv(b[k]) + v[k]*quad(w[k], x[n]-m[k])
        E_nw[k] = temp

    return E_nw


def inference(x, n_classes, Dirichlet, Wishart, Normal, num_classes=10, num_iter=10):

    num, ndim = x.shape
    ao = Dirichlet.a
    wo = Wishart.w
    vo = Wishart.v
    bo = Normal.b
    mo = Normal.m

    #E step
    rho = np.random.rand(num, n_classes)
    r = rho / np.sum(rho,1,keepdims=True)

    for iter in range(num_iter):
        #M step
        N = np.sum(r, 0)
        a = ao + N
        b = bo + N

        xbar = cal_xbar(r, x)/np.expand_dims(N, axis=1)
        m = bo*mo + cal_xbar(r, x)
        m = m / np.expand_dims(b, axis=1)
        v = vo + N
        s = cal_s(r, x, xbar)/np.expand_dims(N, axis=1)
        w_inv = cal_w_inv(wo, s, bo, N, xbar, mo)
        w = np.linalg.inv(w_inv)
        v = vo + N + 1


        #E step
        D = ndim
        E_log_pi = cal_log_pi(a)
        E_log_lambda = cal_log_lambda(v, w, D)
        E_quad = cal_quad(b, x, m, w, v)

        log_rho = (1/2) * E_log_lambda - (D/2)*np.log(2*np.pi)/np.log(np.e) - (1/2)*cal_quad + E_log_pi
        rho = np.exp(log_rho)
        r = rho / np.sum(rho_nk,1,keepdims=True)




if __name__ == '__main__':
    np.random.seed(0)

    n_classes = 6
    Diri = Dirichlet(np.ones(n_classes)*1e-3)
    Wish = Wishart(w = np.eye(2), v = 1)
    Norm = Normal(m = np.zeros(2), b = 1)
    x = np.vstack((np.random.multivariate_normal([-10, -10], np.array([[1.0, 0.0],[0.0, 1.0]]), 10),
                   np.random.multivariate_normal([0, 0], np.array([[1.0, 0.0],[0.0, 1.0]]), 10),
                   np.random.multivariate_normal([10, 10], np.array([[1.0, 0.0],[0.0, 1.0]]), 10)))
    inference(x, n_classes, Diri, Wish, Norm, num_classes=10, num_iter=10)



