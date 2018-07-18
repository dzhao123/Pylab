import numpy as np
from scipy.special import gamma
import scipy.stats as stats



class params(object):
    def __init__(self, mu, k, a, b):
        self.mu = mu
        self.k = k
        self.a = a
        self.b = b



class pposterior(object):
    def __init__(self, params):
        self.params = params

    def pdf_scalar(self, mu, lmbda):
        muo = self.params.mu
        ko = self.params.k
        ao = self.params.a
        bo = self.params.b
        mu_norm = stats.norm(loc=muo, scale=(ko*lmbda)**(-0.5)).pdf(mu)
        lmbda_gamma = stats.gamma(a=ao, scale=1/bo).pdf(lmbda)
        return mu_norm*lmbda_gamma

    def pdf_list(self, mus, lmbdas):
        return [[self.pdf_scalar(mu, lmbda) for mu in mus] for lmbda in lmbdas]

    def pdf(self, mu, lmbda):
        if hasattr(mu, '__iter__'):
            return self.pdf_list(mu, lmbda)
        else:
            return self.pdf_scalar(mu,lmbda)



class qposterior(object):
    def __init__(self, params):
        self.params = params

    def pdf_scalar(self, mu, lmbda):
        mun = self.params.mu
        kn = self.params.k
        an = self.params.a
        bn = self.params.b
        mu_norm = stats.norm(loc=mun, scale=kn**(-0.5)).pdf(mu)
        lmbda_gamma = stats.gamma(a=an, scale=1/bn).pdf(lmbda)
        return mu_norm*lmbda_gamma

    def pdf_list(self, mus, lmbdas):
        return [[self.pdf(mu, lmbda) for mu in mus] for lmbda in lmbdas]

    def pdf(self, mu, lmbda):
            if hasattr(mu, '__iter__'):
                return self.pdf_list(mu, lmbda)
            else:
                return self.pdf_scalar(mu,lmbda)



def ELBO(params):
    return (1/2)*np.log(1/params.k) + np.log(gamma(params.a)) - params.a*np.log(params.b)



def infer_pposterior(data, pprior_params):
    x = data
    n = len(x)
    xm = np.mean(x)
    ns = sum((x - xm)**2)
    muo = pprior_params.mu
    ko = pprior_params.k
    ao = pprior_params.a
    bo = pprior_params.b
    return params(mu = (ko*muo + n*xm)/(ko + n),
                  k = ko + n,
                  a = ao + n/2,
                  b = bo + (1/2)*(ns + (ko*n*(xm - muo)**2)/(ko + n)))



def infer_qposterior(data, pprior_params, init, maxiter=10, eps=1e-100):
    x = data
    xm = np.mean(x)
    n = len(x)
    muo = pprior_params.mu
    ko = pprior_params.k
    ao = pprior_params.a
    bo = pprior_params.b
    mun = init.mu
    kn = init.k
    an = init.a
    bn = init.b
    iter = 0
    params_list = [init]
    cost_list = [ELBO(init)]

    mun = (ko*muo + n*xm)/(ko+n)
    an = ao + (n+1)/2
    while(iter < maxiter):
        kn = (ko+n)*(an/bn)
        bn = bo + ko*((1/kn) + mun**2 + muo**2 - 2*mun*muo) + (1/2)*(sum(x**2) + n*(1/kn + mun**2) - 2*mun*sum(x))
        new_params = params(mu=mun, k=kn, a=an, b=bn)
        params_list.append(new_params)
        cost_list.append(ELBO(new_params))
        if abs(cost_list[iter+1] - cost_list[iter]) < eps:
            break
        iter += 1

    return inference_result(params_list, cost_list)



def sample(mu=0.0, lmba=1.0, size=100):
    return np.random.normal(loc=mu, scale = lmba**(-0.5), size=size)


class inference_result(object):
    def __init__(self, params_list, cost_list):
        self.params = params_list
        self.cost = cost_list
        self.length = len(cost_list)

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        return result_iterator(self)

    def __getitem__(self, i):
        return self.params[i]

    def opt_params(self):
        maximum = max(self.cost[1:])
        return maximum

    def opt_index(self):
        maximum = max(self.cost[1:])
        index = self.cost.index(maximum)
        return index


class result_iterator(object):
    def __init__(self, result):
        self.index = 0
        self.result = result

    def __next__(self):
        if self.index == self.result.length:
            raise StopIteration
        else:
            params = self.result.params[self.index]
            elbo = self.result.cost[self.index]
            self.index += 1
            return params, elbo




if __name__ == '__main__':

    x = sample()
    pprior_params = params(mu=0.0, k=1.0, a=0.001, b=0.001)
    pposterior_params = infer_pposterior(x, pprior_params)
    print(pposterior_params.mu, pposterior_params.k, pposterior_params.a, pposterior_params.b)
    pposterior = pposterior(pposterior_params).pdf(mu=0.0, lmbda=0.1)
    init = params(mu=0.0, k=1.0, a=2.0, b=1.0)
    qposterior_infer_results = infer_qposterior(x, pprior_params, init)
    #print(len(qposterior_infer_results))
    #print(qposterior_infer_results[1].k)
    #x = uninorm.sample(mu, sigma, size)
    #pprior_params = uninorm.Params(u, k, a, b)
    #pposterior_params = uninorm.infer_pposterior(x, pprior_params)
    #pposterior = uninorm.PPosterior(pposterior_params)
    #init = uninorm.Params(u, k, a, b)
    #qposterior_infer_results = uninorm.infer_qposterior(x, pprior_params, init, maxit)
    for item, cost in qposterior_infer_results:
        print(item.mu, item.k, item.a, item.b, cost)

    opt = qposterior_infer_results.opt_params()
    print(opt)
