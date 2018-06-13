import matplotlib.pyplot as plt
import matplotlib.cm as cm
from uninorm import *


def visualize(posterior, axes, xlim, ylim, color):

    locs = np.linspace(xlim[0], xlim[1])
    scales = np.linspace(ylim[0], ylim[1])
    pp = posterior.pdf(locs, scales)

    axes.contour(locs, scales, pp, colors=color)
    axes.set_xlabel('$\mu$')
    axes.set_ylabel('$\lambda$')

def plot_infer_result(pposterior, qposterior_infer_results, xlim, ylim, maxiter):
    opt_index = qposterior_infer_results.opt_index()

    nrows = 3
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows*5))
    axes = axes.flatten()
    for i in range(maxiter):
        visualize(pposterior, axes[i], xlim, ylim, color='red')
        Qposterior = qposterior(qposterior_infer_results.params[i])
        #print(qposterior_infer_results[i])
        #print(qposterior(qposterior_infer_results[i]))
        visualize(Qposterior, axes[i], xlim, ylim, color='green')
        axes[i].set_title('Iteration {:d}{:s}'.format(i, '*' if i == opt_index else ''))
    plt.show()

if __name__ == '__main__':

    x = sample()
    pprior_params = params(mu=0.0, k=0.1, a=0.02, b=0.03)
    pposterior_params = infer_pposterior(x, pprior_params)
    pposterior = pposterior(pposterior_params)#.pdf(mu=0.0, lmbda=0.1)
    init = params(mu=0.0, k=0.1, a=0.02, b=0.03)
    qposterior_infer_results = infer_qposterior(x, pprior_params, init)
    plot_infer_result(pposterior, qposterior_infer_results, xlim=[-0.4,0.4], ylim=[0.2,1.5], maxiter=9)
