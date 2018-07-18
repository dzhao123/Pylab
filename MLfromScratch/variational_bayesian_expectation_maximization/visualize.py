import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy as sp
import scipy.stats as stats


def angle(v):
    a = np.arccos(v[0] / np.linalg.norm(v)) / np.pi * 180
    if v[1] >= 0:
        return a
    else:
        return 360 - a


# https://onlinecourses.science.psu.edu/stat505/node/36
def ellipse_normal(u, S, alpha=0.05):
    l, e = np.linalg.eig(S)
    df = len(l)
    widths = [np.sqrt(l[i] * stats.chi2.ppf(1 - alpha, df=df)) for i in range(df)]
    return matplotlib.patches.Ellipse(u, widths[0], widths[1], angle(e[:, 0]))


def plot_normal(u, S, alpha):
    fig, ax = plt.subplots()
    e = ellipse_normal(u, S, alpha)
    ax.add_artist(e)
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_xlim(u[0] - 2, u[0] + 2)
    ax.set_ylim(u[1] - 2, u[1] + 2)


def plot_q(ax, params, X, xlim=None, ylim=None, s=10):
    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)
    scatter = ax.scatter(X[:, 0], X[:, 1], color='blue', s=s)
    for i in range(len(params.w)):
        S = np.linalg.inv(params.v[i] * params.w[i])
        en = ellipse_normal(params.m[i], S)
        e = ax.add_artist(en)
        e.set_edgecolor('red')
        e.set_facecolor('red')
        e.set_alpha(max(0.1, params.a[i]))
        e.set_linewidth(3)
    scatter.set_zorder(len(params.w) + 1)


def plot_q_it(params_seq, X, xlim=None, ylim=None, n=None, s=20, ncols=3):
    if n is None:
        n = len(params_seq)
    nrows = int(np.ceil(float(n) / ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    ax = ax.flatten()
    for i in range(n):
        plot_q(ax[i], params_seq[i], X, xlim=xlim, ylim=ylim, s=s)
        ax[i].set_title('Iteration {:d}'.format(i))
    plt.show()