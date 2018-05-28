import random
import numpy as np
#from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class SMO(object):

    def __init__(self, X, Y, C, iterations):

        self.X = X
        self.Y = Y
        self.C = C
        self.iter = iterations
        self.alpha = [0 for _ in range(len(Y))]
        self.b = 0
        self.H = 0
        self.L = 0

        it = 0
        while it < self.iter:
            for i in range(len(self.X)):

                x_i = X[i]
                j = self.select_xj(i)
                #j = self.heuristic_xj(i)
                if not j:
                    break
                print('index j:', j)

                x_j = X[j]

                f_x_i = self.f(x_i)
                f_x_j = self.f(x_j)

                y_i = Y[i]
                y_j = Y[j]

                E_i = f_x_i - y_i
                E_j = f_x_j - y_j

                eta = self.K(x_i, x_i) + self.K(x_j, x_j) - 2 * self.K(x_i, x_j)

                alpha_i = self.alpha[i]
                alpha_j = self.alpha[j]

                if y_i != y_j:
                    self.L = max(0, alpha_j - alpha_i)
                    self.H = min(self.C, self.C + alpha_j - alpha_i)

                if y_i == y_j:
                    self.L = max(0, alpha_i + alpha_j - self.C)
                    self.H = min(self.C, alpha_j + alpha_i)

                if eta > 0:
                    alpha_j_new = alpha_j + y_j * (E_i - E_j) / eta
                    alpha_j_new = self.clip(alpha_j_new)
                else:
                    fi = y_i*(E_i + self.b) - alpha_i*self.K(x_i,x_i) - y_i*y_j*alpha_j*self.K(x_i,x_j)
                    fj = y_j*(E_j + self.b) - y_i*y_j*alpha_i*self.K(x_i,x_j) - alpha_j*self.K(x_j,x_j)
                    L1 = alpha_i + y_i*y_j*(alpha_j - self.L)
                    H1 = alpha_i + y_i*y_j*(alpha_j - self.H)
                    lobj = L1 * f1 + self.L*f2 + 0.5*L1*L1*self.K(x_i,x_i) + 0.5*self.L*self.L*self.K(x_j,x_j) + y_i*y_j*self.L*L1*self.K(x_i,x_j)
                    hobj = H1 * f1 + self.H*f2 + 0.5*H1*H1*self.K(x_i,x_i) + 0.5*self.H*self.H*self.K(x_j,x_j) + y_i*y_j*self.H*H1*self.K(x_i,x_j)
                    if lobj < hobj - 1e-3:
                        alpha_j_new = self.L
                    elif lobj > hobj + 1e-3:
                        alpha_j_new = self.H
                    else:
                        alpha_j_new = alpha_j
                    print('eta is non positive')

                if abs(alpha_j - alpha_j_new) < 1e-3*(alpha_j + alpha_j_new + 1e-3):
                    continue

                alpha_i_new = alpha_i + y_i * y_j * (alpha_j - alpha_j_new)

                self.alpha[i], self.alpha[j] = alpha_i_new, alpha_j_new

                b_i_new = - E_i - (alpha_i_new - alpha_i) * y_i * self.K(x_i, x_i) - \
                          (alpha_j_new - alpha_j) * y_j * self.K(x_i, x_j) + self.b

                    #b_i_new = E_i + (alpha_i_new - alpha_i) * y_i * self.K(x_i, x_i) + \
                    #          (alpha_j_new - alpha_j) * y_j * self.K(x_i, x_j) + self.b
                    #print(b_i_new)

                b_j_new = - E_j - (alpha_i_new - alpha_i) * y_i * self.K(x_i, x_j) - \
                          (alpha_j_new - alpha_j) * y_j * self.K(x_j, x_j) + self.b

                    #b_j_new = E_j + (alpha_i_new - alpha_i) * y_i * self.K(x_i, x_j) + \
                    #         (alpha_j_new - alpha_j) * y_j * self.K(x_j, x_j) + self.b

                    #print(b_j_new)

                if alpha_i_new >= 0 and alpha_i_new <= self.C:
                    self.b = b_i_new
                elif alpha_j_new >= 0 and alpha_j_new <= self.C:
                    self.b = b_j_new
                else:
                    self.b = (b_i_new + b_j_new) / 2
            it += 1

    def select_xj(self, i):
        m = list(range(len(self.Y)))
        temp = m[:i] + m[i+1:]
        return random.choice(temp)

    def heuristic_xj(self, i):
        m = list(range(len(self.Y)))
        for index in m:
            if index == i:
                continue
            else:
                if self.alpha[index] < 0 + 1e-3 or self.alpha[index] > self.C - 1e-3:
                    print('alpha:', self.alpha[index])
                    return index
        return None


    def f(self, x):
        return np.dot(np.array(self.alpha) * np.array(self.Y), self.K(self.X,x)) + self.b

    def w(self):
        return np.dot(np.array(self.alpha) * np.array(self.Y), np.array(self.X))


    def clip(self, alpha):
        if alpha >= self.H:
            return self.H
        elif alpha <= self.L:
            return self.L
        else:
            return alpha

    def K(self, x_i, x_j):
        return np.dot(x_i, x_j)



if __name__ == '__main__':

    X,Y = make_classification(n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes = 2, n_clusters_per_class=1)
    #X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
    Y = [pow(-1,num+1) for num in Y]

    s = SMO(X,Y,1,100)
    w = s.w()
    b = s.b

    x1 = max(X, key = lambda x: x[0])
    x2 = min(X, key = lambda x: x[0])
    slope = - w[0] / w[1]
    intercept = - b / w[1]

    print(slope)
    print(intercept)
    print(s.alpha)

    y1 = slope * x1[0] + intercept
    y2 = slope * x2[0] + intercept

    plt.title("One informative feature, one cluster per class", fontsize='small')
    plt.scatter(X[:,0], X[:,1], marker='o', c=Y, s=25, edgecolor='k')
    plt.plot([x1[0], x2[0]], [y1, y2])

    for i, alpha in enumerate(s.alpha):
        if abs(alpha) > 1e-3:
            x, y = X[i][0], X[i][1]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')



    plt.show()
