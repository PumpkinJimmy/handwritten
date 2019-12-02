import numpy as np


def normalize(X):
    X = X.astype(float) - np.c_[X.mean(axis=1)]
    X /= np.c_[X.std(axis=1)]
    return X
def transformLabels(labels, k):
    '''
    @param label m
    @return m*k
    '''
    res = np.zeros((labels.shape[0], k))
    for i, lb in enumerate(labels):
        res[i][lb] = 1
    return res


def softmax(Z):
    '''
    res: k*m, k is number of classes, m is number of samples
    '''
    res = np.exp(-Z)
    res /= res.sum(axis=0).reshape((1, res.shape[1]))
    return res

def h(X, W):
    return softmax(X.dot(W).T)
def cost(err, y):
    m = err.shape[0]
    res = 0
    for i in range(m):
        if err[i, y[i]] / np.sum(err[i, :]) > 0:
            res -= np.log(err[i, y[i]] / np.sum(err[i, :]))
        else:
            res -= 0
    return res / m


#def cost(X, w, y):
#    p = h(X, w)
#    res = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0]))
#    return (-1 / X.shape[0]) * res


def getGrad(X, w, y):
    '''
    X: m*n
    w: n*k
    y: m*k
    '''
    return (h(X, w) - y.T).dot(X).T


def gradDecent(X, y, alpha=0.01, iterCnt=100):
    '''
    X: m*n
    y: m*k
    '''
    w = np.ones((X.shape[1] + 1, y.shape[1])) # n * k
    X = np.c_[X, np.ones((X.shape[0], ))]
    for i in range(iterCnt):
        print(f"Iter {i + 1}")
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        grad = getGrad(X, w, y)
        w = w - alpha * grad
    return w
def gradDecend(X, y, k, alpha=0.01, iterCnt=100):
    '''
    X: m*n
    y: m*k
    '''
    X = np.c_[X, np.ones((X.shape[0], ))]
    m, n = X.shape
    w = np.ones((n, k)) # (n + 1) * k
    for i in range(iterCnt):
        if i % 10 == 0:
            res = h(X, w).argmax(axis=0)
            cnt = np.sum(res != y)
            print(f"Train Error Rate: {cnt / m}")
        print(f"Iter {i + 1}")
        err = np.exp(X.dot(w))
        print(f"Cost: {cost(err, y)}")
        rowsum = -err.sum(axis=1)
        err /= rowsum.reshape(m, 1)
        for x in range(m):
            err[x, y[x]] += 1
        w = w + (alpha / m) * X.T.dot(err)
    return w


