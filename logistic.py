import numpy as np


def get_data08(filename):
    with open(filename, 'rb') as f:
        chunk = np.load(f)
        img0 = chunk['images0']
        img8 = chunk['images8']
    return img0, img8


def transform_data08(img0, img8):
    i0cnt = img0.shape[0]
    i8cnt = img8.shape[0]
    data = np.r_[img0.reshape((i0cnt, 28 * 28)),
                 img8.reshape((i8cnt, 28 * 28))]
    labels = np.zeros((i0cnt + i8cnt, ))
    labels[i0cnt:] = 1
    return data, labels


def normalize(X):
    X = X.astype(float) - np.c_[X.mean(axis=1)]
    X /= np.c_[X.std(axis=1)]
    return X


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, w):
    return sigmoid(X.dot(w))


def cost(X, w, y):
    p = h(X, w)
    res = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0]))
    return (-1 / X.shape[0]) * res


def getGrad(X, w, y):
    return (h(X, w) - y).dot(X)


def logisticClassify(X, w):
    p = np.c_[X, np.ones(X.shape[0])].dot(w)
    p[p > 0.5] = 1
    p[p <= 0.5] = 0
    return p


def _logisticClassify(X, w):
    p = X.dot(w)
    p[p > 0.5] = 1
    p[p <= 0.5] = 0
    return p


def gradDecent(X, y, alpha=1, iterCnt=100):
    w = np.random.random(X.shape[1] + 1)
    X = np.c_[X, np.ones((X.shape[0], ))]
    for i in range(iterCnt):
        print(f"Iter {i + 1}")
        print(f"Cost: {cost(X, w, y)}")
        grad = getGrad(X, w, y)
        w = w - alpha * grad
    return w


def gradDecentPlot(X, y, alpha=0.001, iterCnt=100):
    w = np.random.random(X.shape[1] + 1)
    X = np.c_[X, np.ones((X.shape[0], ))]
    err = np.empty((iterCnt, ))
    for i in range(iterCnt):
        print(f"Iter {i + 1}")
        err[i] = np.sum(_logisticClassify(X, w) != y) / X.shape[0]
        print(f"Cost: {err[i]}")
        print("Mean w", abs(w).mean())
        grad = getGrad(X, w, y)
        print("Mean Gradient", abs(grad).mean())
        w = w - alpha * grad
        if alpha > 0.0001:
            alpha *= 0.98
    return w, err
