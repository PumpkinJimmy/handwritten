import numpy as np
from read import get_data
def get_data_0_8():
    with open("data/train_0_8.npz", 'rb') as f:
        chunk = np.load(f)
        images0 = chunk['images0']
        images8 = chunk['images8']
    return images0, images8

def KNNClassify08(img, imgs0, imgs8, k):
    img = img.astype(int)
    lb0 = np.zeros(imgs0.shape[0])
    lb8 = np.empty(imgs8.shape[0])
    lb8.fill(8)
    lb = np.r_[lb0, lb8]
    data = np.r_[imgs0, imgs8]
    dst = ((data - img) ** 2).sum(axis=(1, 2))
    rk = dst.argsort()
    knn = lb[rk < k]
    if knn[knn == 0].shape[0] >= k / 2:
        return 0
    else:
        return 8

        


