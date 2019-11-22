import struct
import numpy as np


def mnist_get_images(fname):
    with open(fname, 'rb') as f:
        data = f.read()
        magic = struct.unpack('>i', data[:4])
    if magic[0] != 2051:
        raise Exception("Wrong magic number")
    cnt = struct.unpack('>i', data[4:8])[0]
    data = data[16:]
    images = np.empty((cnt, 28, 28))
    offset = 0
    for i in range(cnt):
        image = np.zeros((28, 28))
        for row in range(28):
            for col in range(28):
                image[row][col] = data[offset + row * 28 + col]
        images[i] = image
        offset += 28 * 28
    return images


def mnist_get_labels(fname):
    with open(fname, 'rb') as f:
        data = f.read()
        magic = struct.unpack(">i", data[:4])
    if magic[0] != 2049:
        raise Exception("Wrong magic number")
    cnt = struct.unpack(">i", data[4:8])[0]
    data = data[8:]
    labels = np.zeros((cnt, ))
    for i in range(cnt):
        labels[i] = data[i]
    return labels


def get_data(fname):
    with open(fname, 'rb') as f:
        chunk = np.load(f)
        images = chunk['images']
        labels = chunk['labels']
    return images, labels
