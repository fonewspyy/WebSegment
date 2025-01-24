import numpy as np

def onehot_encoding(label, num_classes):
    shape = label.shape
    onehot = np.zeros((num_classes, shape[0], shape[1]))

    for i in range(num_classes):
        indices = np.where(label == i)
        onehot[i][indices] = 1.0

    return onehot