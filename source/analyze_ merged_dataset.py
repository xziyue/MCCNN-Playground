import pickle
import numpy as np
from source.rel_path import rootDir
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open(os.path.join(rootDir, 'data', 'train_merged.pickle'), 'rb') as inFile:
    xs, ys, ids = pickle.load(inFile)


# find range of each channel
mins = np.asarray([np.min(np.take(xs, ind, axis=4)) for ind in range(xs.shape[4])])
maxs = np.asarray([np.max(np.take(xs, ind, axis=4)) for ind in range(xs.shape[4])])

print(mins)
print(maxs)

def get_standardized_xs(_xs):
    xs = _xs.copy()
    for ind in range(xs.shape[4]):
        theIndex = tuple([slice(None)] * 4 + [ind])
        channel = xs[theIndex]
        channel -= mins[ind]
        channel += 1.0
        np.log(channel, out=channel)
    return xs

newXs = get_standardized_xs(xs)

# find range of new xs
mins = np.asarray([np.min(np.take(newXs, ind, axis=4)) for ind in range(xs.shape[4])])
maxs = np.asarray([np.max(np.take(newXs, ind, axis=4)) for ind in range(xs.shape[4])])

print(mins)
print(maxs)


def plot_grid(ind, ranges):
    pass



