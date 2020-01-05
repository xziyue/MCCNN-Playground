import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from source.rel_path import rootDir
from scipy.stats import describe
from source.data_standardization import DataStandardizer
import pickle
from source.pdbqt_parser import *
from source.fetch_pdbqt_by_id import get_pdbqt_by_id

atoms = ['A', 'C', 'e', 'HD', 'OA']

def plot_affinity_3d(axList, sample, channelStats, **kwargs):
    #assert len(sample.shape) == 4

    drawThreshold = kwargs.get('drawThreshold', 0.0)
    if not isinstance(drawThreshold, np.ndarray):
        if isinstance(drawThreshold, list) or isinstance(drawThreshold, tuple):
            drawThreshold = np.asarray(drawThreshold, np.float)
        elif isinstance(drawThreshold, int) or isinstance(drawThreshold, float):
            drawThreshold = np.asarray([drawThreshold] * sample.shape[3], np.float)
    assert drawThreshold.size == sample.shape[3]

    cmapBase = kwargs.get('cmapBase', 'jet')
    cmap = matplotlib.cm.get_cmap(cmapBase)
    channelCmaps = []
    for i in range(sample.shape[3]):
        stat = channelStats[i]
        norm = matplotlib.colors.Normalize(vmin=stat.minmax[0], vmax=stat.minmax[1])
        colorFunc = lambda val : cmap(norm(val))
        channelCmaps.append(colorFunc)

    marker = kwargs.get('marker', 'o')
    markerSize = kwargs.get('markerSize', 10)

    xx, yy, zz = np.meshgrid(np.arange(sample.shape[0]), np.arange(sample.shape[1]), np.arange(sample.shape[2]), indexing='ij')

    for chInd in range(sample.shape[3]):
        ax = axList[chInd]
        ax.set_title(atoms[chInd])
        xyz = sample[:, :, :, chInd]
        drawRegion = xyz > drawThreshold[chInd]
        drawXyz = xyz[drawRegion]
        colors = channelCmaps[chInd](drawXyz.flat)
        ax.scatter(xx[drawRegion], yy[drawRegion], zs=zz[drawRegion], marker=marker, c=colors, s=markerSize)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

def plot_activation_3d(ax, sample, **kwargs):
    drawThreshold = kwargs.get('drawThreshold', 0.0)
    cmapBase = kwargs.get('cmapBase', 'jet')
    cmap = matplotlib.cm.get_cmap(cmapBase)
    marker = kwargs.get('marker', 'o')
    markerSize = kwargs.get('markerSize', 10)
    overrideColor = kwargs.get('overrideColor', None)

    edges = kwargs.get('edges', None)

    if edges is None:
        xx, yy, zz = np.meshgrid(np.arange(sample.shape[0]), np.arange(sample.shape[1]), np.arange(sample.shape[2]), indexing='ij')
    else:
        ticks = [np.linspace(edges[i, 0], edges[i, 1], sample.shape[i]) for i in range(3)]
        xx, yy, zz = np.meshgrid(*ticks, indexing='ij')

    drawRegion = sample > drawThreshold
    drawXyz = sample[drawRegion]
    if overrideColor is None:
        colors = cmap(drawXyz.flat)
    else:
        colors = overrideColor

    mappable = ax.scatter(xx[drawRegion], yy[drawRegion], zs=zz[drawRegion], marker=marker, c=colors, s=markerSize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return mappable



def plot_atom_3d(ax, item, gridShape, **kwargs):
    marker = kwargs.get('marker', 'o')
    markerSize = kwargs.get('markerSize', 10)

    # turn atom types into atom ids
    atomLoc = item.atomLoc
    atomInd = np.asarray([allAtoms.index(atom) for atom in item.elementType])
    assert np.all(atomInd != -1)

    edges = item.get_grid_edges(*gridShape)

    # draw atoms group by group
    for i in range(len(allAtoms)):
        typeInd = np.where(atomInd == i)[0]
        if typeInd.size == 0:
            continue
        # extract atom location
        typeLoc = atomLoc[typeInd, ...]

        # trim points outside bounding box
        xs = typeLoc[:, 0]
        ys = typeLoc[:, 1]
        zs = typeLoc[:, 2]

        goodXs = np.bitwise_and(xs >= edges[0, 0], xs <= edges[0, 1])
        goodYs = np.bitwise_and(ys >= edges[1, 0], ys <= edges[1, 1])
        goodZs = np.bitwise_and(zs >= edges[2, 0], zs <= edges[2, 1])
        goodMask = np.bitwise_and(np.bitwise_and(goodXs, goodYs), goodZs)

        xs = xs[goodMask]
        ys = ys[goodMask]
        zs = zs[goodMask]

        ax.scatter(xs, ys, zs=zs, c=f'C{i}', s=markerSize * atomRadii[allAtoms[i]], marker=marker, label=allAtoms[i])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlim(tuple(edges[0, :]))
    ax.set_ylim(tuple(edges[1, :]))
    ax.set_zlim(tuple(edges[2, :]))
    ax.legend()


if __name__ == '__main__':


    with open(os.path.join(rootDir, 'data', 'train_merged.pickle'), 'rb') as inFile:
        xs, ys, ids = pickle.load(inFile)

    standardizer = DataStandardizer()
    standardizer.fit(xs)
    newXs = standardizer.transform(xs)

    stats = [describe(np.take(newXs, ind, axis=4).flat) for ind in range(newXs.shape[4])]

    ind = 6

    fig = plt.figure()
    axList = [fig.add_subplot(1, 5, i + 1, projection='3d') for i in range(5)]
    plot_affinity_3d(axList, newXs[ind, ...], stats, drawThreshold=[0.6, 0.6, 0.6, 0.6, 0.6], markerSize=5)
    plt.show()

    pdbqt = get_pdbqt_by_id(ids[ind])

    ax = plt.gca(projection='3d')
    plot_atom_3d(ax, pdbqt, markerSize=300)
    plt.show()