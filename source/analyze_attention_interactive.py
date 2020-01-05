# configure matplotlib
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython.display import HTML, display, Markdown
from tabulate import tabulate

import pickle
import numpy as np
from source.rel_path import rootDir

from source.compute_attention import *
from source.data_visualization import *

y_test_pred = np.argmax(model.predict(x_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

correctPredInd = np.where(y_test_pred == y_test_labels)[0]

testSample0s = np.where(y_test_labels == 0)[0]
testSample1s = np.where(y_test_labels == 1)[0]


# pick samples that are correctly classified
visSample0s = np.intersect1d(correctPredInd, testSample0s)
visSample1s = np.intersect1d(correctPredInd, testSample1s)

print(visSample1s)

for i in range(3):
    testInd = visSample1s[i]
    fig = plt.figure(figsize=(9, 4))
    axs = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

    pdbqt = get_pdbqt_by_id(ids_test[testInd])
    edges = pdbqt.get_grid_edges(15, 15, 15)
    # show the atom
    plot_atom_3d(axs[0], pdbqt, (15, 15, 15), markerSize=200)
    # show the attention
    attention = visualize_cam1(model, -1, 1, x_test[testInd, ...])
    plot_activation_3d(axs[1], attention, drawThreshold=0.5, markerSize=140, cmap='jet')

    plot_atom_3d(axs[2], pdbqt, (15, 15, 15), markerSize=200)
    plot_activation_3d(axs[2], attention, drawThreshold=0.5, markerSize=140, cmap='jet', overrideColor='black', edges=edges)

    axs[0].set_title('Molecule Overview')
    axs[1].set_title('Class Activation')
    axs[2].set_title('Overlay')

    fig.suptitle(ids_test[testInd])
    plt.show()