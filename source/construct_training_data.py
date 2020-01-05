import numpy as np
import os
from source.rel_path import rootDir
import pandas
import pickle

gridX = gridY = gridZ = 15

def parse_map(mapFilename):
    with open(mapFilename, 'r') as infile:
        lines = infile.readlines()
    numbers = list(map(float, lines[6:]))
    arr = np.asarray(numbers).reshape((gridX, gridY, gridZ))
    return arr

gridDataPath = os.path.join(rootDir, 'data', 'grid_data')
atoms = ['A', 'C', 'e', 'HD', 'OA']

def read_training_item(id):
    ret = dict()
    ret['id'] = id
    for atom in atoms:
        name = '.'.join([id, atom, 'map'])
        filename = os.path.join(gridDataPath, name)
        assert os.path.exists(filename)
        arr = parse_map(filename)
        ret[atom] = arr
    return ret


def constructed_merged_dataset():
    mergedDf = pandas.read_csv(os.path.join(rootDir, 'data', 'sequential_features_merge.csv'))
    allItems = []
    for ind in range(mergedDf.shape[0]):
        name = mergedDf.iloc[ind, -2]
        label = mergedDf.iloc[ind, -1]
        try:
            item = read_training_item(name)
            item['y'] = label
            allItems.append(item)
        except AssertionError as e:
            print(f'item {name} not found (discarded)')
            continue

    print('number of items:', len(allItems))

    # make numpy-friendly representations
    xs, ys, ids = [], [], []
    for item in allItems:
        arrs = [item[key] for key in atoms]
        x = np.stack(arrs, axis=3)
        y = item['y']
        xs.append(x)
        ys.append(y)
        ids.append(item['id'])

    xs = np.stack(xs, axis=0)
    ys = np.asarray(ys)

    with open(os.path.join(rootDir, 'data', 'train_merged.pickle'), 'wb') as outFile:
        pickle.dump((xs, ys, ids), outFile)

if __name__ == '__main__':
    constructed_merged_dataset()