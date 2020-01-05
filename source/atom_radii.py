import pandas
import os
from source.rel_path import rootDir

_atomInfoDf = pandas.read_csv(os.path.join(rootDir, 'data', 'atom_radii.csv'))

# construct atom radii dict
atomRadii = dict()
for i in range(_atomInfoDf.shape[0]):
    eleName = _atomInfoDf.iloc[i, 1].strip()
    radii = _atomInfoDf.iloc[i, 4].strip()
    try:
        radii = float(radii) / 100.0
        atomRadii[eleName] = radii
    except Exception as e:
        continue