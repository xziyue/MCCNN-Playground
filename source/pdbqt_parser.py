import os
from source.rel_path import rootDir
from source.process_spatial_data import recursive_search_target
import numpy as np
from source.atom_radii import atomRadii


_atom_translator = {
    'C' : 'C',
    'H' : 'H',
    'N' : 'N',
    'O' : 'O',
    'S' : 'S',
    'OA': 'O',
    'NA' : 'N',
    'SA' : 'S',
    'HD' : 'H',
    'A' : 'C'
}

allAtoms = list(set(_atom_translator.keys()))
allAtoms.sort()

class PDBQTItem:

    def __init__(self, allFields):
        self.nAtoms = len(allFields)
        # gather atom types
        self.atomType = [allFields[i][-1] for i in range(self.nAtoms)]

        self.elementType = [_atom_translator[atom] for atom in self.atomType]

        # gather atom coordinates
        atomLoc = [allFields[i][8:11] for i in range(self.nAtoms)]
        self.atomLoc = np.asarray(atomLoc)

        self.atomCenter = np.mean(self.atomLoc, axis=0)

    def get_grid_edges(self, nx, ny, nz):
        halfx = (nx - 1) / 2
        halfy = (ny - 1) / 2
        halfz = (nz - 1) / 2

        edges = [
            self.atomCenter[0] - halfx, self.atomCenter[0] + halfx,
            self.atomCenter[1] - halfy, self.atomCenter[1] + halfy,
            self.atomCenter[2] - halfz, self.atomCenter[2] + halfz
        ]

        return np.asarray(edges).reshape((3, 2))


_pdbqt_ticks = [(1,6), (7,11), (13,16), (17, 17), (18, 21), (22, 22), (23, 26), (27, 27), (31,38), (39, 46), (47, 54),
                (55, 60), (61, 66), (67, 76), (77, 80)]

_identity = lambda x : x
_fieldFuncs = [_identity, int, _identity, _identity, _identity, _identity, int, _identity, float, float, float, float, float, float, _identity]

def parse_pdbqt(filename):

    with open(filename, 'r') as infile:
        lines = infile.readlines()

    allFields = []
    for line in lines:
        line = line.strip()
        if line.startswith('TER'):
            continue

        fields = []
        for l, r in _pdbqt_ticks:
            field = line[l - 1 : r]
            fields.append(field.strip())

        assert len(fields) == 15
        parsedFields = [_fieldFuncs[i](fields[i]) for i in range(len(fields))]
        allFields.append(parsedFields)

    return PDBQTItem(allFields)

if __name__ == '__main__':
    target = os.path.join(rootDir, 'data', 'test', '3D', '1kzy', '1kzy.pdbqt')
    result = parse_pdbqt(target)
    print(result.atomCenter)
    print(result.get_grid_edges(15, 15, 15))


