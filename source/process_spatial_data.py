import os
import subprocess as sp
from source.rel_path import rootDir
import re
from shutil import copy2, move

def _recursive_search_target(path, result):
    assert os.path.exists(path)
    allFiles = os.listdir(path)

    for filename in allFiles:
        fullname = os.path.join(path, filename)

        if os.path.isdir(fullname):
            _recursive_search_target(fullname, result)
        elif os.path.isfile(fullname):
            _, ext = os.path.splitext(fullname)
            if ext == '.pdbqt':
                result.append(fullname)

def recursive_search_target(path):
    ret = []
    _recursive_search_target(path, ret)
    return ret

def get_autogrid4_template(id):
    rawTemplate = \
r'''npts 15 15 15                        # num.grid points in xyz
gridfld $$$$.maps.fld         # grid_data_file
spacing 1.0                          # spacing(A)
receptor_types A C HD N NA OA SA    # receptor atom types
ligand_types A C HD N NA OA SA       # ligand atom types
receptor $$$$.pdbqt           # macromolecule
gridcenter auto       # xyz-coordinates or auto
smooth 0.5                           # store minimum energy w/in rad(A)
map $$$$.A.map                # atom-specific affinity map
map $$$$.C.map                # atom-specific affinity map
map $$$$.HD.map               # atom-specific affinity map
map $$$$.N.map                # atom-specific affinity map
map $$$$.NA.map               # atom-specific affinity map
map $$$$.OA.map               # atom-specific affinity map
map $$$$.SA.map               # atom-specific affinity map
elecmap $$$$.e.map            # electrostatic potential map
dsolvmap $$$$.d.map              # desolvation potential map
dielectric -0.1465                   # <0, AD4 distance-dep.diel;>0, constant
'''
    return re.sub(r'\${4}', id, rawTemplate)


def run_autogrid4(targetFilename, outputPath):
    # get target id
    head1, _ = os.path.split(targetFilename)
    _, head2 = os.path.split(head1)

    id = head2

    # get template
    pgfTemplate = get_autogrid4_template(id)

    autogridPath = os.path.join(rootDir, 'autogrid')
    autogridExecutable = os.path.join(autogridPath, 'autogrid4')

    tempPgfFilename = os.path.join(autogridPath, id + '.pgf')
    with open(tempPgfFilename, 'w') as outFile:
        outFile.write(pgfTemplate)

    # copy data
    tempTargetFilename = os.path.join(autogridPath, id + '.pdbqt')
    copy2(targetFilename, tempTargetFilename)

    # run autogrid4
    sp.run([autogridExecutable, '-p', tempPgfFilename], check=True, cwd=autogridPath)

    # separate files
    relatedFiles = []
    uselessFiles = []

    for name in os.listdir(autogridPath):
        if name.startswith(id):
            if name.endswith('.map'):
                relatedFiles.append(name)
            else:
                uselessFiles.append(name)

    # move all files to target path
    for name in relatedFiles:
        move(os.path.join(autogridPath, name), outputPath)

    # delete useless files
    for name in uselessFiles:
        os.remove(os.path.join(autogridPath, name))


# extract all features from the dataset
targets = recursive_search_target(os.path.join(rootDir, 'data', 'spatial_features'))
for target in targets:
    run_autogrid4(target, os.path.join(rootDir, 'data', 'grid_data'))
