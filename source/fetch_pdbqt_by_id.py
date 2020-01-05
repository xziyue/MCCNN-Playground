from source.pdbqt_parser import *

def get_pdbqt_by_id(id):
    dataDir = os.path.join(rootDir, 'data', 'spatial_features')
    nextDir = os.path.join(dataDir, id)
    assert os.path.exists(nextDir)
    fileList = list(filter(lambda x : x.endswith('.pdbqt'), os.listdir(nextDir)))
    assert len(fileList) == 1
    fullPath = os.path.join(nextDir, fileList[0])
    return parse_pdbqt(fullPath)