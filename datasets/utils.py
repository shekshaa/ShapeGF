from plyfile import PlyData, PlyElement
import numpy as np

def read_ply(file):
    loaded = PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'], loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'], loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)
    
    points = points.transpose(1, 0)
    x, y, z = points[:, 1].reshape(-1, 1),-points[:, 0].reshape(-1, 1),\
        points[:, 2].reshape(-1, 1)

    return np.concatenate([x, y, z], axis=1)