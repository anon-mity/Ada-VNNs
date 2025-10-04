import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from scipy.spatial.transform import Rotation as sciR
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def R_from_euler_np(angles):
    '''
    angles: [(b, )3]
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(angles[0]), -math.sin(angles[0]) ],
                    [0,         math.sin(angles[0]), math.cos(angles[0])  ]
                    ])
    R_y = np.array([[math.cos(angles[1]),    0,      math.sin(angles[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(angles[1]),   0,      math.cos(angles[1])  ]
                    ])

    R_z = np.array([[math.cos(angles[2]),    -math.sin(angles[2]),    0],
                    [math.sin(angles[2]),    math.cos(angles[2]),     0],
                    [0,                     0,                      1]
                    ])
    return np.dot(R_z, np.dot( R_y, R_x ))

def rotate_point_cloud(data, R = None, max_degree = None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        R:
          3x3 array, optional Rotation matrix used to rotate the input
        max_degree:
          float, optional maximum DEGREE to randomly generate rotation
        Return:
          Nx3 array, rotated point clouds
    """
    # rotated_data = np.zeros(data.shape, dtype=np.float32)
    if R is not None:
      rotation_angle = R
    elif max_degree is not None:
      rotation_angle = np.random.randint(0, max_degree, 3) * np.pi / 180.0
    else:
      rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or  rotation_angle.ndim == 1:
      rotation_matrix = R_from_euler_np(rotation_angle)
    else:
      assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
      rotation_matrix = rotation_angle[:3, :3]

    if data is None:
      return None, rotation_matrix
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data, rotation_matrix   # return [N, 3],




# 归一化到一个以原点为中心，半径为1的单位球体范围内 [-1,1]
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ShapeNetCoreH5Loader_pccomplet(Dataset):
    def __init__(self, data_path, mode='train', category='', num_pts=1024):
        """
        Args:
            h5_file_path (str): Path to the h5 file containing the dataset.
            num_pts (int): Number of points to sample from each point cloud.
        """
        self.file_path = os.path.join(data_path, f'{category}_{mode}.h5')

        # Load file
        with h5py.File(self.file_path, 'r') as f:
            self.part_data = f['partials'][:]
            self.comp_data = f['completes'][:]



    def __len__(self):
        lenght = len(self.part_data)
        return lenght


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        part_pc = self.part_data[idx]
        comp_pc = self.comp_data[idx]

        # R augment
        pc_so3, R_so3 = rotate_point_cloud(part_pc)  # so3
        gt_pc = comp_pc @ R_so3

        #vis(pc_so3,gt_pc)
        return {
            'pc_so3':torch.from_numpy(pc_so3.astype(np.float32)),
            'label': torch.from_numpy(gt_pc.astype(np.float32))
        }


def vis(point_cloud_1_np,point_cloud_2_np):
    # i,output_dir
    fig = plt.figure(figsize=(10, 5))
    # 绘制第一个子图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(point_cloud_1_np[:, 0], point_cloud_1_np[:, 1], point_cloud_1_np[:, 2], c='r', marker='o')
    ax1.set_title('aug')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    # 绘制第二个子图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(point_cloud_2_np[:, 0], point_cloud_2_np[:, 1], point_cloud_2_np[:, 2], c='g', marker='o')
    ax2.set_title('norm')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])

    # 保存图形到文件夹中
    #output_file = os.path.join(output_dir, f'pc{i}.png')
    #plt.savefig(output_file)
    plt.show()
    plt.close()


