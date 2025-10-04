import numpy as np
import os
from torch.utils.data import Dataset
import torch
import json
import math
from scipy.spatial.transform import Rotation as sciR
import matplotlib.pyplot as plt

# 可视化点云函数
def visualize_point_cloud(pc_norm, pc_so3):
    fig = plt.figure(figsize=(12, 6))

    # 原始归一化点云
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pc_norm[:, 0], pc_norm[:, 1], pc_norm[:, 2], c=pc_norm[:, 2], cmap='viridis', s=10)
    ax1.set_title('Normalized Point Cloud (pc_norm)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    # SO3旋转后的点云
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pc_so3[:, 0], pc_so3[:, 1], pc_so3[:, 2], c=pc_so3[:, 2], cmap='viridis', s=10)
    ax2.set_title('SO3 Rotated Point Cloud (pc_so3)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])

    plt.tight_layout()
    plt.show()


# 可视化旋转矩阵
def visualize_rotation_matrix(R):
    plt.figure(figsize=(10, 4))

    plt.subplot(131)
    plt.imshow(R, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Rotation Matrix')
    plt.colorbar()

    plt.subplot(132)
    plt.bar(range(3), np.degrees(np.arccos(np.diag(R))), color='skyblue')
    plt.title('Diagonal Elements (rotation in degrees)')
    plt.ylim(0, 180)

    plt.subplot(133)
    angles = np.array([np.arctan2(-R[1, 2], R[2, 2]),
                       np.arctan2(R[0, 2], np.sqrt(R[1, 2] ** 2 + R[2, 2] ** 2)),
                       np.arctan2(-R[0, 1], R[0, 0])])
    plt.bar(range(3), np.degrees(angles), color='salmon')
    plt.title('Euler Angles (degrees)')

    plt.tight_layout()
    plt.show()


# 归一化到一个以原点为中心，半径为1的单位球体范围内 [-1,1]
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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


class PartNormalDataset(Dataset):
    def __init__(self, root='/home/hanbing/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='trainval', class_choice='Airplane', npoints=1024, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]

        # Randomly sample points
        if self.npoints is not None and len(point_set) > self.npoints:
            sampled_indices = np.random.choice(len(point_set), self.npoints, replace=False)
            point_set = point_set[sampled_indices]

        # Normalize
        point_cloud = pc_normalize(point_set[:, 0:3])

        # R augment
        pc = point_cloud
        # pc_z, R_z = rotate_point_cloud_z(point_cloud)  # z
        pc_so3, R_so3 = rotate_point_cloud(point_cloud)  # so3

        return {
            'pc_norm': torch.from_numpy(pc.astype(np.float32)),
            # 'pc_z': torch.from_numpy(pc_z.astype(np.float32)),
            'pc_so3': torch.from_numpy(pc_so3.astype(np.float32)),
            # 'target_z': torch.from_numpy(R_z.astype(np.float32)),
            'target_so3': torch.from_numpy(R_so3.astype(np.float32)),
        }


    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    data = PartNormalDataset(npoints=1024, split='trainval', class_choice='Cap', normal_channel=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)


