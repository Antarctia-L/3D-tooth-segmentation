import os
import torch
import meshio
import numpy as np
from torch.utils.data import Dataset
from UniformSampling import sample_mesh_cells_distance, PointCloudVector, generate_label_to_id

import json
import meshio
from torch.utils.data import Dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
list_file_path = os.path.join(parent_dir, 'Data', 'lists.txt')

class CustomPointCloudDataset(Dataset):
    def __init__(self, obj_files, label_list, transform=None):
        self.obj_files = obj_files
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.obj_files)

    def pointcloud_sampling(self, mesh):
        # 使用您的 sample_mesh_cells_distance 函数从 mesh 中采样三角形索引
        num_samples = 400
        sampled_indices = sample_mesh_cells_distance(mesh, num_samples)

        # 使用您的 PointCloudVector 函数计算每个采样点的 15 个维度
        sampled_points = []
        for idx in sampled_indices:
            point_vector = PointCloudVector(mesh, idx)
            sampled_points.append(point_vector)

        # 将点云向量列表转换为一个形状为 (10000, 15) 的 NumPy 数组
        sampled_points = np.array(sampled_points)
        assert sampled_points.shape == (num_samples, 15), f"Unexpected shape: {sampled_points.shape}"

        return sampled_points, sampled_indices

    def __getitem__(self, idx):
        obj_file = self.obj_files[idx]
        mesh = meshio.read(obj_file)
        points, indices = self.pointcloud_sampling(mesh)

        label_path = self.label_list[idx]
        with open(label_path, 'r') as f:
            labels = json.load(f)

        label = [labels[i] for i in indices]


        # 假设 num_points 是点云中点的数量，num_features 是每个点的特征数
        num_points = 400  # 请根据实际情况修改
        num_features = 15  # 请根据实际情况修改

        # 将 points 转换为形状为 (1, num_features, num_points) 的张量
        points = torch.tensor(points, dtype=torch.float32).view(1, num_features, num_points)

        sample = {
            'points': points,
            'label': torch.tensor(label, dtype=torch.long)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample




def dataset(partition="train"):
    global obj_files, label_list
    with open(list_file_path, "r") as file:
        content = file.read()
        str_lists = content.split("\n\n")
        lists = [eval(str_list) for str_list in str_lists]
        lower_list = lists[0]
        upper_list = lists[1]
        lower_label = lists[2]
        upper_label = lists[3]

    if partition == 'train':
        obj_files = lower_list[:5]
        label_list = lower_label[:5]
    elif partition == 'test':
        obj_files = upper_list[:5]
        label_list = upper_label[:5]

    Dataset = CustomPointCloudDataset(obj_files, label_list)
    return Dataset






# obj_files = ['path/to/your/obj/file1.obj', 'path/to/your/obj/file2.obj', ...]
# labels = [...]
# from torch.utils.data import DataLoader
#
# batch_size = 32
# num_workers = 4
# dataset = CustomPointCloudDataset(obj_files, labels, pointcloud_sampling)
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
