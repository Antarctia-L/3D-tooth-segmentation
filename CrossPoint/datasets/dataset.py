import os
import torch
import ast
import h5py
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
list_file_path = os.path.join(parent_dir, 'Data', 'lists_1.txt')


def sample_mesh_cells_distance(filepath, filetype, num_points, n_neighbors=5):

    mesh = trimesh.load(filepath, file_type=filetype)
    vertices = mesh.vertices
    cells = mesh.faces
    triangle_centers = np.mean(vertices[cells], axis=1)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(triangle_centers)
    distances, _ = nbrs.kneighbors(triangle_centers)

    weights = np.mean(distances, axis=1)
    weights /= np.sum(weights)

    np.random.seed(42)  # random seed
    sampled_indices = np.random.choice(cells.shape[0], num_points, replace=False, p=weights)
    center = np.mean(vertices[cells[sampled_indices]], axis=1)
    center_min = np.min(center)
    center_max = np.max(center)
    center_normalized = (center - center_min) / (center_max - center_min) * 2 - 1

    X_input = np.hstack(center_normalized)
    return X_input, sampled_indices  


def main(obj_files, label_list):
    h5_filename = '/Users/31475/PycharmProjects/Project_3D/Data/crosspoint_100_test.h5'
    num_points = 8000
    num_features = 3
    
    if not os.path.exists(h5_filename):
        with h5py.File(h5_filename, 'w') as _:
            pass

    with h5py.File(h5_filename, 'a') as h5f:
        for idx in range(len(obj_files)):
            print(idx)
            obj_file = obj_files[idx]
            points, indices = sample_mesh_cells_distance(obj_file, "obj", num_points)
            label_path = label_list[idx]
            f = open(label_path, 'r')
            labels = ast.literal_eval(f.read())
            f.close()

            labels = torch.ones((1, 1), dtype=torch.int64)
            points = torch.tensor(points, dtype=torch.float32).view(1, num_points, num_features)
            pid = [labels[i] for i in indices]
            pid = torch.tensor(pid, dtype=torch.int64).view(1, num_points)
            # print(points.shape)
            # print(labels.shape)
            # print(pid.shape)

            # Check if datasets already exist before creating them.
            if f'data{idx}' not in h5f:
                h5f.create_dataset(f'data{idx}', data=points)
            if f'pid{idx}' not in h5f:
                h5f.create_dataset(f'pid{idx}', data=pid)
            if f'label{idx}' not in h5f:
                h5f.create_dataset(f'label{idx}', data=labels)


def dataset():
    with open(list_file_path, "r") as file:
        content = file.read()
        str_lists = content.split("\n\n")
        lists = [ast.literal_eval(str_list) for str_list in str_lists]
        # 200samples crosspoint b1 train 0-100 test 100-120 val 120-130
        # 900samples crosspoint b1 train 0-250 val 250:300 test 250:300 b2 train 0-200 teat 200:300 
        data_list = lists[0][:250] + lists[1][:250] + lists[4][:200] + lists[5][:200]
        label_list = lists[2][:250] + lists[3][:250] + lists[6][:200] + lists[7][:200]
        
    main(data_list, label_list)


if __name__ == '__main__':
    dataset()

