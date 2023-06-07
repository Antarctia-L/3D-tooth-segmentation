import os
import torch
import ast
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from UniformSampling import calculate_mesh_centers, reassign_labels, MeshPlotter
import h5py
import meshio

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
list_file_path = os.path.join(parent_dir, 'Data', 'lists_1.txt')


def sample_mesh_cells_distance(filepath, filetype, num_samples=10000, n_neighbors=5):

    mesh = trimesh.load(filepath, file_type=filetype)
    vertices = mesh.vertices
    cells = mesh.faces

    triangle_centers = np.mean(vertices[cells], axis=1)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(triangle_centers)
    distances, _ = nbrs.kneighbors(triangle_centers)

    weights = np.mean(distances, axis=1)
    weights /= np.sum(weights)

    sampled_indices = np.random.choice(cells.shape[0], num_samples, replace=False, p=weights)

    vertex1, vertex2, vertex3 = vertices[cells[sampled_indices, 0]], vertices[cells[sampled_indices, 1]], vertices[
        cells[sampled_indices, 2]]
    normal = mesh.face_normals[sampled_indices]
    center = np.mean(vertices[cells[sampled_indices]], axis=1)

    X_input = np.hstack((center, normal,
                         vertex1 - center,
                         vertex2 - center,
                         vertex3 - center))

    return X_input, sampled_indices  # Changed to return two separate values.


def main(obj_files, label_list):
    h5_filename = '/Users/31475/PycharmProjects/Project_3D/Data/data_200_b1_lower.h5'

    if not os.path.exists(h5_filename):
        with h5py.File(h5_filename, 'w') as _:
            pass

    with h5py.File(h5_filename, 'a') as h5f:
        for idx in range(len(obj_files)):
            print(idx)
            obj_file = obj_files[idx]
            points, indices = sample_mesh_cells_distance(obj_file, "obj", num_samples=10000, n_neighbors=5)

            label_path = label_list[idx]
            f = open(label_path, 'r')
            labels = ast.literal_eval(f.read())
            f.close()

            label = [labels[i] for i in indices]

            num_points = 10000
            num_features = 15
            points = torch.tensor(points, dtype=torch.float32).view(1, num_features, num_points)
            print(points.shape)

            # Check if dataset already exists before creating it.
            if f'points_{idx}' not in h5f:
                h5f.create_dataset(f'points_{idx}', data=points)
            if f'label_{idx}' not in h5f:
                h5f.create_dataset(f'label_{idx}', data=label)


def dataset(partition="train"):
    with open(list_file_path, "r") as file:
        content = file.read()
        str_lists = content.split("\n\n")
        lists = [ast.literal_eval(str_list) for str_list in str_lists]

        data_list = lists[0][:200]
        data_label = lists[2][:200]

    if partition == 'train':
        obj_files = data_list
        label_list = data_label

    main(obj_files, label_list)


if __name__ == '__main__':
    dataset(partition='train')

