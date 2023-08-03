import os
import torch
import ast
import h5py
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
list_file_path = os.path.join(parent_dir, 'Data', 'lists_1.txt') # List file saved in dataloader.py


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

    #  15 features
    vertex1, vertex2, vertex3 = vertices[cells[sampled_indices, 0]], vertices[cells[sampled_indices, 1]], vertices[
        cells[sampled_indices, 2]]
    normal = mesh.face_normals[sampled_indices]
    center = np.mean(vertices[cells[sampled_indices]], axis=1)

    X_input = np.hstack((center, normal,
                         vertex1 - center,
                         vertex2 - center,
                         vertex3 - center))

    return X_input, sampled_indices 
    
    # # 24 features
    # vector_24d = np.zeros((num_samples, 24))
    # sampled_cells = cells[sampled_indices]
    # sampled_triangle_centers = triangle_centers[sampled_indices]
    # normals = mesh.vertex_normals
    #
    # for i in range(num_samples):
    #     # Coordinates of the vertex and center
    #     vector_24d[i, :12] = np.hstack((vertices[sampled_cells[i]].ravel(), sampled_triangle_centers[i]))
    #     # Normal vectors of the vertex and center
    #     vector_24d[i, 12:] = np.hstack((normals[sampled_cells[i]].ravel(), normals[sampled_cells[i]].mean(axis=0)))
    
    # return vector_24d, sampled_indices


def main(obj_files, label_list):
    # dataset file path
    h5_filename = '/Users/31475/PycharmProjects/Project_3D/Data/crosspoint_100_test.h5'
    num_points = 10000
    # num_points = 8000
    num_features = 15
    # num_features = 24
    num_classes = 17

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

            points = torch.tensor(points, dtype=torch.float32).view(1, num_points, num_features)
            label = [labels[i] for i in indices]
            label_tensor = torch.tensor(label, dtype=torch.int64)
            labels = F.one_hot(label_tensor, num_classes)
            # print(points.shape)
            # print(labels.shape)

            # Check if datasets already exist before creating them.
            if f'data{idx}' not in h5f:
                h5f.create_dataset(f'data{idx}', data=points)
            if f'label{idx}' not in h5f:
                h5f.create_dataset(f'label{idx}', data=labels)

def dataset(partition="train"):
    with open(list_file_path, "r") as file:
        content = file.read()
        str_lists = content.split("\n\n")
        lists = [ast.literal_eval(str_list) for str_list in str_lists]
        # 8k
        # b1 train 0-150,170-200,210-300 test 150-170 validation 200-210
        # b2 train 0-150,180-300         test 150-170 validation 170-180
        # 1k
        # b1 train 30-300  test 0-20 validation 20-30
        # b2 train 30-300  test 0-20 validation 20-30
        data_list = lists[4][200:] + lists[5][200:]
        data_label = lists[6][200:] + lists[7][200:]

    if partition == 'train':
        obj_files = data_list
        label_list = data_label

    main(obj_files, label_list)


if __name__ == '__main__':
    dataset(partition='train')

