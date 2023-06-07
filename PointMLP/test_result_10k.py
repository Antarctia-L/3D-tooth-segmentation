import os
import torch
import h5py
import ast
import json
import numpy as np
import trimesh
import h5py
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import torch.nn.functional as F
import torch.nn as nn
from Mesh_dataset2 import *
from pointMLP_10k import pointMLP_seg
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm
import plotly.graph_objects as go
from scipy import stats


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)



def sample_mesh_cells_distance(filepath, filetype, num_samples=10000, n_neighbors=5):
    mesh = trimesh.load(filepath, file_type=filetype)

    #rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    #mesh.apply_transform(rotation_matrix)
    # mesh.apply_transform(trimesh.transformations.translation_matrix())
    #mesh.apply_translation([0, 0, -95])
    #mesh.export('/content/gdrive/MyDrive/tooth_mesh_seg-master/new_mesh_t.obj', file_type='obj')
    #print(1)

    # Define rotation matrix
    angle = np.pi / 2  # Rotate 90 degrees
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])  # Rotate around z-axis
    
    # Apply rotation
    mesh.apply_transform(rotation_matrix)

    vertices = mesh.vertices
    cells = mesh.faces
    triangle_centers = np.mean(vertices[cells], axis=1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(triangle_centers)
    distances, _ = nbrs.kneighbors(triangle_centers)

    weights = np.mean(distances, axis=1)
    weights /= np.sum(weights)

    sampled_indices = np.random.choice(cells.shape[0], num_samples, replace=False, p=weights)
    #print(len(sampled_indices))
    # vector_24d = np.zeros((num_samples, 24))
    # sampled_cells = cells[sampled_indices]
    # sampled_triangle_centers = triangle_centers[sampled_indices]
    # normals = mesh.vertex_normals

    # for i in range(num_samples):
    #     vector_24d[i, :12] = np.hstack((vertices[sampled_cells[i]].ravel(), sampled_triangle_centers[i]))
    #     vector_24d[i, 12:] = np.hstack((normals[sampled_cells[i]].ravel(), normals[sampled_cells[i]].mean(axis=0)))
    vertex1, vertex2, vertex3 = vertices[cells[sampled_indices, 0]], vertices[cells[sampled_indices, 1]], vertices[
        cells[sampled_indices, 2]]
    normal = mesh.face_normals[sampled_indices]
    center = np.mean(vertices[cells[sampled_indices]], axis=1)

    X_input = np.hstack((center, normal,
                         vertex1 - center,
                         vertex2 - center,
                         vertex3 - center))
    # return vector_24d, sampled_indices
    #print(X_input.shape)
    return X_input, sampled_indices  # Changed to return two separate values.


def DataProcessing(test_file, label_cells, num_points, num_classes):
    
    points, indices = sample_mesh_cells_distance(test_file, "obj", num_samples=num_points, n_neighbors=5)

    points = torch.tensor(points, dtype=torch.float32).view(num_points, num_channels)
    
    label_cells = [label_cells[i] for i in indices]
    labels = np.zeros((len(label_cells), num_classes))

    # 将每个标签转换为对应的one-hot编码
    for i, label_idx in enumerate(label_cells):
        labels[i, label_idx] = 1
   

    return points, labels


def PointCloudLabelList(data_filename, label_filename):
    mesh = trimesh.load_mesh(data_filename)
    vertices = mesh.vertices
    cells = mesh.faces

    with open(label_filename, 'r') as f:
        json_data = json.load(f)
        vertex_labels = json_data['labels']
        # print(len(vertex_labels))

    cell_labels = []
    for cell in cells:

        vertex_labels_of_cell = [vertex_labels[cell[0]], vertex_labels[cell[1]], vertex_labels[cell[2]]]

        if len(set(vertex_labels_of_cell)) == 3:
            # Compute the center of the triangle
            center = (vertices[cell[0]] + vertices[cell[1]] + vertices[cell[2]]) / 3
            # Calculate the distances to the center
            distances = [np.linalg.norm(vertices[cell[i]] - center) for i in range(3)]
            # Get the index of the closest vertex
            closest_vertex_index = np.argmin(distances)
            # Add the label of the closest vertex to the cell_labels list
            cell_labels.append(vertex_labels_of_cell[closest_vertex_index])
        else:
            # Find the most frequent label and add it to the cell_labels list
            cell_labels.append(stats.mode(vertex_labels_of_cell)[0][0])

    return cell_labels


def rearrange(nparry):
    # 32 permanent teeth
    nparry[nparry == 17] = 1
    nparry[nparry == 37] = 1
    nparry[nparry == 16] = 2
    nparry[nparry == 36] = 2
    nparry[nparry == 15] = 3
    nparry[nparry == 35] = 3
    nparry[nparry == 14] = 4
    nparry[nparry == 34] = 4
    nparry[nparry == 13] = 5
    nparry[nparry == 33] = 5
    nparry[nparry == 12] = 6
    nparry[nparry == 32] = 6
    nparry[nparry == 11] = 7
    nparry[nparry == 31] = 7
    nparry[nparry == 21] = 8
    nparry[nparry == 41] = 8
    nparry[nparry == 22] = 9
    nparry[nparry == 42] = 9
    nparry[nparry == 23] = 10
    nparry[nparry == 43] = 10
    nparry[nparry == 24] = 11
    nparry[nparry == 44] = 11
    nparry[nparry == 25] = 12
    nparry[nparry == 45] = 12
    nparry[nparry == 26] = 13
    nparry[nparry == 46] = 13
    nparry[nparry == 27] = 14
    nparry[nparry == 47] = 14
    nparry[nparry == 18] = 15
    nparry[nparry == 38] = 15
    nparry[nparry == 28] = 16
    nparry[nparry == 48] = 16
    # deciduous teeth
    nparry[nparry == 55] = 3
    nparry[nparry == 55] = 3
    nparry[nparry == 54] = 4
    nparry[nparry == 74] = 4
    nparry[nparry == 53] = 5
    nparry[nparry == 73] = 5
    nparry[nparry == 52] = 6
    nparry[nparry == 72] = 6
    nparry[nparry == 51] = 7
    nparry[nparry == 71] = 7
    nparry[nparry == 61] = 8
    nparry[nparry == 81] = 8
    nparry[nparry == 62] = 9
    nparry[nparry == 82] = 9
    nparry[nparry == 63] = 10
    nparry[nparry == 83] = 10
    nparry[nparry == 64] = 11
    nparry[nparry == 84] = 11
    nparry[nparry == 65] = 12
    nparry[nparry == 85] = 12

    return nparry

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def process_labels(data_filename, label_filename):
    
    # 重新计算标签列表
    label_cell = PointCloudLabelList(data_filename, label_filename)
    label_cell = np.array(label_cell)
    # print(type(La))
    label_cell_rerrange = rearrange(label_cell)
    label_cell_rerrange = label_cell_rerrange.tolist()
    #print(len(label_cell_rerrange))


    return label_cell_rerrange # new_label_path



def test_model(points, labels, checkpoint_path, checkpoint_name, device):
    """
    A function to test a pre-trained model.

    Parameters:
    - input_file: the path to the input file
    - checkpoint_path: the path to the saved model checkpoint
    - checkpoint_name: the name of the checkpoint file
    - device: the device (CPU or GPU) to use for model inference

    Returns:
    - predicted_output: the output of the model
    """

    # Load the saved model checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_name))
    model = pointMLP_seg(num_classes=num_classes, num_channels=num_channels)
    model = nn.DataParallel(model)
    model = model.to(device, dtype=torch.float)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    
    # Load the input data
    # test_dataset = Mesh_Dataset(data_list_path=input_file,
    test_dataset = Mesh_Dataset(points = points, labels = labels,
                                num_classes=num_classes,
                                patch_size=10000)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=2,
                             shuffle=False,
                             num_workers=2)

    with torch.no_grad():  # Do not calculate gradients to save memory

        for i_batch, batched_sample in enumerate(test_loader):
            
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)
            #print(inputs.shape, labels.shape)

            outputs = model(inputs)
            #print(outputs.shape)
            op = outputs.reshape(-1, num_classes)
            op = torch.argmax(op, dim=1)
            op = op[:num_points]
            label = torch.argmax(labels, dim=1)
            lbl = label.view(-1)
            lbl = lbl[:num_points]
            # print(op)
            # print(lbl)
            

        # print("predict label size:", op.shape)
        # print("origion label size:", lbl.shape)
        # _, predicted_output = torch.max(outputs.data, 1)  # Use the class with the highest score as the prediction

    return op, lbl


def get_keys(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        return list(file.keys())


def upsample(predict, points, data_filename, label_cells):

    # points = slide_data['data'][()]
    cell_labels = predict #slide_data['label'][()]
    # cell_labels = torch.argmax(cell_labels, dim=1)
    cell_labels = torch.nn.functional.one_hot(cell_labels, 17)
    # print(torch.argmax(torch.from_numpy(slide_data['label'][()]), dim=1))
    
    
    # points = points.reshape(-1, 24)[:, 9:12]  # 提取前三个坐标作为点的位置 # Extract the first three coordinates as the position of the point
    points = points.reshape(-1, 15)[:, :3]  # 提取前三个坐标作为点的位置 # Extract the first three coordinates as the position of the point


    mesh = trimesh.load_mesh(data_filename)
    angle = np.pi / 2  # Rotate 90 degrees
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])  # Rotate around z-axis
    
    # Apply rotation
    mesh.apply_transform(rotation_matrix)
    centroids = mesh.triangles_center
    #print(centroids.shape)

    # 3. Use KNN to classify each centroid
    for k in [1,3]:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(points, cell_labels)

        # predict labels for centroids
        predicted_labels = knn.predict(centroids)
        predicted_labels = torch.from_numpy(predicted_labels)
        predicted_labels = torch.argmax(predicted_labels, dim=1)
    
    #print("upsample label size:", predicted_labels.shape)

    l = label_cells
    #print("gold label size:", l.shape)

    return predicted_labels, l


def visualization(data_filename, labels):

    mesh = trimesh.load_mesh(data_filename)
    cell_labels = np.array(labels)
    unique_labels = np.unique(cell_labels)
    label_colors = matplotlib.cm.get_cmap('tab20')

    face_colors = np.zeros((len(mesh.faces), 3))
    for i, label in enumerate(unique_labels):
        face_colors[cell_labels == label] = label_colors(label / 16)[:-1]  # 只提取 RGB 颜色值，而不是 RGBA

    # face_colors = np.ones((len(mesh.faces), 3))
    mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)

    # 可视化
    mesh.show()


def calculate_metrics(tensor1, tensor2, num_classes):
    intersection = torch.zeros(num_classes, dtype=torch.float)
    union = torch.zeros(num_classes, dtype=torch.float)
    true_positives = torch.zeros(num_classes, dtype=torch.float)
    false_positives = torch.zeros(num_classes, dtype=torch.float)
    false_negatives = torch.zeros(num_classes, dtype=torch.float)

    for class_idx in range(num_classes):
        class_mask1 = (tensor1 == class_idx)
        class_mask2 = (tensor2 == class_idx)

        intersection[class_idx] = (class_mask1 & class_mask2).sum().float()
        union[class_idx] = (class_mask1 | class_mask2).sum().float()
        true_positives[class_idx] = intersection[class_idx]
        false_positives[class_idx] = (class_mask1 & ~class_mask2).sum().float()
        false_negatives[class_idx] = (~class_mask1 & class_mask2).sum().float()

    miou = intersection / union
    valid_classes = torch.isnan(miou) == False
    miou = miou[valid_classes]
    
    acc = (true_positives.sum() / len(tensor1)).item()
    sen = (true_positives[valid_classes] / (true_positives[valid_classes] + false_negatives[valid_classes] + 1e-8)).mean().item()
    dsc = (2 * true_positives[valid_classes] / (2 * true_positives[valid_classes] + false_positives[valid_classes] + false_negatives[valid_classes] + 1e-8)).mean().item()
    ppv = (true_positives[valid_classes] / (true_positives[valid_classes] + false_positives[valid_classes] + 1e-8)).mean().item()

    return miou.mean().item(), acc, sen, dsc, ppv




if __name__ == '__main__':

    num_channels = 15
    num_classes = 17
    num_points = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #test file path
    obj_files = '/content/gdrive/MyDrive/tooth_mesh_seg-master/main/01M24NSZ_upper.obj'
    label_vertices_files = '/content/gdrive/MyDrive/tooth_mesh_seg-master/main/01M24NSZ_upper.json'
    
    #procuce cell label 
    label_cells = process_labels(obj_files, label_vertices_files) 
    #print(len(label_cells)) 

    #checkpoint path
    checkpoint_path = os.path.join(parent_dir, 'tooth_mesh_seg-master', 'models')
    checkpoint_name = 'Mesh_Segementation_520samples_2.0_best.tar'

    #produce test file
    points, labels = DataProcessing(obj_files, label_cells, num_points, num_classes)
    
    #prediction
    predict, label = test_model(points, labels, checkpoint_path, checkpoint_name, device)
    print("Downsample miou accuracy:")
    miou, acc, sen, dsc, ppv = calculate_metrics(predict, label, num_classes)
    print("mIoU:", miou)
    print("Accuracy:", acc)
    print("SEN:", sen)
    print("DSC:", dsc)
    print("PPV:", ppv)


    #upsample
    upsample_label, original_label = upsample(predict.cpu(), points, obj_files, label_cells)
    upsample_label = torch.tensor(upsample_label)
    original_label = torch.tensor(original_label)
    print("Upsample miou accuracy:")
    miou, acc, sen, dsc, ppv = calculate_metrics(upsample_label, original_label, num_classes)
    print("mIoU:", miou)
    print("Accuracy:", acc)
    print("SEN:", sen)
    print("DSC:", dsc)
    print("PPV:", ppv)

    #visualization
    #visualization(obj_files, upsample_label)

    #save
    # mesh = trimesh.load_mesh(obj_files)
    # centroids = mesh.triangles_center
    # print(centroids.shape, upsample_label.shape)
    # with h5py.File('new_data_C06U.h5', 'w') as f:
    # # 在文件中创建两个数据集，分别为 'data' 和 'label'
    #   f.create_dataset('data', data=centroids)
    #   f.create_dataset('label', data=upsample_label)

    









