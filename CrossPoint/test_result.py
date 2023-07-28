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
import plotly.graph_objects as go
from scipy import stats
# from __future__ import print_function
import argparse
import sklearn.metrics as metrics
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt
from matplotlib import cm
from datasets.shapenet_part import ShapeNetPart
from models.dgcnn import DGCNN_partseg_test
from torch.utils.data import DataLoader
from util import cal_loss, IOStream

from helper import *


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)



def sample_mesh_cells_distance(filepath, filetype, num_samples=10000, n_neighbors=5):

    mesh = trimesh.load(filepath, file_type=filetype)

    # Define rotation matrix
    # angle = np.pi / 2  # Rotate 90 degrees
    # rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])  # Rotate around z-axis
    
    # Apply rotation
    # mesh.apply_transform(rotation_matrix)

    vertices = mesh.vertices
    cells = mesh.faces

    triangle_centers = np.mean(vertices[cells], axis=1)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(triangle_centers)
    distances, _ = nbrs.kneighbors(triangle_centers)

    weights = np.mean(distances, axis=1)
    weights /= np.sum(weights)

    np.random.seed(42)  # set random seed as 42
    sampled_indices = np.random.choice(cells.shape[0], num_samples, replace=False, p=weights)
    # print(sampled_indices)
    center = np.mean(vertices[cells[sampled_indices]], axis=1)
    center_min = np.min(center)
    center_max = np.max(center)
    center_normalized = (center - center_min) / (center_max - center_min) * 2 - 1

    X_input = np.hstack(center_normalized)
    # X_input = np.reshape(X_input, (8000, 3))
    # print('X_input', X_input.shape)

    return X_input, sampled_indices  # Changed to return two separate values.


def DataProcessing(test_file, label_cells, num_points, num_classes):
    
    points, indices = sample_mesh_cells_distance(test_file, "obj", num_samples=num_points, n_neighbors=5)
    
    points = torch.tensor(points, dtype=torch.float32).view(num_points, num_channels)
    #print('points', points.shape)
    
    label_cells = [label_cells[i] for i in indices]
    # labels = np.zeros((len(label_cells), num_classes))

    # Convert each tag to the corresponding one-hot encoding
    # for i, label_idx in enumerate(label_cells):
    #     labels[i, label_idx] = 1
    seg_tensor = torch.tensor(label_cells, dtype=torch.int64)
    seg = seg_tensor.unsqueeze(0)
    points = points.unsqueeze(0)
    label = torch.tensor([[1]], dtype=torch.int64)
   

    return points, label, seg



class ShapeNetPart():
    def __init__(self, num_points, data, label, seg, partition='test', class_choice=None):
        self.data, self.label, self.seg = data, label, seg
        # self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
        #                'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
        #                'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        # self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        # self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.cat2id = {'tooth': 1}
        self.seg_num = [17]
        self.index_start = [0]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        # self.partseg_colors = load_color_partseg()
        
        self.seg_num_all = 50
        self.seg_start_index = 0
            
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]

def test(data, label, seg, num_points, test_batch_size, checkpoint, device):
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=num_points, class_choice=None, data=data, label=label, seg=seg),
                             batch_size=test_batch_size, shuffle=True, drop_last=False)
    # Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    model = DGCNN_partseg_test(seg_num_all, pretrain=False).to(device)
    net = torch.load(checkpoint, map_location=device)
    model.load_state_dict(net, strict=False)
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in test_loader:
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        # print("data", data.shape)
        # print("seg", seg.shape)

        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data, label_one_hot)

        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
        # print(test_true_cls.shape, test_true_seg.shape)

        # visiualization
        # visualization(args.visu, args.visu_format, data, pred, seg, label, partseg_colors, args.class_choice)
    test_true_cls = np.concatenate(test_true_cls)
    #print(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    #print(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    #print(test_acc, avg_per_class_acc)
    test_true_cls = torch.tensor(test_true_cls)
    test_pred_cls = torch.tensor(test_pred_cls)

    return test_pred_cls, test_true_cls

def get_keys(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        return list(file.keys())


def upsample(predict, points, data_filename, label_cells):

    # points = slide_data['data'][()]
    
    cell_labels = predict #slide_data['label'][()]
    # print(cell_labels.shape)
    # cell_labels = torch.argmax(cell_labels, dim=1)
    cell_labels = torch.nn.functional.one_hot(cell_labels, 17)
    # print(torch.argmax(torch.from_numpy(slide_data['label'][()]), dim=1))
    # print(cell_labels.shape)
    points = points.squeeze(0)
    # print(points.shape)


    mesh = trimesh.load_mesh(data_filename)
    # angle = np.pi / 2  # Rotate 90 degrees
    # rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])  # Rotate around z-axis
    # mesh.apply_transform(rotation_matrix)

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
        face_colors[cell_labels == label] = label_colors(label / 16)[:-1]

    # face_colors = np.ones((len(mesh.faces), 3))
    mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)

    # 可视化
    mesh.show()

def visualization_pointcloud(points, cell_labels):
    
    points = points.squeeze(0)
    print(points.shape,cell_labels.shape)

    # Prepare to visualize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 为每个 label 分配固定颜色
    unique_labels = np.unique(cell_labels)
    label_colors = cm.get_cmap('tab20')

    for i, label in enumerate(unique_labels):
        label_points = points[cell_labels == label]
        color = label_colors(label / 16)[:-1]  # Only extract RGB color values, not RGBA
        ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2], c=np.array(color).reshape(1, -1), label=f'label_{label}', alpha=0.5)

    ax.legend()
    plt.show()
    #fig.savefig('visualization_pointcloud_151.png')


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

    num_channels = 3
    num_classes = 17
    num_points = 8000
    test_batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #test file path
    obj_files = '/content/gdrive/MyDrive/CrossPoint/data/01M24NSZ_lower.obj'
    label_vertices_files = '/content/gdrive/MyDrive/CrossPoint/data/01M24NSZ_lower.json'
    
    #procuce cell label 
    label_cells = process_labels(obj_files, label_vertices_files) 
    print(len(label_cells)) 

    #checkpoint path
    checkpoint = "model_200_300ep_acc81_iou81_8000.t7"

    #produce test file
    points, label, seg = DataProcessing(obj_files, label_cells, num_points, num_classes)
    print(points.shape, label.shape, seg.shape)
    
    #prediction
    predict, labels = test(points, label, seg, num_points, test_batch_size, checkpoint, device)
    print(predict.shape, labels.shape)
    

    # print("origional label:", label)
    # print("predict label:", predict)
    print("Downsample miou accuracy:")
    miou, acc, sen, dsc, ppv = calculate_metrics(predict, labels, num_classes)
    print("mIoU:", miou)
    print("Accuracy:", acc)
    print("SEN:", sen)
    print("DSC:", dsc)
    print("PPV:", ppv)

    visualization_pointcloud(points, labels)


    #upsample
    # upsample_label, original_label = upsample(predict.cpu(), points, obj_files, label_cells)
    # upsample_label = torch.tensor(upsample_label)
    # original_label = torch.tensor(original_label)
    # print("Upsample miou accuracy:")
    # miou, acc, sen, dsc, ppv = calculate_metrics(upsample_label, original_label, num_classes)
    # print("mIoU:", miou)
    # print("Accuracy:", acc)
    # print("SEN:", sen)
    # print("DSC:", dsc)
    # print("PPV:", ppv)

    # visualization
    # mesh = trimesh.load_mesh(data)
    # visualization(mesh, upsample_label)

    #save
    # mesh = trimesh.load_mesh(obj_files)
    # centroids = mesh.triangles_center

    # print(centroids.shape, upsample_label.shape)
    # print(centroids.shape, labels.shape)
    with h5py.File('crosspoint_151_v.h5', 'w') as f:
      f.create_dataset('data', data=points)
      f.create_dataset('label', data=predict)

    









