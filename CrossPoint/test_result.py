import os
import torch
import h5py
import ast
import json
import numpy as np
import trimesh
import h5py
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from helper import *


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

class ShapeNetPart():
    def __init__(self, num_points, data, label, seg, partition='test', class_choice=None):
        self.data, self.label, self.seg = data, label, seg
        self.cat2id = {'tooth': 1}
        self.seg_num = [17]
        self.index_start = [0]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        
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







if __name__ == '__main__':

    num_channels = 3
    num_classes = 17
    num_points = 8000
    test_batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #test file path
    obj_files = '/content/gdrive/MyDrive/CrossPoint/data/YNKZHRP0_lower.obj'#YNKZHRP0
    label_vertices_files = '/content/gdrive/MyDrive/CrossPoint/data/YNKZHRP0_lower.json'
    
    #procuce cell label 
    label_cells = process_labels(obj_files, label_vertices_files) 
    # print(len(label_cells)) 

    #checkpoint path
    checkpoint = "model_900_1.t7"

    #produce test file
    points, label, seg = DataProcessing(obj_files, label_cells, num_points, num_classes, num_channels)
    # print(points.shape, label.shape, seg.shape)
    
    #prediction
    predict, labels = test(points, label, seg, num_points, test_batch_size, checkpoint, device)
    # print(predict.shape, labels.shape)
    print("Downsample miou accuracy:")
    miou, acc, sen, dsc, ppv = calculate_metrics(predict, labels, num_classes)
    print("mIoU:", miou)
    print("Accuracy:", acc)
    print("SEN:", sen)
    print("DSC:", dsc)
    print("PPV:", ppv)

    # visualization
    visualization_pointcloud(points, labels)

    #save
    with h5py.File('crosspoint_test.h5', 'w') as f:
      f.create_dataset('data', data=points)
      f.create_dataset('label', data=predict)

    









