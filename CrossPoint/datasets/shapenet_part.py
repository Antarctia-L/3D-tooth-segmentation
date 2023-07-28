import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
from torch.utils.data import Dataset

def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))

def load_data_partseg(partition):
    # download_shapenetpart()
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'train':
        file = glob.glob(os.path.join(DATA_DIR, 'crosspoint_data', 'crosspoint_900_train.h5')) # \
              #  + glob.glob(os.path.join(DATA_DIR, 'crosspoint_data', 'crosspoint_10_val_b1.h5'))
        # file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
        #        + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        # file = glob.glob(os.path.join(DATA_DIR, 'crosspoint_data', '*%s*.h5'%partition))
        # file = glob.glob(os.path.join(DATA_DIR, 'crosspoint_data', 'crosspoint_20_test_b1.h5'))
        file = glob.glob(os.path.join(DATA_DIR, 'crosspoint_data', 'crosspoint_50_val.h5'))

    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        keys = f.keys()
        num_datasets = sum(1 for key in keys if key.startswith('data'))
        for i in range(num_datasets):
            data = f[f'data{i}'][:].astype('float32')
            # print(data.shape)
            label = f[f'label{i}'][:].astype('int64')
            # print(label.shape)
            seg = f[f'pid{i}'][:].astype('int64')
            # print(seg.shape) 
            all_data.append(data)
            all_label.append(label)
            all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    # print("check", type(all_data))
    # print(all_data)
    # print(all_label)
    # print(all_seg)
    return all_data, all_label, all_seg

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        print(self.data.shape, self.label.shape, self.seg.shape)
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
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
            
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]