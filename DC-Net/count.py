import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 加载点云数据和标签数据 # Load point cloud data and label data
h5_filename = '/Users/31475/PycharmProjects/Project_3D/Data/data_no.h5'
index = 10  # 根据需要选择要可视化的数据的索引 # Select the index of the data to be visualized as needed

with h5py.File(h5_filename, 'r') as h5f:
    points = h5f[f'points_{index}'][()]
    cell_labels = h5f[f'label_{index}'][()]

unique_classes = np.unique(cell_labels)
class_counts = np.bincount(cell_labels)

for cls, count in zip(unique_classes, class_counts):
    print(f"Class {cls}: {count} instances")