import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_keys(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        return list(file.keys())

# 加载点云数据和标签数据 # Load point cloud data and label data
h5_filename = '/Users/31475/PycharmProjects/Project_3D/Data/new_data_b1_0_202.h5'
# index = 10  # 根据需要选择要可视化的数据的索引 # Select the index of the data to be visualized as needed
hdf5_file = h5py.File(h5_filename, "r")
# keys = get_keys(h5_filename)
# slide_data = hdf5_file[keys[0]]
points = hdf5_file['data'][()]
cell_labels = hdf5_file['label'][()]
# with h5py.File(h5_filename, 'r') as h5f:
#     points = h5f[f'points_{index}'][()]
#     cell_labels = h5f[f'label_{index}'][()]
print(points.shape,cell_labels.shape)
# 将点云数据转换为适合可视化的格式 # Convert point cloud data into a format suitable for visualization
points = points.reshape(-1, 24)[:, 9:12]  # 提取前三个坐标作为点的位置 # Extract the first three coordinates as the position of the point

# 准备可视化 # Prepare to visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 为每个 label 分配固定颜色
unique_labels = np.unique(cell_labels)
label_colors = cm.get_cmap('tab20')

for i, label in enumerate(unique_labels):
    label_points = points[cell_labels == label]
    color = label_colors(label / 16)[:-1]  # 只提取 RGB 颜色值，而不是 RGBA # Only extract RGB color values, not RGBA
    ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2], c=np.array(color).reshape(1, -1), label=f'label_{label}', alpha=0.5)

ax.legend()
plt.show()


#
#
import json
from UniformSampling import calculate_mesh_centers, read_stl_and_sample_points, reassign_labels, rearrange, MeshPlotter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh
import matplotlib.cm

List = 'C:\\Users\\31475\\PycharmProjects\\Project_3D\\Data\\3D_scans_per_patient_obj_files_b2\\IF1GKK1E\\IF1GKK1E_lower.obj'
label = 'C:\\Users\\31475\\PycharmProjects\\Project_3D\\Data\\ground-truth_labels_instances_b2\\IF1GKK1E\\IF1GKK1E_lower_cell.json'




with open(label, 'r') as f:
    json_data = json.load(f)
    labels = json_data  # ['labels']

# print(labels)
# 加载 OBJ 文件
mesh = trimesh.load_mesh(List)

# 加载 cell label 列表
cell_labels = np.array(labels)

# 为每个 label 分配固定颜色
# 使用 matplotlib 的 "tab20" colormap，它包含 20 种不同的颜色
unique_labels = np.unique(cell_labels)
label_colors = matplotlib.cm.get_cmap('tab20')

face_colors = np.zeros((len(mesh.faces), 3))
for i, label in enumerate(unique_labels):
    face_colors[cell_labels == label] = label_colors(label / 16)[:-1]  # 只提取 RGB 颜色值，而不是 RGBA

# face_colors = np.ones((len(mesh.faces), 3))
mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)

# 可视化
mesh.show()
#
