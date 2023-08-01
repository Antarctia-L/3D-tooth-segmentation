import os
import ast
import h5py
import torch
import random
import trimesh
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

"""
    Processing:
    1. Choose data files with only 4 crop teeth labeled (Randomly selected in the sample code)
    2. Produce the point cloud of crop teeth
    3. Alignment
    4. Down-sample to fixed number
    5. Save as file
    6. Visualization
    
"""

# get data files path from list
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
list_file_path = os.path.join(parent_dir, 'Data', 'lists_1.txt')

# four classes of teeth
list_1 = [1, 2, 13, 14]  # molar teeth
list_2 = [3, 4, 11, 12]  # canine teeth
list_3 = [5, 6, 9, 10]  # incisor teeth
list_4 = [7, 8, 7, 8]  # front tooth
# jaw = [0]


# Select labeled files
with open(list_file_path, "r") as file:
    content = file.read()
    str_lists = content.split("\n\n")
    lists = [ast.literal_eval(str_list) for str_list in str_lists]
    data_path = lists[0][120:170]  # + lists[0][120:130]
    label_path = lists[2][120:170]  # + lists[2][120:130]

# icp
def icp(source_points, target_points, max_iterations=100, tolerance=1e-6):
    """
    The ICP algorithm is used to align the source_points point cloud so
    that it coincides with the target_points point cloud as much as possible.

    Parameters：
    source_points：numpy array，shape(N, 3)
    target_points：numpy array，shape(N, 3)
    max_iterations：default = 100。
    tolerance：Tolerance for convergence judgment. The default is 1e-6.

    Return：
    aligned_points：numpy array，shape(N, 3)，aligned source point cloud.
    """

    aligned_points = source_points.copy()
    prev_error = 0

    for iteration in range(max_iterations):
        # Step 1: Find the nearest neighbors between the points
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
        distances, indices = nbrs.kneighbors(aligned_points)

        # Step 2: Compute the transformation (rotation and translation)
        delta = target_points[indices[:, 0]] - aligned_points
        mean_error = np.mean(np.linalg.norm(delta, axis=1))

        if np.abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

        # Step 3: Apply the transformation to the source points
        aligned_points += delta

    return aligned_points


all_label = []
all_data = []
# Convert files to point clouds and labels（8000，3）（8000，）
for idx in range(len(label_path)):
    # read data from mesh files
    label_file = label_path[idx]
    f = open(label_file, 'r')
    labels = ast.literal_eval(f.read())
    f.close()
    unique_classes = np.unique(labels)
    np.random.seed(42)
    lists = [list_1, list_2, list_3, list_4]
    # Get crop teeth randomly
    elements = [list(set(lst) & set(unique_classes)) for lst in lists]
    if [] in elements:
        continue
    list_selected = [random.choice(elem) for elem in elements]
    # Get indices of crop teeth mesh
    selected_indices = []
    for selected_label in list_selected:
        indices = [index for index, label in enumerate(labels) if label == selected_label]
        selected_indices.extend(random.choices(indices, k = 2000))

    # Convert mesh into pointcloud (num_points,3)
    obj_file = data_path[idx]
    mesh = trimesh.load_mesh(obj_file)
    vertices = mesh.vertices
    cells = mesh.faces
    center = np.mean(vertices[cells[selected_indices]], axis=1)
    center_min = np.min(center)
    center_max = np.max(center)
    center_normalized = (center - center_min) / (center_max - center_min) * 2 - 1
    points = center_normalized
    # Get labels (num_points,)
    cell_labels = np.array(labels)
    cell_labels = cell_labels[selected_indices]

    all_data.append(center_normalized)
    all_label.append(cell_labels)
all_data_seg = np.concatenate(all_data, axis=0)
all_label_seg = np.concatenate(all_label, axis=0)
# Got the data of labeled crop teeth, 4 teeth each jaw
print(all_data_seg.shape, all_label_seg.shape)


unique_classes = np.unique(all_label_seg)
# print(unique_classes)

all_block = []
all_block_label = []
all_block_seg = []
all_block_label_seg = []
all_block_sample_seg = []
for class_label in unique_classes:
    # Get indices of each label class
    class_indices = np.where(np.array(all_label_seg) == class_label)[0]
    merged_points = all_data_seg[class_indices]
    points = np.array(merged_points)
    mean_center = np.mean(merged_points, axis=0)

    block_size = 2000
    blocks = [merged_points[i:i + block_size] for i in range(0, len(merged_points), block_size)]
    all_block_sample = []
    # Use ICP to align the current block with the first block
    for i, block in enumerate(blocks):
        if i == 0:
            all_block.append(block)  # The first block remains unchanged
        else:
            aligned_block = icp(block, blocks[0])
            all_block.append(aligned_block)
        block_label = np.full(block_size, class_label)
        all_block_label.append(block_label)
    all_block_seg = np.concatenate(all_block, axis=0)
    all_block_label_seg = np.concatenate(all_block_label, axis=0)
print("results", all_block_seg.shape, all_block_label_seg.shape)


# Random downsample
data_tensor = torch.from_numpy(all_block_seg)  # Enter data tensor (num_points,3)
label_tensor = torch.from_numpy(all_block_label_seg)   # Enter label tensor (num_points,)

unique_labels = torch.unique(label_tensor)  # Gets a unique label classes
selected_data = []
selected_labels = []
num_sample = 8000 # Num of pointcloud of each mean shape tooth

for label in unique_labels:
    indices = torch.nonzero(label_tensor == label).squeeze()  # Gets a unique tag value
    if len(indices) > num_sample:
        indices = indices[torch.randperm(len(indices))[:num_sample]]
    selected_data.append(data_tensor[indices])
    selected_labels.append(label_tensor[indices])

selected_data = torch.cat(selected_data, dim=0)
selected_labels = torch.cat(selected_labels, dim=0)

print(selected_data.shape)  # (num_points,3)
print(selected_labels.shape)  # (num_points,)

with h5py.File('mean_shape_filter_8000points.h5', 'w') as f:
    # Save in the file as the filter
      f.create_dataset('data', data=selected_data)
      f.create_dataset('label', data=selected_labels)


# Pointcloud visualization
cell_labels = selected_labels
points = selected_data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
unique_labels = np.unique(cell_labels)
label_colors = matplotlib.colormaps.get_cmap('tab20')
for i, label in enumerate(unique_labels):
    label_points = points[cell_labels == label]
    color = label_colors(label / 16)[:-1]
    ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2], c=np.array(color).reshape(1, -1), label=f'label_{label}', alpha=0.5)

ax.legend()
plt.show()


# Mesh visualization
# mesh = trimesh.load_mesh(data_path)
# # mesh = mesh[indices]
#
# cell_labels = np.array(labels)
# # cell_labels = cell_labels[indices]
#
# unique_labels = np.unique(cell_labels)
# label_colors = matplotlib.colormaps.get_cmap('tab20')
#
# face_colors = np.zeros((len(mesh.faces), 3))
# for i, label in enumerate(unique_labels):
#     face_colors[cell_labels == label] = label_colors(label / 16)[:-1]
#
# # face_colors = np.ones((len(mesh.faces), 3))
# mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)

# mesh.show()
