import os
import ast
import h5py
import torch
import random
import trimesh
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

"""
    Processing:
    1. Choose an empty test file with only 4 crop teeth labeled (Randomly selected in the sample code)
    2. Produce the point cloud of the empty data
    3. Split the jaw into 5 areas by the 4 labeled crop teeth (gum not included) 
    4. Calculate the unlabeled tooth centers 
    5. Compare the shape of tooth with mean shape filters from its area
    6. Add the labeled teeth 
    7. Visualization

"""

# get data files path from list
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
list_file_path = os.path.join(parent_dir, 'Data', 'lists_1.txt')

# Select one test file (empty mesh)
with open(list_file_path, "r") as file:
    content = file.read()
    str_lists = content.split("\n\n")
    lists = [ast.literal_eval(str_list) for str_list in str_lists]
    data_path = lists[0][10]
    label_path = lists[2][10]

# Convert files to point clouds and labels（8000，3）（8000，）
label_file = label_path
f = open(label_file, 'r')
labels = ast.literal_eval(f.read())
f.close()
unique_classes = np.unique(labels)
obj_file = data_path
mesh = trimesh.load_mesh(obj_file)
vertices = mesh.vertices
cells = mesh.faces
center = np.mean(vertices[cells], axis=1)
center_min = np.min(center)
center_max = np.max(center)
center_normalized = (center - center_min) / (center_max - center_min) * 2 - 1
points = center_normalized
print(len(labels), points.shape)

# four classes of teeth
list_1 = [1, 2, 13, 14]  # molar teeth
list_2 = [3, 4, 11, 12]  # canine teeth
list_3 = [5, 6, 9, 10]  # incisor teeth
list_4 = [7, 8, 7, 8]  # front tooth
all_labels = unique_classes
all_labels = all_labels.tolist()
all_labels.remove(0)

# Four labels are randomly selected as split points
lists = [list_1, list_2, list_3, list_4]
elements = [list(set(lst) & set(unique_classes)) for lst in lists]
np.random.seed(42)
list_selected = [random.choice(elem) for elem in elements]
list_selected.sort()
# print(list_selected)

# Use the slicing operation to split into five lists
selected_labels = list_selected
split_indices = [all_labels.index(label) for label in selected_labels]
split_lists = []
start_index = 0
for idx in split_indices:
    split_lists.append(all_labels[start_index:idx])
    # all_labels.remove()
    start_index = idx+1
split_lists.append(all_labels[start_index:])
# eg. [[],[2,3],[5,6,7,8],[10,11],[13,14]]
# print(split_lists)


def calculate_similarity(small_clouds, large_cloud, key_points):
    """
        Parameters：
        large_cloud：numpy array，shape(N, 3), empty tooth area
        small_clouds：list of crop teeth pointclouds, which are numpy array，shape(N, 3)
        key_points：teeth centers for alignment

        Return：
        best_similarity
        best_small_cloud_index
        indices: nearest neighbor points on large point clouds

    """

    # Calculate the KDTree of large cloud
    large_kd_tree = KDTree(large_cloud)

    best_similarity = 0.0
    best_small_cloud = None

    # The inner loop traverses each small dot cloud cluster and its index
    for small_cloud_index, small_cloud in enumerate(small_clouds):
        # print(key_points)
        # Calculate the offset vector from the center point to the key point
        offset_vector = key_points - np.mean(small_cloud, axis=0)

        # Move the small point cloud according to the offset vector
        transformed_small_cloud = small_cloud + offset_vector

        # Use KDTree to find nearest neighbor points on large point clouds
        _, indices = large_kd_tree.query(transformed_small_cloud, k=1)

        # Calculate shape similarity # Euclidean distance
        distances = np.linalg.norm(large_cloud[indices] - transformed_small_cloud, axis=1)
        similarity = np.mean(1.0 / (1.0 + distances))
        # print(similarity)
        # Update the maximum similarity and corresponding small point cloud clusters
        if similarity > best_similarity:
            best_similarity = similarity
            best_small_cloud_index = small_cloud_index

    return best_similarity, best_small_cloud_index, indices

# mean shape filter file
h5_filename = '/Users/31475/PycharmProjects/Project_3D/model/semi_8000.h5'
hdf5_file = h5py.File(h5_filename, "r")
mean_points = hdf5_file['data'][()]
mean_labels = hdf5_file['label'][()]
print(mean_points.shape, mean_labels.shape)


indices = []
processed_data = []
processed_label = []
all_processed_data = []
all_processed_label = []
unique_classes = np.unique(mean_labels)
# for each sublist in the split_lists
for classes in split_lists:
    if not classes:
        continue
    all_center = []
    all_indices = []
    small_cloud = []
    # collect all the crop tooth filter
    for class_label in classes:
        filter_indices = np.where(np.array(mean_labels) == class_label)[0]
        merged_points = mean_points[filter_indices]
        small_cloud.append(merged_points)
    # for each empty teeth area, calculate the center point
    for class_label in classes:
        print(class_label)
        class_indices = np.where(np.array(labels) == class_label)[0]
        merged_points = points[class_indices]
        mean_center = np.mean(merged_points, axis=0)
        large_cloud = merged_points
        key_points = mean_center
        key_points = key_points.reshape(1, 3)

        best_similarity, small_cloud_index, best_indices = calculate_similarity(small_cloud, large_cloud, key_points)
        # print("results", classes[small_cloud_index], len(best_indices))
        # Get the pseudo-label for ground truth of the rest 10 teeth
        if small_cloud_index is not None:
            processed_data.append(large_cloud[best_indices])
            processed_label.append([classes[small_cloud_index]] * len(best_indices))

# Add 4 labeled crop teeth
for class_label in list_selected:
    print(class_label)
    class_indices = np.where(np.array(labels) == class_label)[0]
    merged_points = points[class_indices]
    processed_data.append(merged_points)
    processed_label.append([class_label] * len(class_indices))
all_processed_data = np.concatenate(processed_data, axis=0)
all_processed_label = np.concatenate(processed_label, axis=0)


# Pointcloud visualization
points = all_processed_data
cell_labels = all_processed_label

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

