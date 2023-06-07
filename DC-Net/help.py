import numpy as np
import meshio
import os
import json
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import pyvista as pv
from scipy import stats


#  Uniformly sample 10,000 mesh and regard each face center as a point.
#  1. the position of the face(3)
#  2. the 3-dimensional normal vector of the face surface(3)
#  3. the face shape feature(9)


# DownSampling 1 (不保存结构信息)(KNN效果很差）
#####################################################

def quick_uniform_sampling(tree, centers, n_samples):
    n_samples_per_dim = int(np.ceil(n_samples**(1/3)))
    x = np.linspace(centers[:, 0].min(), centers[:, 0].max(), n_samples_per_dim)
    y = np.linspace(centers[:, 1].min(), centers[:, 1].max(), n_samples_per_dim)
    z = np.linspace(centers[:, 2].min(), centers[:, 2].max(), n_samples_per_dim)

    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    _, sampled_indices = tree.query(grid_points, k=1)

    return np.unique(sampled_indices)[:n_samples]

def farthest_point_sampling(tree, centers, n_samples):
    current_point = tree.data[np.random.choice(tree.n, 1)][0]
    mask = np.ones(tree.n, dtype=bool)

    sampled_indices = []

    for _ in range(n_samples):
        distances, indices = tree.query(current_point, k=tree.n)
        indices = indices[::-1]  # 从最远的点开始
        farthest_index = None

        for index in indices:
            if mask[index]:
                farthest_index = index
                break

        if farthest_index is not None:
            mask[farthest_index] = False
            sampled_indices.append(farthest_index)

            current_point = centers[farthest_index]
        else:
            break

    return np.array(sampled_indices)

def combined_sampling(tree, centers, n_samples, initial_points=1000):
    initial_indices = quick_uniform_sampling(tree, centers, initial_points)
    initial_centers = centers[initial_indices]

    initial_tree = cKDTree(initial_centers)

    sampled_indices = farthest_point_sampling(initial_tree, initial_centers, n_samples)
    final_indices = initial_indices[sampled_indices]
    sampled_centers = centers[final_indices]

    return final_indices, sampled_centers

def read_stl_and_sample_points(stl_file, sampled_points=10000):
    mesh = meshio.read(stl_file)
    vertices = mesh.points[:, :3]
    cells = mesh.cells_dict["triangle"]

    centers = (vertices[cells[:, 0]] + vertices[cells[:, 1]] + vertices[cells[:, 2]]) / 3

    tree = cKDTree(centers)

    sampled_indices, sampled_centers = combined_sampling(tree, centers, sampled_points, initial_points=40000)

    return sampled_indices, sampled_centers

# print("Sampled indices:", indices[100])
# print("Sampled centers:", sampled_centers[100])




# Downsampling 2 (保留结构信息)(效果也不太好)
####################################################

def sample_mesh_cells_distance(mesh, num_samples, n_neighbors=5):
    input_mesh = mesh #meshio.read(input_stl_path, file_format="stl")
    vertices = input_mesh.points[:, :3]
    cells = input_mesh.get_cells_type("triangle")

    # 计算三角形网格的中心
    triangle_centers = np.mean(vertices[cells], axis=1)

    # 使用最近邻算法查找距离较远的三角形
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(triangle_centers)
    distances, _ = nbrs.kneighbors(triangle_centers)

    # 以距离的平均值作为采样权重
    weights = np.mean(distances, axis=1)
    weights /= np.sum(weights)

    # 按权重随机采样三角形
    sampled_indices = np.random.choice(cells.shape[0], num_samples, replace=False, p=weights)

    return sampled_indices



# 重点采样(效果还不如上面那个)
####################################################

def weighted_random_sampling(stl_file, labels, num_samples=10000, weight=50):
    input_mesh = meshio.read(stl_file)
    cells = input_mesh.get_cells_type("triangle")
    # 找出三个顶点标签不全相同的单元格
    unique_label_cells = [i for i, cell in enumerate(cells) if len(set([labels[cell[0]], labels[cell[1]], labels[cell[2]]])) > 1]

    # 为这些特殊的单元格分配权重
    weights = np.ones(len(cells))
    weights[unique_label_cells] = weight

    # 对所有单元格进行加权随机采样
    sampled_indices = np.random.choice(len(cells), size=num_samples, replace=False, p=weights / np.sum(weights))

    return sampled_indices





# save sampled mesh as stl file (存为新的stl文件)
########################################################

def save_sampled_mesh(input_stl_path, sampled_indices, output_stl_path):
    # 读取原始 STL 文件
    input_mesh = meshio.read(input_stl_path)
    cells = input_mesh.get_cells_type("triangle")

    # 根据采样索引获取采样后的三角形
    sampled_cells = cells[sampled_indices]

    # 创建新的 mesh 对象，使用原始顶点和采样后的单元
    sampled_mesh = meshio.Mesh(points=input_mesh.points, cells=[("triangle", sampled_cells)])

    # 将采样后的 mesh 保存为新的 STL 文件
    meshio.write(output_stl_path, sampled_mesh)




# 15vectors + label
#####################################################

def compute_triangle_normals(vertices, cells):
    vertex1, vertex2, vertex3 = vertices[cells[:, 0]], vertices[cells[:, 1]], vertices[cells[:, 2]]
    edge1 = vertex2 - vertex1
    edge2 = vertex3 - vertex1
    # print("edge1.shape:", edge1.shape)
    # print("edge2.shape:", edge2.shape)
    normals = np.cross(edge1, edge2)
    normalized_normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normalized_normals

# def mode(numbers):
#     counter = Counter(numbers)
#     frequency = counter.most_common()
#     if len(frequency) == 1:
#         return frequency[0][0]
#     elif frequency[0][1] == frequency[1][1]:
#         return None
#     else:
#         return frequency[0][0]

def PointCloudLabel(label_path, stl_file, index):
    # Label = []
    your_mesh = meshio.read(stl_file)
    vertices = your_mesh.points
    cell = your_mesh.cells_dict["triangle"]
    with open(label_path, 'r') as f:
        json_data = json.load(f)
        labels = json_data['instances']
    vertex_labels_of_cell = [labels[cell[index, 0]], labels[cell[index, 1]], labels[cell[index, 2]]]
    if len(set(vertex_labels_of_cell)) == 3:
        # Compute the center of the triangle
        center = (vertices[cell[index, 0]] + vertices[cell[index, 1]] + vertices[cell[index, 2]]) / 3
        # Calculate the distances to the center
        distances = [np.linalg.norm(vertices[cell[index, i]] - center) for i in range(3)]
        # Get the index of the closest vertex
        closest_vertex_index = np.argmin(distances)
        # Return the label of the closest vertex
        return vertex_labels_of_cell[closest_vertex_index]
        # return vertex_labels_of_cell[0]
    else:
        return stats.mode(vertex_labels_of_cell)[0][0]
    # return vertex_labels_of_cell


#原始cell label列表
def PointCloudLabelList(label_path, stl_file):
    your_mesh = meshio.read(stl_file)
    vertices = your_mesh.points[:, :3]
    cells = your_mesh.cells_dict["triangle"]

    with open(label_path, 'r') as f:
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



def PointCloudVector(mesh, index):
    vertices = mesh.points[:, :3]
    cells = mesh.cells_dict["triangle"]

    # Get the vertices of the specific triangle
    vertex1, vertex2, vertex3 = vertices[cells[index, 0]], vertices[cells[index, 1]], vertices[cells[index, 2]]

    # Compute the normal of the specific triangle
    normal = np.cross(vertex2 - vertex1, vertex3 - vertex1)
    normal /= np.linalg.norm(normal)

    # Compute the center of the specific triangle
    center = (vertex1 + vertex2 + vertex3) / 3

    v1 = vertex1 - center
    v2 = vertex2 - center
    v3 = vertex3 - center

    result = list(center) + list(normal) + list(v1) + list(v2) + list(v3)
    return result


# 重新分配标签
####################################################
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

def calculate_mesh_centers(stl_file):
    your_mesh = meshio.read(stl_file)
    vertices = your_mesh.points[:, :3]
    cells = your_mesh.cells_dict["triangle"]
    centers = (vertices[cells[:, 0]] + vertices[cells[:, 1]] + vertices[cells[:, 2]]) / 3
    return centers

def reassign_labels(mesh_centers, sampled_indices, labels, k_values=[2, 3, 5, 7]):
    labels = np.array(labels)  # 将 labels 转换为 NumPy 数组
    sampled_centers = mesh_centers[sampled_indices]
    sampled_labels = labels[sampled_indices]

    knn_classifiers = [KNeighborsClassifier(n_neighbors=k) for k in k_values]

    for knn_classifier in knn_classifiers:
        knn_classifier.fit(sampled_centers, sampled_labels)

    predicted_labels = [knn_classifier.predict(mesh_centers) for knn_classifier in knn_classifiers]

    # 使用Counter统计每个数据点的预测结果，选择出现次数最多的类别作为最终分类结果
    new_labels = [Counter(pred_labels).most_common(1)[0][0] for pred_labels in zip(*predicted_labels)]

    return np.array(new_labels)

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





# Visualization(形状)
#####################################################

def MeshPlotterDemo(stl_file, indices):
# def MeshPlotter(stl_file):


    # 从STL文件中读取原始mesh
    your_mesh = meshio.read(stl_file)

    vertices = your_mesh.points[:, :3]
    cells = your_mesh.cells_dict["triangle"]

    # 假设您已经有了sampled_indices
    sampled_indices = indices  # 示例：替换为实际采样得到的索引

    # 根据sampled_indices采样原始的mesh
    sampled_cells = cells#[sampled_indices]

    # 创建一个pyvista的PolyData对象
    poly_data = pv.PolyData(vertices, faces=np.hstack((np.full((sampled_cells.shape[0], 1), 3), sampled_cells)))

    # 可视化采样后的mesh
    plotter = pv.Plotter()
    plotter.add_mesh(poly_data, show_edges=True, color="white")
    plotter.show_grid()
    plotter.show()




# Visualization(加颜色)
#####################################################

def MeshPlotter(stl_file, labels):
    # 从STL文件中读取原始mesh
    your_mesh = meshio.read(stl_file)

    # 旋转
    # angle = np.pi / 4.0  # 旋转角度为45度
    # axis = np.array([0, 1, 0])  # 绕y轴旋转
    # your_mesh = rotate_mesh(your_mesh, angle, axis)

    vertices = your_mesh.points[:, :3]
    cells = your_mesh.cells_dict["triangle"]

    # 创建一个pyvista的PolyData对象
    poly_data = pv.PolyData(vertices, faces=np.hstack((np.full((cells.shape[0], 1), 3), cells)))

    # 创建颜色映射
    color_map = {
        0: [0.0, 0.0, 0.0],  # Black
        1: [1.0, 0.0, 0.0],  # Red
        2: [0.0, 0.0, 0.5],  # Green
        3: [0.0, 0.0, 0.0],  # Blue
        4: [0.0, 0.0, 0.0],  # Yellow
        5: [0.0, 0.0, 0.0],  # Magenta
        6: [0.0, 0.0, 0.0],  # Cyan
        7: [0.0, 0.0, 0.0],  # Orange
        8: [0.0, 0.0, 0.0],  # Purple
        9: [0.0, 0.0, 0.0],  # Light Blue
        10: [0.0, 0.0, 0.0],  # Pink
        11: [0.0, 0.0, 0.0],  # Lime Green
        12: [0.0, 0.0, 0.0],  # Chartreuse Green
        13: [0.0, 0.0, 0.0],  # Violet
        14: [0.0, 0.0, 0.0],  # Olive Green
        15: [0.0, 0.0, 0.0],  # Teal
        16: [0.0, 0.0, 0.0],  # Gray
    }

    # 根据标签为每个cell分配颜色
    cell_colors = np.array([color_map[label] for label in labels])

    # 将颜色添加到PolyData对象
    poly_data.cell_data["colors"] = cell_colors

    # 可视化带有颜色的mesh
    plotter = pv.Plotter()
    plotter.add_mesh(poly_data, scalars="colors", show_edges=True)
    plotter.show_grid()
    plotter.show()




def generate_label_to_id(label_file_paths):
    labels = set()
    for file_path in label_file_paths:
        with open(file_path, 'r') as f:
            label_data = json.load(f)
        for label in label_data:
            labels.add(label)

    print("Unique labels found:", len(labels))

    label_to_id = {label: i for i, label in enumerate(sorted(list(labels)))}
    return label_to_id


# 定义旋转函数
def rotate_mesh(mesh, angle, axis):
    # 将旋转轴标准化
    axis = axis / np.sqrt(np.dot(axis, axis))

    # 计算旋转矩阵
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    R = np.array([[t*axis[0]*axis[0] + c, t*axis[0]*axis[1] - s*axis[2], t*axis[0]*axis[2] + s*axis[1]],
                  [t*axis[0]*axis[1] + s*axis[2], t*axis[1]*axis[1] + c, t*axis[1]*axis[2] - s*axis[0]],
                  [t*axis[0]*axis[2] - s*axis[1], t*axis[1]*axis[2] + s*axis[0], t*axis[2]*axis[2] + c]])

    # 旋转顶点
    vertices = mesh.points[:, :3].T
    vertices = np.dot(R, vertices)
    mesh.points = vertices.T

    return mesh


# # example


# #
# #
# stl_file = lower_list[99]
# label_path = lower_label[99]
# # # sampled_indices, sampled_centers = read_stl_and_sample_points(stl_file)
# mesh = meshio.read(lower_list)
# 加载3D模型
# mesh = meshio.read("model.obj")
# 指定旋转轴和角度进行数据增强
# angle = np.pi/4.0  # 旋转角度为45度
# axis = np.array([0, 1, 0])  # 绕y轴旋转
# rotated_mesh = rotate_mesh(mesh, angle, axis)
#

#
# vertices = your_mesh.points[:, :3]
# cells = your_mesh.cells_dict["triangle"]
# print(len(cells))
#
# with open(lower_label, 'r') as f:
#     json_data = json.load(f)
#     labels = json_data # ['instances'] # 请用实际的顶点标签列表替换
# # #
# print(len(labels))

# # 读取标签列表
# L = PointCloudLabelList(label_path, stl_file)
#

# import json
# import os
# import numpy as np

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

def process_labels(lower_label, lower_list):
    for idx, label_path in enumerate(lower_label):
        stl_file = lower_list[idx]

        # 重新计算标签列表
        La = PointCloudLabelList(label_path, stl_file)
        La = np.array(La)
        # print(type(La))
        L = rearrange(La)
        L = L.tolist()
        # print(L)

        # 生成新文件名
        dir_name, file_name = os.path.split(label_path)
        file_name_without_ext, ext = os.path.splitext(file_name)
        new_file_name = file_name_without_ext + "_cell" + ext
        new_label_path = os.path.join(dir_name, new_file_name)

        # 将新的标签列表保存为JSON文件
        with open(new_label_path, 'w') as outfile:
            json.dump(L, outfile, cls=NumpyEncoder)

        print(f"Processed {label_path} and saved as {new_label_path}")

# def process_data(lower_list):
#     for idx, lower_list in enumerate(lower_list):
#         stl_file = lower_list # [idx]
#
#         # 重新计算标签列表
#         # L = PointCloudLabelList(label_path, stl_file)
#         mesh = meshio.read(stl_file)
#         # 指定旋转轴和角度进行数据增强
#         angle = np.pi/4.0  # 旋转角度为90度
#         axis = np.array([0, 1, 0])  # 绕y轴旋转
#         rotated_mesh = rotate_mesh(mesh, angle, axis)
#
#         # 生成新文件名
#         dir_name, file_name = os.path.split(lower_list)
#         file_name_without_ext, ext = os.path.splitext(file_name)
#         new_file_name = file_name_without_ext + "_y45" + ext
#         new_label_path = os.path.join(dir_name, new_file_name)
#
#         # 将新的标签列表保存为JSON文件
#         # with open(new_label_path, 'w') as outfile:
#         #     json.dump(L, outfile, cls=NumpyEncoder)
#         meshio.write(new_label_path, rotated_mesh)
#
#         print(f"Processed {lower_list} and saved as {new_label_path}")

# 调用函数处理所有lower_label中的路径
# import ast
#     # dataset = h5f[key]
#     #     print(f"Dataset shape: {dataset.shape}")
#     #     print(f"Dataset dtype: {dataset.dtype}")
#     #
#     #     print(f"First few elements: {dataset[:5]}")
# with open("C:\\Users\\31475\\PycharmProjects\\Project_3D\\Data\\lists.txt", "r") as file:
#     content = file.read()
#     str_lists = content.split("\n\n")
#     lists = [ast.literal_eval(str_list) for str_list in str_lists]
# #
# #
# # # #
# List = lists[1]
# label_ = lists[3]
# # print(len(List))
# # # print(List[0])
# process_labels(label_, List)
# List = 'C:\\Users\\31475\\PycharmProjects\\Project_3D\\Data\\3D_scans_per_patient_obj_files_b1\\013FHA7K\\013FHA7K_lower.obj'
# label = 'C:\\Users\\31475\\PycharmProjects\\Project_3D\\Data\\ground-truth_labels_instances_b1\\013FHA7K\\013FHA7K_lower_cell.json'
#
# with open(label, 'r') as f:
#     json_data = json.load(f)
#     labels = json_data#['labels'] # 请用实际的顶点标签列表替换
#
#
# # # #
# #
# # # 计算 mesh 中心点
# mesh_centers = calculate_mesh_centers(List)
# # # 已知采样索引
# # # sampled_indices = weighted_random_sampling(stl_file, L)
# # # print(len(sampled_indices))
# num_samples = 10000
# # # # sampled_indices = np.random.choice(182645, num_samples, replace=False)
# # #sampled_indices = sample_mesh_cells_distance(List, num_samples)
# sampled_indices, sampled_centers = read_stl_and_sample_points(List)
# # # 从文件或其他来源获取采样索引列表
# # #
# # # # 重新分配标签
# new_labels = reassign_labels(mesh_centers, sampled_indices, labels)
# print(len(new_labels))
# # #
# # # # 可视化原模型和KNN模型
# # # MeshPlotter(lower_list, labels)
# MeshPlotter(List, labels)
# # # # print(index)
# # # # print(Results)
# # # # print("label:", Label)

