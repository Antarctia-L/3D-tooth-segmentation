import os


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)


# Get the mesh list and mesh label
#####################################################
# 定义两个列表存储文件名
lower_list = []
upper_list = []

# 定义需要遍历的路径
path1 = os.path.join(parent_dir, 'Data', '3D_scans_per_patient_obj_files_b2')
path2 = os.path.join(parent_dir, 'Data', 'ground-truth_labels_instances_b2')

# 遍历路径label下的所有文件夹
for root, dirs, files in os.walk(path1):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        # 遍历每个文件夹中的文件
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            # 判断文件名中是否包含'lower'或'upper'
            if 'lower' in file_name and '_y' not in file_name:
                lower_list.append(file_path)
            elif 'upper' in file_name and '_y' not in file_name:
                upper_list.append(file_path)

# 初始化列表
lower_label = []
upper_label = []

for root, dirs, files in os.walk(path2):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        # 遍历每个文件夹中的文件
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            # 判断文件名中是否包含'lower'或'upper'
            if 'lower' in file_name and 'cell' not in file_name:
                lower_label.append(file_path)
            elif 'upper' in file_name and 'cell' not in file_name:
                upper_label.append(file_path)
#
print(len(lower_list))
print(len(upper_list))
print(len(lower_label))
print(len(upper_label))

with open("/Users/31475/PycharmProjects/Project_3D/Data/lists.txt", "w") as file:
    file.write(str(lower_list) + "\n\n" + str(upper_list) + "\n\n" + str(lower_label) + "\n\n" + str(upper_label))
    # file.write(str(lower_list) + "\n\n" + str(lower_label) + "\n\n")



