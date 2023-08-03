import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Define the path to traverse
path1 = os.path.join(parent_dir, 'Data', '3D_scans_per_patient_obj_files_b2')
path2 = os.path.join(parent_dir, 'Data', 'ground-truth_labels_instances_b2')

# Initialize mesh file list
lower_list = []
upper_list = []

# Traverse all folders under the path label
for root, dirs, files in os.walk(path1):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        # Traverse the files in each folder
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            # Determine whether the file name contains 'lower' or 'upper'
            if 'lower' in file_name and '_y' not in file_name:  # '_y' for rotation data made by data augmentation
                lower_list.append(file_path)
            elif 'upper' in file_name and '_y' not in file_name:
                upper_list.append(file_path)

# Initialize label file list
lower_label = []
upper_label = []

for root, dirs, files in os.walk(path2):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        # Traverse the files in each folder
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            # Determine whether the file name contains 'lower' or 'upper'
            if 'lower' in file_name and 'cell' not in file_name:
                lower_label.append(file_path)
            elif 'upper' in file_name and 'cell' not in file_name:
                upper_label.append(file_path)
#
print(len(lower_list))
print(len(upper_list))
print(len(lower_label))
print(len(upper_label))

# Save file
with open("/Users/31475/PycharmProjects/Project_3D/Data/lists.txt", "w") as file:
    file.write(str(lower_list) + "\n\n" + str(upper_list) + "\n\n" + str(lower_label) + "\n\n" + str(upper_label))
    # file.write(str(lower_list) + "\n\n" + str(lower_label) + "\n\n")



