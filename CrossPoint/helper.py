import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import json
import torch.nn.init as init
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda:0')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class Transform_Net_test(nn.Module):
    def __init__(self):
        super(Transform_Net_test, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg_test(nn.Module):
    def __init__(self, seg_num_all=None, pretrain=True):
    # def __init__(self, args):
        super(DGCNN_partseg_test, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = 40
        self.emb_dims = 1024
        self.dropout = 0.5
        self.pretrain = pretrain
        self.transform_net = Transform_Net_test()
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.inv_head = nn.Sequential(
                            nn.Linear(self.emb_dims, self.emb_dims),
                            nn.BatchNorm1d(self.emb_dims),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.emb_dims, 256)
                            )
        
        if not self.pretrain:
            self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp1 = nn.Dropout(p=self.dropout)
            self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                       self.bn9,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp2 = nn.Dropout(p=self.dropout)
            self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                       self.bn10,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l = None):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        
        if self.pretrain:
            print("Pretrain")
            x = x.squeeze()
            inv_feat = self.inv_head(x)
            
            return x, inv_feat, x
        
        else:
            l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
            l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

            x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
            x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

            x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

            x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
            x = self.dp1(x)
            x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
            x = self.dp2(x)
            x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
            x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
            
            return x


def sample_mesh_cells_distance(filepath, filetype, num_samples=10000, n_neighbors=5):

    mesh = trimesh.load(filepath, file_type=filetype)
    vertices = mesh.vertices
    cells = mesh.faces

    triangle_centers = np.mean(vertices[cells], axis=1)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(triangle_centers)
    distances, _ = nbrs.kneighbors(triangle_centers)

    weights = np.mean(distances, axis=1)
    weights /= np.sum(weights)

    np.random.seed(42)  
    sampled_indices = np.random.choice(cells.shape[0], num_samples, replace=False, p=weights)
    # print(sampled_indices)
    center = np.mean(vertices[cells[sampled_indices]], axis=1)
    center_min = np.min(center)
    center_max = np.max(center)
    center_normalized = (center - center_min) / (center_max - center_min) * 2 - 1

    X_input = np.hstack(center_normalized)

    return X_input, sampled_indices  # Changed to return two separate values.


def DataProcessing(test_file, label_cells, num_points, num_classes, num_channels):
    
    points, indices = sample_mesh_cells_distance(test_file, "obj", num_samples=num_points, n_neighbors=5)
    
    points = torch.tensor(points, dtype=torch.float32).view(num_points, num_channels)
    #print('points', points.shape)
    
    label_cells = [label_cells[i] for i in indices]
    seg_tensor = torch.tensor(label_cells, dtype=torch.int64)
    seg = seg_tensor.unsqueeze(0)
    points = points.unsqueeze(0)
    label = torch.tensor([[1]], dtype=torch.int64)
   
    return points, label, seg

def PointCloudLabelList(data_filename, label_filename):
    mesh = trimesh.load_mesh(data_filename)
    vertices = mesh.vertices
    cells = mesh.faces

    with open(label_filename, 'r') as f:
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



def process_labels(data_filename, label_filename):

    label_cell = PointCloudLabelList(data_filename, label_filename)
    label_cell = np.array(label_cell)
    # print(type(La))
    label_cell_rerrange = rearrange(label_cell)
    label_cell_rerrange = label_cell_rerrange.tolist()
    #print(len(label_cell_rerrange))

    return label_cell_rerrange # new_label_path



def visualization_pointcloud(points, cell_labels):
    
    points = points.squeeze(0)
    # print(points.shape,cell_labels.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(cell_labels)
    label_colors = cm.get_cmap('tab20')

    for i, label in enumerate(unique_labels):
        label_points = points[cell_labels == label]
        color = label_colors(label / 16)[:-1]  
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


