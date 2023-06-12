import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import json

from scipy import stats
from time import time

def knn(x, k):
    k = k + 1
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

class Attention_block(nn.Module):
    '''
    Used in attention U-Net.
    '''
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz, chunk_size=500):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
        chunk_size: size of the subset to compute distances for at a time
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N)

    new_xyz_chunks = new_xyz.chunk(chunk_size, dim=1)

    group_idx_list = []
    for new_xyz_chunk in new_xyz_chunks:
        _, chunk_S, _ = new_xyz_chunk.shape
        chunk_group_idx = group_idx.repeat([B, chunk_S, 1])
        
        sqrdists = square_distance(new_xyz_chunk, xyz)
        chunk_group_idx[sqrdists > radius ** 2] = N
        chunk_group_idx = chunk_group_idx.sort(dim=-1)[0][:, :, :nsample]

        group_first = chunk_group_idx[:, :, 0].view(B, chunk_S, 1).repeat([1, 1, nsample])
        mask = chunk_group_idx == N
        chunk_group_idx[mask] = group_first[mask]

        group_idx_list.append(chunk_group_idx)

    group_idx = torch.cat(group_idx_list, dim=1)

    return group_idx

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) * 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    torch.cuda.empty_cache()

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()

    new_points = index_points(points, idx)
    torch.cuda.empty_cache()

    if returnfps:
        return new_xyz, new_points, idx
    else:
        return new_xyz, new_points

class Attention_block(nn.Module):
    '''
    Used in attention U-Net.
    '''
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.leaky_relu(g1+x1, negative_slope=0.2)
        psi = self.psi(psi)

        return psi, 1. - psi


class LPFA(nn.Module):
    def __init__(self, in_channel, out_channel, k, mlp_num=2, initial=False):
        super(LPFA, self).__init__()
        self.k = k
        self.device = torch.device('cuda')
        self.initial = initial

        if not initial:
            self.xyz2feature = nn.Sequential(
                        nn.Conv2d(9, in_channel, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_channel))

        self.mlp = []
        for _ in range(mlp_num):
            self.mlp.append(nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                 nn.BatchNorm2d(out_channel),
                                 nn.LeakyReLU(0.2)))
            in_channel = out_channel
        self.mlp = nn.Sequential(*self.mlp)        

    def forward(self, x, xyz, idx=None):
        x = self.group_feature(x, xyz, idx)
        x = self.mlp(x)

        if self.initial:
            x = x.max(dim=-1, keepdim=False)[0]
        else:
            x = x.mean(dim=-1, keepdim=False)

        return x

    def group_feature(self, x, xyz, idx):
        batch_size, num_dims, num_points = x.size()

        if idx is None:
            idx = knn(xyz, k=self.k)[:,:,:self.k]  # (batch_size, num_points, k)

        idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        xyz = xyz.transpose(2, 1).contiguous() # bs, n, 3
        point_feature = xyz.view(batch_size * num_points, -1)[idx, :]
        point_feature = point_feature.view(batch_size, num_points, self.k, -1)  # bs, n, k, 3
        points = xyz.view(batch_size, num_points, 1, 3).expand(-1, -1, self.k, -1)  # bs, n, k, 3

        point_feature = torch.cat((points, point_feature, point_feature - points),
                                dim=3).permute(0, 3, 1, 2).contiguous()

        if self.initial:
            return point_feature

        x = x.transpose(2, 1).contiguous() # bs, n, c
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)  #bs, n, k, c
        x = x.view(batch_size, num_points, 1, num_dims)
        feature = feature - x

        feature = feature.permute(0, 3, 1, 2).contiguous()
        point_feature = self.xyz2feature(point_feature)  #bs, c, n, k
        feature = F.leaky_relu(feature + point_feature, 0.2)
        return feature #bs, c, n, k


class PointNetFeaturePropagation2(nn.Module):
    def __init__(self, in_channel, mlp, att=None):
        super(PointNetFeaturePropagation2, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        self.att = None
        if att is not None:
            self.att = Attention_block(F_g=att[0],F_l=att[1],F_int=att[2])
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S], skipped xyz
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S], skipped features
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        # skip attention
        if self.att is not None:
           psix, psig = self.att(interpolated_points.permute(0, 2, 1), points1)
           points1 = points1 * psix
           
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.leaky_relu(bn(conv(new_points)), 0.2)

        return new_points


class CIC(nn.Module):
    def __init__(self, npoint, radius, k, in_channels, output_channels, bottleneck_ratio=2, mlp_num=2, curve_config=None):
        super(CIC, self).__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.bottleneck_ratio = bottleneck_ratio
        self.radius = radius
        self.k = k
        self.npoint = npoint

        planes = in_channels // bottleneck_ratio

        self.use_curve = curve_config is not None
        if self.use_curve:
            self.curveaggregation = CurveAggregation(planes)
            self.curvegrouping = CurveGrouping(planes, k, curve_config[0], curve_config[1])

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels,
                      planes,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(in_channels // bottleneck_ratio),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv1d(planes, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels))

        if in_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels,
                          output_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm1d(output_channels))

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.maxpool = MaskedMaxPool(npoint, radius, k)

        self.lpfa = LPFA(planes, planes, k, mlp_num=mlp_num, initial=False)

    def forward(self, xyz, x):
 
        # max pool
        if xyz.size(-1) != self.npoint:
            xyz, x = self.maxpool(
                xyz.transpose(1, 2).contiguous(), x)
            xyz = xyz.transpose(1, 2)

        shortcut = x
        x = self.conv1(x)  # bs, c', n

        idx = knn(xyz, self.k)

        if self.use_curve:
            # curve grouping
            curves = self.curvegrouping(x, xyz, idx[:,:,1:]) # avoid self-loop

            # curve aggregation
            x = self.curveaggregation(x, curves)

        x = self.lpfa(x, xyz, idx=idx[:,:,:self.k]) #bs, c', n, k

        x = self.conv2(x)  # bs, c, n

        if self.in_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)

        x = self.relu(x + shortcut)

        return xyz, x

class CurveAggregation(nn.Module):
    def __init__(self, in_channel):
        super(CurveAggregation, self).__init__()
        self.in_channel = in_channel
        mid_feature = in_channel // 2
        self.conva = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convb = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convc = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convn = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convl = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convd = nn.Sequential(
            nn.Conv1d(mid_feature * 2,
                      in_channel,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(in_channel))
        self.line_conv_att = nn.Conv2d(in_channel,
                                       1,
                                       kernel_size=1,
                                       bias=False)

    def forward(self, x, curves):
        curves_att = self.line_conv_att(curves)  # bs, 1, c_n, c_l

        curver_inter = torch.sum(curves * F.softmax(curves_att, dim=-1), dim=-1)  #bs, c, c_n
        curves_intra = torch.sum(curves * F.softmax(curves_att, dim=-2), dim=-2)  #bs, c, c_l

        curver_inter = self.conva(curver_inter) # bs, mid, n
        curves_intra = self.convb(curves_intra) # bs, mid ,n

        x_logits = self.convc(x).transpose(1, 2).contiguous()
        x_inter = F.softmax(torch.bmm(x_logits, curver_inter), dim=-1) # bs, n, c_n
        x_intra = F.softmax(torch.bmm(x_logits, curves_intra), dim=-1) # bs, l, c_l
        

        curver_inter = self.convn(curver_inter).transpose(1, 2).contiguous()
        curves_intra = self.convl(curves_intra).transpose(1, 2).contiguous()

        x_inter = torch.bmm(x_inter, curver_inter)
        x_intra = torch.bmm(x_intra, curves_intra)

        curve_features = torch.cat((x_inter, x_intra),dim=-1).transpose(1, 2).contiguous()
        x = x + self.convd(curve_features)

        return F.leaky_relu(x, negative_slope=0.2)

class CurveGrouping(nn.Module):
    def __init__(self, in_channel, k, curve_num, curve_length):
        super(CurveGrouping, self).__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.in_channel = in_channel
        self.k = k

        self.att = nn.Conv1d(in_channel, 1, kernel_size=1, bias=False)

        self.walk = Walk(in_channel, k, curve_num, curve_length)

    def forward(self, x, xyz, idx):
        # starting point selection in self attention style
        x_att = torch.sigmoid(self.att(x))
        x = x * x_att

        _, start_index = torch.topk(x_att,
                                    self.curve_num,
                                    dim=2,
                                    sorted=False)
        # start_index = start_index.squeeze().unsqueeze(2)
        start_index = start_index.squeeze().unsqueeze(-1)

        curves = self.walk(xyz, x, idx, start_index)  #bs, c, c_n, c_l
        
        return curves

class MaskedMaxPool(nn.Module):
    def __init__(self, npoint, radius, k):
        super(MaskedMaxPool, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k

    def forward(self, xyz, features):
        sub_xyz, neighborhood_features = sample_and_group(self.npoint, self.radius, self.k, xyz, features.transpose(1,2))

        neighborhood_features = neighborhood_features.permute(0, 3, 1, 2).contiguous()
        sub_features = F.max_pool2d(
            neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]]
        )  # bs, c, n, 1
        sub_features = torch.squeeze(sub_features, -1)  # bs, c, n
        return sub_xyz, sub_features

def batched_index_select(input, dim, index):
	views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index)

def gumbel_softmax(logits, dim, temperature=1):
    """
    ST-gumple-softmax w/o random gumbel samplings
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = F.softmax(logits / temperature, dim=dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    y_hard = (y_hard - y).detach() + y
    return y_hard


class Walk(nn.Module):
    '''
    Walk in the cloud
    '''
    def __init__(self, in_channel, k, curve_num, curve_length):
        super(Walk, self).__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.k = k

        self.agent_mlp = nn.Sequential(
            nn.Conv2d(in_channel * 2,
                        1,
                        kernel_size=1,
                        bias=False), nn.BatchNorm2d(1))
        self.momentum_mlp = nn.Sequential(
            nn.Conv1d(in_channel * 2,
                        2,
                        kernel_size=1,
                        bias=False), nn.BatchNorm1d(2))

    def crossover_suppression(self, cur, neighbor, bn, n, k):
        # cur: bs*n, 3
        # neighbor: bs*n, 3, k
        neighbor = neighbor.detach()
        cur = cur.unsqueeze(-1).detach()
        dot = torch.bmm(cur.transpose(1,2), neighbor) # bs*n, 1, k
        norm1 = torch.norm(cur, dim=1, keepdim=True)
        norm2 = torch.norm(neighbor, dim=1, keepdim=True)
        divider = torch.clamp(norm1 * norm2, min=1e-8)
        ans = torch.div(dot, divider).squeeze() # bs*n, k

        # normalize to [0, 1]
        ans = 1. + ans
        ans = torch.clamp(ans, 0., 1.0)

        return ans.detach()

    def forward(self, xyz, x, adj, cur):
        bn, c, tot_points = x.size()

        # raw point coordinates
        xyz = xyz.transpose(1,2).contiguous # bs, n, 3

        # point features
        x = x.transpose(1,2).contiguous() # bs, n, c

        flatten_x = x.view(bn * tot_points, -1)
        batch_offset = torch.arange(0, bn, device=torch.device('cuda')).detach() * tot_points

        # indices of neighbors for the starting points
        tmp_adj = (adj + batch_offset.view(-1,1,1)).view(adj.size(0)*adj.size(1),-1) #bs, n, k
    
        # batch flattened indices for teh starting points
        flatten_cur = (cur + batch_offset.view(-1,1,1)).view(-1)

        curves = []

        # one step at a time
        for step in range(self.curve_length):

            if step == 0:
                # get starting point features using flattend indices
                starting_points =  flatten_x[flatten_cur, :].contiguous()
                pre_feature = starting_points.view(bn, self.curve_num, -1, 1).transpose(1,2) # bs * n, c
            else:
                # dynamic momentum
                cat_feature = torch.cat((cur_feature.squeeze(), pre_feature.squeeze()),dim=1)
                att_feature = F.softmax(self.momentum_mlp(cat_feature),dim=1).view(bn, 1, self.curve_num, 2) # bs, 1, n, 2
                cat_feature = torch.cat((cur_feature, pre_feature),dim=-1) # bs, c, n, 2
                
                # update curve descriptor
                pre_feature = torch.sum(cat_feature * att_feature, dim=-1, keepdim=True) # bs, c, n
                pre_feature_cos =  pre_feature.transpose(1,2).contiguous().view(bn * self.curve_num, -1)

            pick_idx = tmp_adj[flatten_cur] # bs*n, k
            
            # get the neighbors of current points
            pick_values = flatten_x[pick_idx.view(-1),:]

            # reshape to fit crossover suppresion below
            pick_values_cos = pick_values.view(bn * self.curve_num, self.k, c)
            pick_values = pick_values_cos.view(bn, self.curve_num, self.k, c)
            pick_values_cos = pick_values_cos.transpose(1,2).contiguous()
            
            pick_values = pick_values.permute(0,3,1,2) # bs, c, n, k

            pre_feature_expand = pre_feature.expand_as(pick_values)
            
            # concat current point features with curve descriptors
            pre_feature_expand = torch.cat((pick_values, pre_feature_expand),dim=1)
            
            # which node to pick next?
            pre_feature_expand = self.agent_mlp(pre_feature_expand) # bs, 1, n, k

            if step !=0:
                # cross over supression
                d = self.crossover_suppression(cur_feature_cos - pre_feature_cos,
                                               pick_values_cos - cur_feature_cos.unsqueeze(-1), 
                                               bn, self.curve_num, self.k)
                d = d.view(bn, self.curve_num, self.k).unsqueeze(1) # bs, 1, n, k
                pre_feature_expand = torch.mul(pre_feature_expand, d)

            pre_feature_expand = gumbel_softmax(pre_feature_expand, -1) #bs, 1, n, k

            cur_feature = torch.sum(pick_values * pre_feature_expand, dim=-1, keepdim=True) # bs, c, n, 1

            cur_feature_cos = cur_feature.transpose(1,2).contiguous().view(bn * self.curve_num, c)

            cur = torch.argmax(pre_feature_expand, dim=-1).view(-1, 1) # bs * n, 1

            flatten_cur = batched_index_select(pick_idx, 1, cur).squeeze() # bs * n

            # collect curve progress
            curves.append(cur_feature)

        return torch.cat(curves,dim=-1)




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


def process_labels(data_filename, label_filename):
    
    # 重新计算标签列表
    label_cell = PointCloudLabelList(data_filename, label_filename)
    label_cell = np.array(label_cell)
    # print(type(La))
    label_cell_rerrange = rearrange(label_cell)
    label_cell_rerrange = label_cell_rerrange.tolist()
    #print(len(label_cell_rerrange))


    return label_cell_rerrange # new_label_path

def visualization(data_filename, labels):

    mesh = trimesh.load_mesh(data_filename)
    cell_labels = np.array(labels)
    unique_labels = np.unique(cell_labels)
    label_colors = matplotlib.cm.get_cmap('tab20')

    face_colors = np.zeros((len(mesh.faces), 3))
    for i, label in enumerate(unique_labels):
        face_colors[cell_labels == label] = label_colors(label / 16)[:-1]  # 只提取 RGB 颜色值，而不是 RGBA

    # face_colors = np.ones((len(mesh.faces), 3))
    mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)

    # 可视化
    mesh.show()


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


