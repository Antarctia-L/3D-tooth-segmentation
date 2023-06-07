import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x



class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 5

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(15 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 15 * 15)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(15, 15))

    def forward(self, x):
        batch_size = x.size(0)
        # print("T-Net input", x.shape)
        x = self.conv1(x)  # (batch_size, 15*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x = self.conv4(x)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DC_Net(nn.Module):
    def __init__(self, args, output_channels=17):
        super(DC_Net, self).__init__()
        self.args = args
        self.k = args.k
        self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(64)
        self.bn9 = nn.BatchNorm2d(64)
        self.bn10 = nn.BatchNorm2d(1024)
        self.bn11 = nn.BatchNorm2d(256)
        self.bn12 = nn.BatchNorm2d(256)
        self.bn13 = nn.BatchNorm2d(128)

        # self.bn15 = nn.BatchNorm1d(256)
        # self.bn16 = nn.BatchNorm1d(256)
        # self.bn17 = nn.BatchNorm1d(256)
        # self.bn18 = nn.BatchNorm1d(256)

        self.max1 = nn.AdaptiveMaxPool1d(32)
        self.avg1 = nn.AdaptiveAvgPool1d(32)
        self.max2 = nn.AdaptiveMaxPool1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(128 * 2, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(320, 1024, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv2d(1344, 256, kernel_size=1, bias=False),
                                    self.bn11,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv12 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    self.bn12,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv13 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                    self.bn13,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv14 = nn.Conv2d(128, 17, kernel_size=1, bias=False)
        # self.conv14 = nn.Sequential(nn.Conv2d(128, 17, kernel_size=1, bias=False),)

        self.bn15 = nn.BatchNorm1d(32)
        self.conv15 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=1, bias=False),
                                    self.bn15,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.bn16 = nn.BatchNorm1d(32)
        self.conv16 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=1, bias=False),
                                    self.bn16,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.bn17 = nn.BatchNorm1d(512)
        self.conv17 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                    self.bn17,
                                    nn.LeakyReLU(negative_slope=0.2))
        # self.linear11 = nn.Linear(args.emb_dims*2, 512, bias=False)
        # self.bn16 = nn.BatchNorm1d(512)
        # self.dp11 = nn.Dropout(p=args.dropout)
        # self.linear12 = nn.Linear(512, 256)
        # self.bn17 = nn.BatchNorm1d(256)
        # self.dp12 = nn.Dropout(p=args.dropout)
        # self.linear13 = nn.Linear(256, output_channels)

        # self.linear1 = nn.Linear(128, output_channels)

    def forward(self, x):
        # print("shape input", x.shape)
        x = x.squeeze(1)
        batch_size = x.size(0)
        num_points = x.size(2)
        output_channels = 17

        # print(x.shape)
        x0 = get_graph_feature(x, k=self.k)  # (batch_size, 15, num_points) -> (batch_size, 15*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 15, 15)
        x = x.transpose(2, 1)  # (batch_size, 15, num_points) -> (batch_size, num_points, 15)
        x = torch.bmm(x, t)  # (batch_size, num_points, 15) * (batch_size, 15, 15) -> (batch_size, num_points, 15)
        x = x.transpose(2, 1)  # (batch_size, num_points, 15) -> (batch_size, 15, num_points)
        # print("shape x before feature", x.shape)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        # print("shape conv3", x.shape)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # print("shape x", x.shape)
        # x = self.conv15(x)
        x1 = F.adaptive_max_pool1d(x, num_points)
        x2 = F.adaptive_avg_pool1d(x, num_points)
        x3 = torch.cat((x1, x2), dim=1)  # (batch_size, 128, num_points)
        # print("shape x3", x3.shape)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # print("shape feature", x.shape)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv6(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        # print("shape conv6", x.shape)
        x = x.max(dim=-1, keepdim=False)[0]
        # print("shape x", x2.shape)
        # x = self.conv16(x)
        x1 = F.adaptive_max_pool1d(x, num_points)
        x2 = F.adaptive_avg_pool1d(x, num_points)
        x4 = torch.cat((x1, x2), dim=1)  # (batch_size, 64, num_points)
        # print("shape x4", x4.shape)

        x = get_graph_feature(x4, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # print("shape feature", x.shape)
        x = self.conv7(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv8(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv9(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        # print("shape conv9", x.shape)
        x5 = x.max(dim=-1, keepdim=False)[0]
        # print("shape x5", x5.shape)

        # 64 * 3
        x = torch.cat((x3, x4, x5), dim=1)  # (batch_size, 64*3, num_points)
        # print("shape torch x", x.shape)
        k = self.k
        x = x.unsqueeze(-1).expand(-1, -1, -1, k)
        x = self.conv10(x)  # (batch_size, 64*3, num_points, k) -> (batch_size, 1024, num_points, k)

        # print("shape x10", x.shape)

        x = x.max(dim=-1, keepdim=False)[0]
        # x = self.conv17(x) # 1024 -> 512
        x = F.adaptive_max_pool1d(x, num_points)
        # print("shape max", x.shape)
        x = torch.cat((x, x3, x4, x5), dim=1)  # (batch_size, 512+64*3, num_points)
        # print("shape torch", x.shape)
        x = x.unsqueeze(-1).expand(-1, -1, -1, k)
        # print("before shape x11", x.shape)
        x = self.conv11(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv12(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv13(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv14(x)  # (batch_size, 128, num_points) -> (batch_size, 17, num_points)
        # print("shape x", x.shape)
        # x = x.max(dim=-1, keepdim=False)[0]
        # x = self.conv15(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        # x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        # x = F.leaky_relu(self.bn16(self.linear11(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        # x = self.dp11(x)
        # x = F.leaky_relu(self.bn17(self.linear12(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        # x = self.dp12(x)
        # x = self.linear13(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        x = x.max(dim=-1, keepdim=False)[0]
        # print(x.shape)
        x = x.transpose(2, 1).contiguous()
        # print(x.shape)
        softmax = nn.Softmax(dim=-1)
        x = softmax(x)
        # x = torch.nn.Softmax(dim=-1)(x.view(-1, output_channels))

        # print(x.shape)
        # x = x.view(batch_size, num_points, output_channels)
        # x = x.transpose(2,1).contiguous()
        # print(x.shape)

        return x