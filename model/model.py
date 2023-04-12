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
        x = x.view(batch_size, 15, 15)  # (batch_size, 3*3) -> (batch_size, 3, 3)

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
        self.bn10 = nn.BatchNorm1d(1024)
        self.bn11 = nn.BatchNorm1d(256)
        self.bn12 = nn.BatchNorm1d(256)
        self.bn13 = nn.BatchNorm1d(128)

        self.max1 = nn.AdaptiveMaxPool1d(32)
        self.avg1 = nn.AdaptiveAvgPool1d(32)
        self.max2 = nn.AdaptiveMaxPool1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(15 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(704, 256, kernel_size=1, bias=False),
                                   self.bn11,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv12 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn12,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv13 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn13,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv14 = nn.Conv1d(128, 17, kernel_size=1, bias=False)

        # self.linear1 = nn.Linear(128, output_channels)


    def forward(self, x):
        # print("shape input", x.shape)
        batch_size = x.size(0)
        num_points = x.size(3)

        x = x.squeeze(1)
        x0 = get_graph_feature(x, k=self.k)  # (batch_size, 15, num_points) -> (batch_size, 15*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 15, 15)
        x = x.transpose(2, 1)  # (batch_size, 15, num_points) -> (batch_size, num_points, 15)
        x = torch.bmm(x, t)  # (batch_size, num_points, 15) * (batch_size, 15, 15) -> (batch_size, num_points, 15)
        x = x.transpose(2, 1)  # (batch_size, num_points, 15) -> (batch_size, 15, num_points)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        # print("shape conv3", x.shape)
        x = x.max(dim=-1, keepdim=False)[0]
        print("shape x", x.shape)
        x = x.transpose(1, 2)
        x1 = self.max1(x)
        x2 = self.avg1(x)
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        # print("shape x1", x1.shape)
        # print("shape x2", x2.shape)
        x3 = torch.cat((x1, x2), dim=1)  # (batch_size, 2 * emb_dims, num_points)
        # print("shape x3", x3.shape)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        print("shape feature", x.shape)
        x = self.conv4(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv6(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        # print("shape conv6", x.shape)
        x = x.max(dim=-1, keepdim=False)[0]
        print("shape x", x2.shape)
        x = x.transpose(1, 2)
        x1 = self.max1(x)
        x2 = self.avg1(x)
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        x4 = torch.cat((x1, x2), dim=1)  # (batch_size, 2 * emb_dims, num_points)
        # print("shape x4", x4.shape)

        x = get_graph_feature(x4, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # print("shape feature", x.shape)
        x = self.conv7(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv8(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv9(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        # print("shape conv9", x.shape)
        x5 = x.max(dim=-1, keepdim=False)[0]
        # print("shape x5", x5.shape)

        x = torch.cat((x3, x4, x5), dim=1)   # (batch_size, 64*3, num_points)
        # print("shape torch x", x.shape)

        x = self.conv10(x)  # (batch_size, 64*3, num_points, k) -> (batch_size, 1024, num_points, k)
        print("shape x", x.shape)
        x = x.transpose(1, 2)
        x = self.max2(x)
        x = x.transpose(1, 2) # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        # print("shape max", x.shape)
        x = torch.cat((x, x3, x4, x5), dim=1)   # (batch_size, 512+64*3, num_points)
        # print("shape torch", x.shape)
        x = self.conv11(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv12(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv13(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        # print("shape conv13", x.shape)
        x = self.conv14(x)  # (batch_size, 256, num_points) -> (batch_size, 17, num_points)
        # print("shape x", x.shape)
        return x