# -*- coding: utf-8 -*-

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import DC_Net
from model_1 import DGCNN_cls
import os
import numpy as np
from sklearn.metrics import jaccard_score
import argparse
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
list_file_path_lower = os.path.join(parent_dir, 'Data', 'lower_data_b1.h5')
list_file_path_upper = os.path.join(parent_dir, 'Data', 'upper_data_b1.h5')
list_file_path_test = os.path.join(parent_dir, 'Data', 'data_test.h5')

# 创建一个目录用于保存模型权?
if not os.path.exists("model_weights"):
    os.makedirs("model_weights")


def calculate_mean_iou(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_outputs = y_true.shape[1]
    iou_values = []
    for i in range(num_outputs):
        iou_values.append(jaccard_score(y_true[:, i], y_pred[:, i], average='weighted'))
    return np.mean(iou_values)

# 在每个epoch结束时对测试数据进行评估
def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    count = 0
    test_true = []
    test_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_reshaped = output.permute(0, 2, 1).contiguous().view(-1, 17)
            target_reshaped = target.view(-1)
            loss = torch.nn.functional.cross_entropy(output_reshaped, target_reshaped)

            test_loss += loss.item()
            count += 1

            _, predicted = torch.max(output.data, 1)
            test_true.extend(target.detach().cpu().numpy())
            test_pred.extend(predicted.detach().cpu().numpy())

    mean_iou = calculate_mean_iou(test_true, test_pred)

    return test_loss / count, mean_iou

# ...其他代码（例如模型定义，数据集定义等?..




class H5Dataset(Dataset):
    def __init__(self, h5_filename):
        self.h5_filename = h5_filename

        with h5py.File(self.h5_filename, 'r') as h5f:
            self.num_samples = len([key for key in h5f.keys() if key.startswith(f'points_')])
        self.scaler = StandardScaler()

    def __getitem__(self, index):
        with h5py.File(self.h5_filename, 'r') as h5f:
            points = h5f[f'points_{index}'][()]
            label = h5f[f'label_{index}'][()]
            points = points[:, :3, :]

            # Reshape points to (N, C) for normalization
            points = points.transpose((1, 0, 2)).reshape((3, -1)).transpose((1, 0))
            # Normalize points using Scikit-learn's StandardScaler
            points = self.scaler.fit_transform(points)
            # Reshape points back to (1, C, N)
            points = points.transpose((1, 0)).reshape((1, 3, -1))

        return {'points': torch.tensor(points, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    points = torch.stack([item['points'] for item in batch], dim=0)
    target = torch.stack([item['label'] for item in batch], dim=0)
    return points, target


def train(args):

    train_h5_filename = list_file_path_lower
    test_h5_filename = list_file_path_upper

    train_dataset = H5Dataset(train_h5_filename)
    test_dataset = H5Dataset(test_h5_filename)


    train_loader = DataLoader(train_dataset, num_workers=2,
                              batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, num_workers=2,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

    # Your training and testing code goes here


    print("Training dataset size: {}".format(len(train_dataset)))
    print("Training dataset length: {}".format(len(train_loader)))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DC_Net(args).to(device)
    model = DGCNN_cls(args).to(device)


    # Load pre-trained weights
    weights_path = "model_weights/DC_Net_best_weights_20.pth"
    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Could not find weights file at {weights_path}")

    # 
    num_gpus = torch.cuda.device_count()
    print(f'Number of available GPUs: {num_gpus}')
    
    if num_gpus > 1:
        model = nn.DataParallel(model)    
    # device = torch.device('cpu')
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_iou = 0.0
    for epoch in range(args.epochs):

        print("Starting epoch {}/{}".format(epoch + 1, args.epochs))
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        count = 0
        train_true = []
        train_pred = []

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()  # 添加时间?
            print("Processing batch {}".format(batch_idx + 1))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            # print(target)
            output_reshaped = output.permute(0, 2, 1).contiguous().view(-1, 17)
            target_reshaped = target.view(-1)
            loss = torch.nn.functional.cross_entropy(output_reshaped, target_reshaped)

            # loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1

            _, predicted = torch.max(output.data, 1)
            train_true.extend(target.detach().cpu().numpy())
            train_pred.extend(predicted.detach().cpu().numpy())

            batch_end_time = time.time()  # 添加时间?
            print("Batch {} processed in {:.2f} seconds".format(batch_idx + 1, batch_end_time - batch_start_time))


            mean_iou = calculate_mean_iou(train_true, train_pred)

            outstr = 'Train %d, loss: %.6f, train mean IoU: %.6f' % (epoch, train_loss * 1.0 / count, mean_iou)
            print(outstr)

            #
        Validation_loss, Validation_iou = evaluate(model, test_loader, device)
        print("Validation loss: {:.6f}, Validation mean IoU: {:.6f}".format(Validation_loss, Validation_iou))


        # 如果测试表现有所改善，则保存模型权重
        if Validation_iou > best_iou:
            best_iou = Validation_iou
        # if mean_iou > best_iou:
        #     best_iou = mean_iou
            print("Improved test mean IoU. Saving model weights...")
            torch.save(model.state_dict(), "model_weights/DC_Net_best_weights_TestUseless.pth")

        epoch_end_time = time.time()  # 添加时间?
        print("Epoch {}/{} completed in {:.2f} seconds".format(epoch + 1, args.epochs, epoch_end_time - epoch_start_time))

    # 在训练循环内部，周期结束时进行测试集评估
    test_loss, test_iou = evaluate(model, test_loader, device)
    print("Test loss: {:.6f}, Test mean IoU: {:.6f}".format(test_loss, test_iou))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DC_Net Training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=2, help='batch size for testing')
    parser.add_argument('--k', type=int, default=1, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    args = parser.parse_args()

    train(args)



