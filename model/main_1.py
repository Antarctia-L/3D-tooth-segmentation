import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import dataset
from model import DC_Net
from sklearn import metrics
import os
import numpy as np
from sklearn.metrics import jaccard_score
import argparse
import time

# 创建一个目录用于保存模型权重
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

# ...其他代码（例如模型定义，数据集定义等）...

def collate_fn(batch):
    points = torch.stack([item['points'] for item in batch], dim=0)
    target = torch.stack([item['label'] for item in batch], dim=0)
    return points, target




def train(args):
    train_dataset = dataset(partition='train')

    train_loader = DataLoader(dataset(partition='train'), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset(partition='test'), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Training data loader length: {len(train_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = DC_Net(args).to(device)
    # 如果有多个 GPU，使用 DataParallel 将模型复制到多个 GPU 上
    num_gpus = torch.cuda.device_count()
    print(f'Number of available GPUs: {num_gpus}')

    if num_gpus > 1:
        model = nn.DataParallel(model)
        # device = torch.device('cpu')

    num_workers = os.cpu_count()
    print(num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    best_iou = 0.0
    for epoch in range(args.epochs):

        print(f"Starting epoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        count = 0
        train_true = []
        train_pred = []

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()  # 添加时间戳
            print(f"Processing batch {batch_idx + 1}")
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

            batch_end_time = time.time()  # 添加时间戳
            print(f"Batch {batch_idx + 1} processed in {batch_end_time - batch_start_time:.2f} seconds")

            mean_iou = calculate_mean_iou(train_true, train_pred)

            outstr = 'Train %d, loss: %.6f, train mean IoU: %.6f' % (epoch, train_loss * 1.0 / count, mean_iou)
            print(outstr)

        # 在训练循环内部，周期结束时进行验证集评估
        Validation_loss, Validation_iou = evaluate(model, test_loader, device)
        print(f"Validation loss: {Validation_loss:.6f}, Validation mean IoU: {Validation_iou:.6f}")

        # 如果测试表现有所改善，则保存模型权重
        if Validation_iou > best_iou:
            best_iou = Validation_iou
            print("Improved test mean IoU. Saving model weights...")
            torch.save(model.state_dict(), "model_weights/DC_Net_best_weights.pth")

        epoch_end_time = time.time()  # 添加时间戳
        print(f"Epoch {epoch + 1}/{args.epochs} completed in {epoch_end_time - epoch_start_time:.2f} seconds")

    # 在训练循环外部，周期结束时进行测试集评估
    test_loss, test_iou = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.6f}, Test mean IoU: {test_iou:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DC_Net Training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=2, help='batch size for testing')
    parser.add_argument('--k', type=int, default=5, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout rate')
    args = parser.parse_args()

    train(args)



