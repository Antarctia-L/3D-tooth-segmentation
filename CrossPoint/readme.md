# Semi-supervised Learning: CrossPoint

## Citation
This project is a reproduction of CrossPoint on the public data set MICCAI dental scan data set.
#### [Paper Link](https://arxiv.org/abs/2203.00680) | [Project Page](https://mohamedafham.github.io/CrossPoint/) 

## Dependencies

Refer `requirements.txt` for the required packages.

## Pretrained Models

CrossPoint pretrained models with DGCNN feature extractor are available [here.](https://drive.google.com/drive/folders/10TVEIRUBCh3OPulKI4i2whYAcKVdSURn?usp=sharing)

## 3D Object Part Segmentation

For fine-tuning experiment for 3D part segmentation.

Run `python train_partseg.py --exp_name teeth_partseg --pretrained_path dgcnn_partseg_best.pth --batch_size 8 --k 40 --test_batch_size 8 --epochs 300` 

## Fine-tuned Models

CrossPoint fine-tuned models on MICCAI dataset [here.](https://drive.google.com/file/d/11oIWQSW02WV5wxXp8l4j2t2l9Ye8L5zi/view?usp=drive_link)

## Test

1. Use your own test file and checkpoints directory
2. Run `python test_result.py`

## Progress

Dataset production, data preprocessing, fine-tuning, evaluation, visualization