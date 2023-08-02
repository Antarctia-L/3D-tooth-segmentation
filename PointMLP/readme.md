# Supervised Learning: PointMLP

## Citation
This project is a reproduction of PointMLP on the public data set MICCAI dental scan data set.
#### [Paper Link](https://arxiv.org/abs/2301.10531) | [Project Page](https://github.com/ananyajana/tooth_mesh_seg) 

## Dependencies

Refer `requirements.txt` for the required packages.

## Train

Run `python step3_training.py`

## Checkpoints

PointMLP pretrained models on MICCAI dataset.

300 dataset/24 features/8k points [here.](https://drive.google.com/file/d/1FWynGrOIT5wp_tVbSmvu9pS5VZwOI_zY/view?usp=sharing)

520 dataset/15 features/10k points [here.](https://drive.google.com/file/d/14JV7P0UJc2k3qHvAYHte7cXiMX23209V/view?usp=sharing)

## Test

1. Enter your own test file and checkpoints directories
2. Run `python test_result.py`/`python test_result_10k.py`

## Progress

Dataset production, model modification, data preprocessing, fine-tuning, evaluation, visualization