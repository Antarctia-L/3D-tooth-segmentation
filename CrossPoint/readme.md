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

* Dataset Production: `datasets/dataset.py`
    * Read the data file path: `datasets/dataloader.py`
    * Make model point clouds: Uniform downsampling
    * Extract key features: Coordinates of center points (x,y,z,num_channels = 3) 
    * Rearrange labels
    * Split datasets: Train : Validation: Test = 900 : 50 : 250
* Data Preprocessing: `datasets/shapenet_part.py`
    * Load dataset
    * Reshape tensors: Data - Pointclouds(num_points, 3) / Label - Categories(num_points, 1) / Seg - Segmentation part labels(num_points,)
* Fine-tuning: `train_partseg.py`
    * Pretrained dgcnn feature extractor
    * Adding output layers
    * Training: -- optimizer SGD --lr 0.001 --scheduler 'cos' --dropout 0.5 --k 40
    * Save best weights
    * Evaluation:
        * Loss: cal_loss
        * Accuracy: metrics.accuracy_score()
        * mIou
    * Visualization
        * Running logs
        * Train loss / Train mIou
        * Validation loss / Validation mIou
* Testing: `test_result.py`
    * Read the data file path
    * Make model point clouds
    * Extract key features
    * Data loader
    * Import model
    * Load checkpoints
    * Evaluation: Accuracy, mIou, DSC, PPV, SEN
    * Visualization
* Extended: `extended`
    * Mean shape filters
    * Geometric similarity calculation
