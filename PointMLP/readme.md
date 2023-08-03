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

* Dataset Production: `dataset.py`
    * Read the data file path: `dataloader.py`
    * Make model point clouds: Uniform downsampling
    * Extract key features: 
        * 15 features: Coordinates of center point(3) + normal vector(3) +  Coordinates of three vertices(9)
        * 24 features: Coordinates of three vertices and center points(12) + normal vectors of three vertices and center points(12)
    * Rearrange labels
    * Split datasets: Train : Validation: Test = 13 : 1 : 1
* Data Preprocessing: `Mesh_dataset2.py`
    * Load dataset
    * Reshape tensors: Data - Pointclouds(num_points, 3) / Label - Segmentation part labels(num_points,)
* Model Modification: `step2_training.py`
    * Parameters fine-tuning: `PointMLP.py`/`PointMLP_10k.py`
    * Training: -- optimizer SGD --lr 0.01 --scheduler 'step' --dropout 0.5 --k 20
    * Save best weights
* Evaluation: `losses_and_metrics_for_mesh.py`
    * Loss: Generalized_Dice_Loss()
    * Metrics: DSC, PPV, SEN
* Testing: `test_result.py`/`test_result_10k.py`
    * Read the data file path
    * Make model point clouds
    * Extract key features
    * Data loader
    * Import model
    * Load checkpoints
    * Evaluation: Accuracy, mIou, DSC, PPV, SEN
    * Visualization