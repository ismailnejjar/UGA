# UGA
Code for our paper Uncertainty Guided Alignment with Deep Evidential Learning for Unsupervised Domain Adaptation in Regression


## Prerequisites
- Python3
- Numpy
- PyTorch == 1.12.1 (with CUDA and CuDNN (cu113))
- torchvision == 0.13.1
- PIL
- scikit-learn

Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment.

```python
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate uga 
```

## Train and Test EDAR model
The program can be run with the default parameters for feature alignement with C-Mixup using the following:

```python
#Train for dSprites
sh uga.sh
```
Code was tested on a RTX 3090.


## Data links
dSprites can be downloaded from: https://drive.google.com/drive/folders/1HBZgMxf_KgbIench770SG_ii4PgxPkO0

The files should be unziped and the folder for color, scream and noise should be inside the code folder.
