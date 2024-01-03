# DSAF-Net
Official PyTorch implementation of Defocus and Similarity Attention Fusion Net (DSAF-Net) in paper [A Defocus and Similarity Attention-Based Cascaded Network for Multi-Focus and Misaligned Image Fusion](https://doi.org/10.1016/j.inffus.2023.102125), which is an end-to-end network to merge information from several multi-focus and misaligned images in the focal stack to obtain a clear and detailed image.
## Installation
* Download the repository: `git clone https://github.com/PeimingCHEN/DSAF-Net`.<br>
* Install Python 3.7.10, Pytorch 1.8.1, and CUDA 11.1.
* Install the dependencies: `pip install -r requirement.txt`.<br>
## Overall Framework
<img src="https://github.com/PeimingCHEN/DSAF-Net/blob/main/models/Overall%20Framework.png" width="802"/><br/>
## Dataset
You can find the WHU-MFM dataset [here](https://github.com/PeimingCHEN/WHU-MFM-Dataset). Please upload the training set to the `dataset/` folder and the testing set to the `test/` folder. Each sample consists of a focal stack with 5 images. We adopt the 480 × 360 version of WHU-MFM and shuffle it in the unit of the focal stack to train and test DSAF-Net.
## Training and Testing
* To train the Defocus-Net: `python train.py`.<br>
* To train the OpticalFlow-Net: `python train_raft.py`.<br>
* To joint training and test the Fusion-Net: `python train_fusion.py`.<br>
A single GPU is enough to carry out it.<br>
## Results
More fusion results on WHU-MFM testing dataset are shown in the `results/` folder.<br>
<img src="https://github.com/PeimingCHEN/DSAF-Net/blob/main/results/result.png"/><br/>
