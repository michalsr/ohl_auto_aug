# Online Hyper-parameter Learning for Auto-Augmentation
Implementation of CIFAR-10 experiments in [Online Hyper-parameter Learning for Auto-Augmentation Strategy] (https://arxiv.org/abs/1905.07373) from ICCV 2019
# Dependencies 
- PyTorch 1.8
- TorchVision 

# Code
To run the CIFAR-10 experiments:
````python main.py --output_dir {OUTPUT directory} --prefix {directory before main directoroy name} --auto_aug {boolean for whether to use normal augmentations or auto-augmentations} --epoch {number of total epochs} --trajec {number of trajectories}````
