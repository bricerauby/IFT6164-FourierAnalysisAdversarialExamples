# IFT6164-FourierAnalysisAdversarialExamples

## Results Replication 
* For basic natural training  use 
`python basic_train.py` from [1]

* For adversarial training  use 
`python pgd_adversarial_training.py` from [1]

* use `python pgd_adversarial_training_low_freq_random_<0125,025,05,075>.py` for low frequency adverasrial training 

* use `python robustness_analysis.py --checkpoint_path <PATH TO CHECKPOINT> --norm_fourierHM 4 ` to compute fourier heat map (with norm 4), accuracy and pgd maps on CIFAR-10. (see docstring functino for more precise acknoledgment but pgd attack is from [2])  
* Use `python robustness_analysis_filter.py --checkpoint_path <PATH TO CHECKPOINT> --norm_fourierHM 4 ` to run the robustness analysis on CIFAR-10 with only the low pass filter. (see function docstrings for more precise acknoledgments but pgd attack is from [2])

* use `python train_resnet_fast.py` for fast resnet training and `python robustness_analysis_resnet_fast.py --checkpoint_path <PATH TO CHECKPOINT> --norm_fourierHM 4 ` for its robustness analysis.

## Acknowledgment
[1] Training script for resnet : https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR

[2] For the pgd attack computation : https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb 

[3] For the fast training resnet : https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min 
