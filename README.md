# SEffiNet

The code is associated with the following paper: S. Liu, J. Ren, Z. Chen, et al., SEffiDiag: a Super Efficient Framework for Breast Cancer Diagnosis in Multi-Gigapixel Whole Slide Images, which is currently in its peer-review process.

## 1.Installation
-	Install PyTorch (1.4.0).
-	Install numpy (1.17.2).
-	Install torchstat (0.0.7).

## 2.Datasets
For our breast cancer diagnosis work, use Camelyon16 benchmark. The training data consists of 270 WSIs (160 normal and 110 tumor) with pixel-level annotations. The test data consists of 129 WSIs. 22 and 27 of them contain macro and micro tumors, respectively. 
Currently our experiments are carried out on the Camelyon16 benchmark (see https://camelyon16.grand-challenge.org/ for downloading).

## 3.Training 
Since our work involves patch-based classification model generation stage, we directly extract massive image patches according to the coordinates released by NCRF[1](see https://github.com/baidu-research/NCRF), which is a public resource associated with Camelyon16. It contains the coordinates of over 400k representative level-0 patches. We further employed different data augmentation methods like rotation, flipping, and color jittering. SGD with a momentum of 0.9 is used to optimize the network and the learning rate is fixed to 0.001. We conduct all the experiments using PyTorch on a workstation with one NVIDIA TITAN Xp GPU.

We have provided our SEffiNet.ckpt in the checkpoint forlder. 
You can download it for further experiments.

## 4.Evaluation
To evaluate the model performace:

'Python SEffiNet.py'

You will see the memory, MAdd, Flops, MemR+W of each layer and total.

## 5.Performance
Total params: 65,498

Total memory: 13.89MB

Total MAdd: 102.03M

Total Flops: 52.56M

Total MemR+W: 27.74MB

## Reference
[1] Li, Y., Ping, W.: Cancer metastasis detection with neural conditional random field. arXiv preprint arXiv:1806.07064 (2018)

