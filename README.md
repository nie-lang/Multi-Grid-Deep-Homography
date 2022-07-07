# Depth-Aware Multi-Grid Deep homogrpahy Estimation with Contextual Correlation ([paper](https://arxiv.org/pdf/2107.02524.pdf))
<p align="center">Lang Nie*, Chunyu Lin*, Kang Liao*, Shuaicheng Liu`, Yao Zhao*</p>
<p align="center">* Institute of Information Science, Beijing Jiaotong University</p>
<p align="center">` School of Information and Communication Engineering, University of Electronic Science and Technology of China</p>

![image](https://github.com/nie-lang/Multi-Grid-Deep-Homography/blob/main/network.jpg)
## Requirement
* python 3.6
* numpy 1.18.1
* tensorflow 1.13.1

## For pytorch users
The official codes are based on tensorflow. We also provide a simple pytorch implementation of CCL for pytorch users, please refer to https://github.com/nie-lang/Multi-Grid-Deep-Homography/blob/main/CCL_pytorch.py.

The pytorch version has not been strictly tested. If you encounter some problems, please feel free to concat me (nielang@bjtu.edu.cn).

## Dataset Preparation
#### step 1
We use [UDIS-D](https://github.com/nie-lang/UnsupervisedDeepImageStitching) for training. Please download it.
#### step 2
We adopt a pretrained monocular depth estimation model to get the depth of 'input2' in the training set. Please download the results of depth estimation in [Google Drive](https://drive.google.com/file/d/1UTDIpNpl5te8RaO_Zt22bxYjNMLwl5ql/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/16u2qyYRw6ciMuZz9hrdLoA)(Extraction code: 1234). Then place the 'depth2' folder in the 'training' folder of UDIS-D. (Please refer to the paper for more details about the depth. )

## For windows system
For windows OS users, you have to change '/' to '\\\\' in 'line 73 of Codes/utils.py'.

## Training
#### Step 1: Training without depth assistance
Modidy the 'Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, we set 'ITERATIONS' to 300,000.

Modify the weight of shape-preserved loss in 'Codes/train_H.py' by setting 'lam_mesh' to 0.

Then, start the training without depth assistance:
```
cd Codes/
python train_H.py
```
#### Step 2: Finetuning with depth assistance
Modidy the 'Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, we set 'ITERATIONS' to 500,000.

Modify the weight of shape-preserved loss in 'Codes/train_H.py' by setting 'lam_mesh' to 10.

Then, finetune the model with depth assistance:
```
python train_H.py
```

## Testing
#### Our pretrained model
Our pretrained homography model can be available at [Google Drive](https://drive.google.com/drive/folders/1UO0_rttHDANPXX4eY4vizV99spWcqNod?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1cvrPjAfqBozkmU5XiSiJzA)(Extraction code: 1234). And place it to 'Codes/checkpoints/' folder.
#### Testing with your own model
Modidy the 'Codes/constant.py'to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'Codes/inference.py'.

Run:
```
python inference.py
```



## Meta
NIE Lang -- nielang@bjtu.edu.cn
```
@ARTICLE{9605632,
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Depth-Aware Multi-Grid Deep Homography Estimation With Contextual Correlation}, 
  year={2022},
  volume={32},
  number={7},
  pages={4460-4472},
  doi={10.1109/TCSVT.2021.3125736}}
```
