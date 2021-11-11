# Depth-Aware Multi-Grid Deep homogrpahy Estimation with Contextual Correlation ([paper](https://arxiv.org/pdf/2107.02524.pdf))
<p align="center">Lang Nie*, Chunyu Lin*, Kang Liao*, Shuaicheng Liu`, Yao Zhao*</p>
<p align="center">* Institute of Information Science, Beijing Jiaotong University</p>
<p align="center">` School of Information and Communication Engineering, University of Electronic Science and Technology of China</p>

## Requirement
* python 3.6
* numpy 1.18.1
* tensorflow 1.13.1

## Dataset for training 
#### step 1
We use [UDIS-D](https://github.com/nie-lang/UnsupervisedDeepImageStitching) for training. Please download it.
#### step 2
We adopt a pretrained monocular depth estimation model to get the depth of 'input2' in the training set. Please download the results of depth estimation in [Google Drive](https://drive.google.com/file/d/1UTDIpNpl5te8RaO_Zt22bxYjNMLwl5ql/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/16u2qyYRw6ciMuZz9hrdLoA)(Extraction code: 1234). Then place the 'depth2' folder in the 'training' folder of UDIS-D. (Please refer to the paper for more details about the depth. )

## Training

## Testing

## Meta
NIE Lang -- nielang@bjtu.edu.cn
```
@misc{nie2021depthaware,
      title={Depth-Aware Multi-Grid Deep Homography Estimation with Contextual Correlation}, 
      author={Lang Nie and Chunyu Lin and Kang Liao and Shuaicheng Liu and Yao Zhao},
      year={2021},
      eprint={2107.02524},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
