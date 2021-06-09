# Multi-Grid Deep Homogarphy (paper)
**Multi-grid** deep homogarphy network in the scenes of **low overlap rates**. 

The official implement of "Depth-Aware Multi-Grid Deep Homography Estimation with Contextual Correlation" in TensorFlow.

## Qualitatively comparisons with the state-of-the-arts
Visual comparisons with other multi-grid homgraphy solutions on various scenes of varying degrees of parallax. 

In the following cases, we warp the target image (the second input image) to align with the reference image (the first input image). Then the reference image and the warped target image can be fused by setting the intensity of blue channel in reference image and that of red channel in warped target image to zero. In this manner, the non-overlapping regions are shown in orange and the misalignments in the overlapping regions would be highlighted in different color. Although the proposed method can not completely eliminate misalignments, the remained misalignments in our results are less than that of other methods.

![image](https://github.com/nie-lang/Multi-Grid-Deep-Homogarphy/blob/main/figures/real_comparison.jpg)

## Cross-dataset validation
We train our network in [UDIS-D](https://github.com/nie-lang/UnsupervisedDeepImageStitching) and test it in other datasets. Specifically, we collect the datasets from classic image stitching papers, where these datasets are captured from different scenes and contains different degrees of parallax. Even if it is tested on other datasets, our solution still has good alignment capabilities. Especially the overlap rates between input images in these examples are pretty low and most deep learning solutions could fail in these scenes.

![image](https://github.com/nie-lang/Multi-Grid-Deep-Homogarphy/blob/main/figures/cross_dataset.png)

## Dataset for training 
We use [UDIS-D](https://github.com/nie-lang/UnsupervisedDeepImageStitching) for training, which is a real-world unsupervised image stitching dataset.

## Training

## Testing

## Meta
NIE Lang -- nielang@bjtu.edu.cn
