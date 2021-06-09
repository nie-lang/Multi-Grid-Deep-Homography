# Multi-Grid Deep Homogarphy (paper)
Multi-grid deep homogarphy network in the scenes of low overlap rates. 

The official implement of "Depth-Aware Multi-Grid Deep Homography Estimation with Contextual Correlation" in TensorFlow.

## Qualitatively comparisons with the state-of-the-arts
Visual comparisons with other multi-grid homgraphy solutions on various scenes of varying degrees of parallax. 

In the following cases, we warp the target image (the second input image) to align with the reference image (the first input image). Then the reference image and the warped target image can be fused by setting the intensity of blue channel in reference image and that of red channel in warped target image to zero. In this manner, the non-overlapping regions are shown in orange and the misalignments in the overlapping regions would be highlighted in different color. Although the proposed method can not completely eliminate misalignments, the remained misalignments in our results are less than that of other methods.

![image](https://github.com/nie-lang/Multi-Grid-Deep-Homogarphy/blob/main/figures/real_comparison.jpg)

## Cross-dataset validation

![image](https://github.com/nie-lang/Multi-Grid-Deep-Homogarphy/blob/main/figures/cross_dataset.png)

## Meta
NIE Lang -- nielang@bjtu.edu.cn
