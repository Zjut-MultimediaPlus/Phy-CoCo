# Phy-CoCo
The code of "Phy-CoCo: Physical Constraint-based Correlation Learning for Tropical Cyclone Intensity and Size Estimation" accepted by ECAI2024.

## Introduction

![***Phy-CoCo_framework***](https://github.com/Zjut-MultimediaPlus/Phy-CoCo/blob/main/figs/phy-coco.png)

Contribution:
1. We proposed CoM based on Centrally Expanded Pooling (CEP) to model the correlation between the extracted features and the estimated attributes, fully exploring task-specific features.
2. To facilitate cross-task interaction, we designed bidirectional physical constraints applied to the transformation of features of interrelated tasks using Multi-Domain Recurrent Convolutions (MDRC).
3. Extensive experiments are conducted on multi-modal TC datasets to demonstrate the superiority of Phy-CoCo over the state-of-the-art TC estimation methods. The results highlight that Phy-CoCo is effective for both TC MSW and RMW estimation.

## Requirements 
* python 3.8.8
* Pytorch 1.1.0
* CUDA 11.7
## Dataset
通过网盘分享的文件：phycoco数据集
链接: https://pan.baidu.com/s/11qnFtYeErHXH0VYNO7HRsA 提取码: coco 
--来自百度网盘超级会员v1的分享
