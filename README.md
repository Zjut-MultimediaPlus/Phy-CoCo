# Phy-CoCo
The code of "Phy-CoCo: Physical Constraint-based Correlation Learning for Tropical Cyclone Intensity and Size Estimation" accepted by ECAI2024.

## Introduction

![***Phy-CoCo_framework***](https://github.com/Zjut-MultimediaPlus/Phy-CoCo/tree/main/image/framework.jpg)

Contribution:
1. We proposed CoM based on Centrally Expanded Pooling (CEP) to model the correlation between the extracted features and the estimated attributes, fully exploring task-specific features.
2. To facilitate cross-task interaction, we designed bidirectional physical constraints applied to the transformation of features of interrelated tasks using Multi-Domain Recurrent Convolutions (MDRC).
3. Extensive experiments are conducted on multi-modal TC datasets to demonstrate the superiority of Phy-CoCo over the state-of-the-art TC estimation methods. The results highlight that Phy-CoCo is effective for both TC MSW and RMW estimation.

## Requirements 
* python 3.8.8
* Pytorch 1.1.0
* CUDA 11.7
