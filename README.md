# Construct Dynamic Graphs for Hand Gesture Recognition via Spatial-Temporal Attention (BMVC 2019)

This repository holds the Pytorch implementation of [Construct Dynamic Graphs for Hand Gesture Recognition via Spatial-Temporal Attention](https://arxiv.org/abs/1904.03345) by Yuxiao Chen, Long Zhao, Xi Peng, Jianbo Yuan, and Dimitris N. Metaxas. If you find our code useful in your research, please consider citing:

```
@inproceedings{zhaoCVPR19semantic,
  author    = {Chen, Yuxiao and Zhao, Long and Peng, Xi and Yuan, Jianbo and Metaxas, Dimitris N.},
  title     = {Semantic Graph Convolutional Networks for 3D Human Pose Regression},
  booktitle = {BMVC},
  year      = {2019}
}
```

<p align="center"><img src="example.gif" width="70%" alt="" /></p>

## Introduction

We propose a Dynamic Graph-Based Spatial-Temporal Attention (DG-STA) method for hand gesture recognition. The key idea is to first construct a fully-connected graph from a hand skeleton, where the node features and edges are then automatically learned via a self-attention mechanism that performs in both spatial and temporal domains. The code of training our approach for skeleton-based hand gesture recognition on the [DHG-14/28 Dataset](http://www-rech.telecom-lille.fr/DHGdataset/) and the [SHREC’17 Track Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/) is provided in this repository.
<p align="center"><img src="figures/fig1.pdf" alt="" width="600"></p>
