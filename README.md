# MICCAI2023_CCFV

<img src="Figs/Framework.png" width = "980" height = "480" alt="" align=center />

# Introduction

Official implementation of MICCAI 2023 paper "Pick The Best Pre-trained Model: Towards Transferability Estimation For Medical Image Segmentation"(**early accepted, 14%**)


# Preparation
## Environment Setup
    python=3.7
    scipy=1.7.3
    numpy=1.21.6
    monai=1.0.1
    torch=1.13.0

## Dataset Preparation
1. Download the MSD Dataset from [here](http://medicaldecathlon.com/)
2. Preprocess the data using [nnUNet](https://github.com/MIC-DKFZ/nnUNet)

## Checkpoint Download
You can download our pretrained checkpoint of unet
and unetr from [Baidu Netdisk](https://pan.baidu.com/s/1EU0CzI2XnvsfHj84Q7gzTw) or [Google Drive](https://drive.google.com/file/d/1TFQla-ByBt3JpbiVQgcMwXqTGQJ05r2H/view?usp=drive_link)

Code:v3p5

## Usage


To evaluate checkpoints pretrained on your own dataset, please change the layers in configs to match your model. And make sure the configs match your training setting. You can also use the ccfv based on features extracted by your own methods.

## Example
    bash run.sh

