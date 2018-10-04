# Semantic segmentation of remotely sensing images
This repository provides the codes to address the semantic segmentation problem over two publicly available aerial images.
The codes are divided into two folders: one is to generate samples and the other is to train or test the proposed network.

## Required software
- MATLAB
- PyTorch

## Generating samples for training/validation/test sets
First, you need to generate the samples needed for training,validation or testing the network.
-[To generate samples of INRIA Dataset](semanticSegmentation/generatingSmples/GeneratingDataset_INRIA.m)
-[To generate samples of Vaihingen Dataset](semanticSegmentation/generatingSmples/GeneratingDataset_ISPRS.m)

## Training/Testing the network
