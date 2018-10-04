# Semantic segmentation of remotely sensing images
This repository provides the codes to address the semantic segmentation problem over two publicly available aerial images.
The codes are divided into two folders: one is to generate samples and the other is to train or test the proposed network.

## Required software
- [MATLAB](https://www.mathworks.com/)
- [PyTorch](https://pytorch.org/)

## Generating samples for training/validation/test sets
Two publicly available datasets are used in our experiments:

1. [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
2. [ISPRS Vaihingen data](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)

After requesting and downloading these datasets, the samples needed for training,validation or testing the network should be generated. These generated samples are stored in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file enabling fast reading during training.
The codes used to generate the datasets files are written in MATLAB and provided the following locations:
1. [To generate samples of INRIA Dataset](https://github.com/sinaghassemi/semanticSegmentation/blob/master/generatingSmples/GeneratingDataset_INRIA.m)
2. [To generate samples of Vaihingen Dataset](https://github.com/sinaghassemi/semanticSegmentation/blob/master/generatingSmples/GeneratingDataset_ISPRS.m)

## Training/Testing the network

After the ".h5" files are generated, training the network can be proceeded. The correspondig files are written in the python it uses PyTorch.
