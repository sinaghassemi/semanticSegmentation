# Semantic segmentation of remotely sensing images
This repository provides the codes needed to address the semantic segmentation problem over two publicly available aerial images.


-[Generating samples](#1-generating-samples-for-training/validation/test-sets)
-[Generating ](####-122-test)

# Prerequesties
- Computer with Linux
- [MATLAB](https://www.mathworks.com/)
- [PyTorch](https://pytorch.org/)
- NVIDIA GPU is highly recommended particularly to speed up the training.

# 1. Generating samples for training/validation/test sets
Two publicly available datasets are used in our experiments:

1. [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
2. [ISPRS Vaihingen data](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)

After requesting and downloading these datasets, the samples needed for training,validation or testing the network should be generated. These generated samples are stored in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file enabling fast reading during training.
The codes used to generate the datasets files are written in MATLAB and provided the following locations:
1. [To generate samples of INRIA Dataset](https://github.com/sinaghassemi/semanticSegmentation/blob/master/generatingSmples/GeneratingDataset_INRIA.m)
2. [To generate samples of Vaihingen Dataset](https://github.com/sinaghassemi/semanticSegmentation/blob/master/generatingSmples/GeneratingDataset_ISPRS.m)

## 1.1 Generating samples of INRIA dataset
INRIA dataset includes 5 cities for train and validation samples and another 5 cities for the test samples.
We invite readers interested in more details to refer to this [link](https://project.inria.fr/aerialimagelabeling/contest/).



### 1.1.1 Training and validation
The generate the training and validation samples we developed following MATLAB [code](https://github.com/sinaghassemi/semanticSegmentation/blob/master/generatingSmples/GeneratingDataset_INRIA.m) .

For cities in the first set (Austin, Chicago, Kitsap County, Western Tyrol, Vienna) , we use the first five images of each city for extracting validation samples and the rest for training samples.

In first line of the code, the path to inria images shoud be defined, next the variable 'set' should be set to 'train' and then city should be selected. 

```matlab
path = 'AerialImageDataset/test/'  ; %Path to the data 
set  = 'test'                      ; %Extracting samples for 'val' | 'train' | 'test' set 
city = 'bellingham';                                           
% train and val = {'austin','chicago','kitsap','tyrol-w','vienna'}
% test = {'bellingham','bloomington','innsbruck','sfo','tyrol-e'}
```
For each city first train and then validation samples should be extracted.
Note that mean and std of samples are compuited over training samples and also use for extracitng validation samples.
This justify the first line in code:
```matlab
clearvars -except patchMean patchSTD
```
The training and validation samples of each city are stored in a separate file.
Then this [code](https://github.com/sinaghassemi/semanticSegmentation/blob/master/generatingSmples/combiningCitiesToADataset_inria.m)  is used to combine the samples of all cities into a single file.

### 1.1.1 Test

To generate the test samples enabling the evaluation of network performance, there are two choices: using the validation images. using the test images.
Since the annotation of test images are not provided we can use validation images as test set to be able measure network performance. However, by using test images, the network outputs should sent to dataset provider for analysis.
By setting the flag 'set' to 'test' and using validation areas and setting the flag 'withAnnotation' to 1, test samples are generated from validation images.
While using test areas and setting the flag 'withAnnotation' to 0, test samples are generated from test images.

## 1.2 Generating samples of ISPRS Vaihigen dataset

### 1.2.1 Training and validation


### 1.2.2 Test


## Training/Testing the network
```bash
INRIA TRAINING
CUDA_VISIBLE_DEVICES=1 python main.py --fileNameData inria.h5 --experiment 1 --depth 50 --imageSize 360 --patchSize 256 --nChannelsIn 3 --nChannelsOut 2 --dataset inria
-------------------
INRIA TEST ON Val Set
CUDA_VISIBLE_DEVICES=1 python main.py --experiment 1 --depth 50  --nChannelsIn 3 --nChannelsOut 2 --dataset inria --testModule ex1_bestNet_valF1.pt --set val  --batchSize 4
-------------------------
INRIA TEST ON Test Set
CUDA_VISIBLE_DEVICES=1 python main.py --experiment 1 --depth 50  --nChannelsIn 3 --nChannelsOut 2 --dataset inria --testModule ex1_bestNet_valF1.pt --set test --batchSize 4
ISPRS TRAINING
CUDA_VISIBLE_DEVICES=1 python main.py --fileNameData vaihingen.h5 --experiment 2 --depth 50 --imageSize 364 --patchSize 256 --nChannelsIn 4 --nChannelsOut 6 --dataset isprs
-------------------
ISPRS TEST ON Val Set
CUDA_VISIBLE_DEVICES=1 python main.py --experiment 2 --depth 50  --nChannelsIn 4 --nChannelsOut 6 --dataset isprs --testModule ex2_bestNet_valF1.pt --set val --batchSize 4
-------------------
ISPRS TEST ON Test Set
CUDA_VISIBLE_DEVICES=1 python main.py --experiment 2 --depth 50  --nChannelsIn 4 --nChannelsOut 6 --dataset isprs --testModule ex2_bestNet_valF1.pt --set test  --batchSize 4

```



After the ".h5" files are generated, training the network can be proceeded. The correspondig files are written in the python it uses PyTorch.

For example, to train the network with depth of 152 layers in encoder, over inria datsets:

```bash
python main.py --fileNameData dataset_inria.h5 --experiment ex1 --depth 152 --batchSize 16 --imageSize 360 --patchSize 256 --nChannelsIn 3 --nChannelsOut 2 --dataset inria  
```
Or, the trained network can be evaluted over a test area:
```bash
python main.py --experiment ex1 --depth 152 --batchSize 8 --imageSize 360 --patchSize 256 --nChannelsIn 3 --nChannelsOut 2 --dataset inria --testModule nets/trainedNetwork.pt --set test
```

## Some Examples

The segmentation map over a tile from INRIA dataset:

**INRIA**
![Alt text](images/prediction_allClasses_isprs_vaihingen11.tif)





