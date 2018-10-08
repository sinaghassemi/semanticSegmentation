# Semantic segmentation of remotely sensing images
This repository contains the codes which address the semantic segmentation on remotely sensing images and over two publicly available aerial images:

1. [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
2. [ISPRS Vaihingen Dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)

To be able to train and test the proposed network, first training, validation and test samples should be extracted from these datasets. Then the extracted samples must stored in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file which enables fast data reading during training. The instruction required to generate the samples are provided in the following section:
- [Generating samples](#1-generating-training,-validation-and-test-samples)

Next, after the HDF5 files are prepared, we can proceed to train the proposed network using instructions provided at the following section:
- [Training](#2-training-the-network)

At the end, when the training course has been completed, the trained network can be deployed over test areas using the instruction in this following section:
- [Testing](#3-testing-the-trained-network)

# Prerequesties
- Computer with Linux
- [MATLAB](https://www.mathworks.com/)
- [PyTorch](https://pytorch.org/)
- NVIDIA GPU is highly recommended to speed up the training.

# 1. Generating training, validation and test samples

The samples required for training, validation or testing the network should be generated and stored in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file enabling fast reading during training.
The codes used to generate the dataset files are written in MATLAB and provided each dataset.


## 1.2 Training and validation samples (INRIA datset)
Training and validation samples are generated over five cities of Austin, Chicago, Kitsap County, Western Tyrol
and Vienna using the MATLAB [code](https://github.com/sinaghassemi/semanticSegmentation/blob/master/generatingSmples/GeneratingDataset_INRIA.m).

Each city includes 36 images sized 5000×5000 which covers a surface of 1500 m × 1500 m at the 30 cm resolution.
Based on what dataset providers recommended, we use the first five images of each city for extracting validation samples and the other images for training samples.

In first lines of the code, in the configuration section, there are variables that should be set to extarcted training, validation and test samples.  

```matlab
%% CONFIGURATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = '/AerialImageDataset/train/';     % Path to the data 
set  = 'train';                          % set flag can be set to 'val' | 'train' | 'test' to generate the corresponding samples

% train and val = {'austin','chicago','kitsap','tyrol-w','vienna'}
% test          = {'bellingham','bloomington','innsbruck','sfo','tyrol-e'}

city = 'bellingham';                % can be selected from the list above
hdf5Filename = strcat(city,'.h5');  % The h5 file in which samples will be stored         
datatype_patch = 'uint8';           % The data format 
datatype_label = 'uint8';           % The data format 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```
For each city use for training, first train and then validation samples are extracted.
In the first run ```set  = 'train'```, training samples are generated in each city.
Then in the second run ```set  = 'val'```  validation samples are extracted from the same city.
These samples are then stored in a '.h5' file named after the city.

After training and validation samples are extracted from each of the five cities and in five separate files,
These files are combined together and stored in a single file using a MATLBA [code](https://github.com/sinaghassemi/semanticSegmentation/blob/master/generatingSmples/combiningCitiesToADataset_inria.m).


## 1.2 Test samples (INRIA datset)

To generate the test samples enabling the evaluation of network performance, there are two choices: 1. using the validation images. 2. using the test images.
Since the annotation of test images are not provided we can use validation images as test set to be able measure network performance. However, by using test images, and after the network output uploaded to the website, the corresponding analysis are given within few weeks by dataset providers.

To generate test samples from validation images, 'set' should be set to 'test' ```set  = 'test'```, city can be chosen from the first set that provided with annotation ```city = 'austin' %{'austin','chicago','kitsap','tyrol-w','vienna'}```, and in the section "Configurations for each set" and the test configurations, ```areas=[1,2,3,4,5]``` and ```withAnnotation = 1```.

To generate test samples from test images, 'set' should be set to 'test' ```set  = 'test'```, city can be chosen from the second set which are not provided with annotation ```city = 'bellingham' %{'bellingham','bloomington','innsbruck','sfo','tyrol-e'}```, and in the section "Configurations for each set" and the test configurations, ```areas=[1,2,...,36]``` and ```withAnnotation = 0```.


## 1.2 Generating samples of ISPRS Vaihigen dataset

### 1.2.1 Training and validation


### 1.2.2 Test


# 2. Training the network
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


# 3. Testing the trained network


## Some Examples

The segmentation map over a tile from INRIA dataset:

**INRIA**
![Alt text](images/prediction_allClasses_isprs_vaihingen11.tif)





