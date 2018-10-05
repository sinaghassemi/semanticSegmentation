# Copyright 2018 Telecom Italia S.p.A.

# Redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or other materials provided
# with the distribution.
# Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
# derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch 
import os
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from math import ceil
import h5py
import numpy as np
import argparse
from models import encoder_decoder_resnet
from loaders import loader
from functions import trainSegmentation, valSegmentation, logEpoch, reset_BN ,scores_test, stitchPatchesFull, savePredictions, crf
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Lock
import random

#=================Argument============================#
parser = argparse.ArgumentParser(description='PyTorch Satellite Segmentation')
parser.add_argument('--seed'		, default= 1		, type=int	, metavar='N', help='Torch and NumPy pseudorandom number generators seed (-1 disables manual seeding)')
parser.add_argument('--fileNameData'	, default=None		, type=str	, metavar='N', help='hdf5 file of the dataset, images file')
parser.add_argument('--preload'		, default='true'	, type=str	, metavar='N', help='set to true to preload entire dataset into memory')
parser.add_argument('--experiment'	, default='ex500'	, type=str	, metavar='N', help='Experiment Identifier')
parser.add_argument('--batchSize'	, default=8		, type=int	, metavar='N', help='batch size')
parser.add_argument('--imageSize'	, default=364		, type=int	, metavar='N', help='imageSize')
parser.add_argument('--patchSize'	, default=256		, type=int	, metavar='N', help='imageSize')
parser.add_argument('--nEpochs'		, default=300		, type=int	, metavar='N', help='total number of Epoches')
parser.add_argument('--nChannelsIn'	, default=4		, type=int	, metavar='N', help='number of input Channels')
parser.add_argument('--nChannelsOut'	, default=2		, type=int	, metavar='N', help='number of output Channels')
parser.add_argument('--nThreads'	, default=1		, type=int	, metavar='N', help='number of threads for data loading')
parser.add_argument('--depth'		, default=34		, type=int	, metavar='N', help='number of layers in encoder')
parser.add_argument('--optim'		, default='SGD'		, type=str	, metavar='N', help='optimizer [SGD | adagrad]')
parser.add_argument('--lr'		, default=1		, type=float	, metavar='N', help='learning rate strategy (SGD, >=1) or base value (Adagrad, any value)')
parser.add_argument('--lrDecay'		, default=1		, type=float	, metavar='N', help='learning rate decay (Adagrad)')
parser.add_argument('--weightDecay'	, default=1		, type=float	, metavar='N', help='weight rate decay (Adagrad)')
parser.add_argument('--GPU'		, default=True		, type=bool	, metavar='N', help='training over GPU')
parser.add_argument('--criterion'	, default='BCE'		, type=str	, metavar='N',choices=['2dNLL', 'BCE', 'MSE'], help='criterion')
parser.add_argument('--finetuneModule'	, default=None		, type=str	, metavar='N', help='network needed to be fine-tunned')
parser.add_argument('--testModule'	, default=None		, type=str	, metavar='N', help='running a testmodule over test areas')
parser.add_argument('--set'	  	, default='train'	, type=str	, metavar='N', choices=['train', 'val', 'test']	,help='set')
parser.add_argument('--dataset'	  	, default='isprs'	, type=str	, metavar='N', choices=['isprs','inria']	,help='dataset')
parser.add_argument('--crf'	 	, default=False  	, type=bool	, metavar='N', help='conditional random field')
parser.add_argument('--finetunedBN'	, default=False  	, type=bool	, metavar='N', help='fine tunning BN parameters  over test area')
parser.add_argument('--resetBN'		, default=False  	, type=bool	, metavar='N', help='reseting BN')
parser.add_argument('--BNFinetuningEpochs', default=1		, type=int	, metavar='N', help='BNFinetuningEpochs')
args = parser.parse_args()

if args.preload == 'true':
	preload = True
else:
	preload = False


#===================Log File===========================
logFileName = 'ex%s_report.txt' % (args.experiment)
print ('Report will be saved to ' + logFileName)


#============================= System level stuff ============================

# AF: this will seed numpy's RNG for all processes
if args.seed > -1:
	# AF: this will seed torch's RNG for the main process, each worker will have its torch seed set to base_seed + worker_id, where base_seed is a long generated by main process using its RNG.
	#http://pytorch.org/docs/master/data.html
	torch.manual_seed(args.seed)
	# Setting Python RNG seed
	random.seed(args.seed)
	# Setting numpy seed for main and worker processes
	numPySeed = torch.initial_seed() % (2^32)
	np.random.seed(numPySeed)
	print ("Main PID " + str(os.getpid()) + " torch RNG seed " + str(torch.initial_seed()) + " numpy RNG seed " + str(numPySeed))
	
	#https://github.com/pytorch/pytorch/pull/2893
	torch.backends.cudnn.deterministic = True
	print ("All RNGs seeds set to " + str(args.seed))
else:
	print ("main_process PID " +  str(os.getpid()) + " torch RNG seed " + str (torch.initial_seed()))

#AF: TODO this should be better explored
#torch.backends.cudnn.benchmark =  True
#torch.backends.cudnn.verbose =  True
#============================= Datasets =====================================
returnLabels = True
normalize = False
areaBasedNorm = False
imageSize = args.imageSize
print('Dataset : %s'%(args.dataset))
if args.dataset == 'isprs':
	if args.set == 'val':
		returnLabels = False
		normalize = True
		imageSize = 512
		areaNumbers = [11,15,28,30,34]#[11,15,28,30,34]
		areaList=['isprs_vaihingen'+str(i) for i in areaNumbers]
		pyList=[7,7,7,7,7]
		pxList=[5,5,5,5,4]
		syList=[2566,2565,2567,2563,2555]
		sxList=[1893,1919,1917,1934,1388]
		trList=[400,400,400,400,400]
	elif args.set == 'test':
		returnLabels = False
		normalize = True
		imageSize = 512
		areaNumbers = [2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38]
		areaList=['isprs_vaihingen'+str(i) for i in areaNumbers]
		pyList=[7,7,7,7,7,7,7,7,6,7,7,9,7,7,7,5,7]
		pxList=[6,5,5,5,5,5,5,5,5,5,5,5,5,5,4,7,10]
		syList=[2767,2557,2557,2557,2557,2575,2565,2565,2315,2546,2546,3313,2563,2555,2555,1884,2550]
		sxList=[2428,1887,1887,1887,1887,1922,1919,1919,1866,1903,1903,1917,1917,1980,1581,2805,3816]
		trList=[400 for i in range(len(pyList))]
elif args.dataset == 'inria':
	if args.set == 'train':
		areaBasedNorm = True
		normalize = True
	elif args.set == 'val':
		returnLabels = False
		normalize = True
		imageSize = 1024
		cities = ['austin','chicago','kitsap','tyrol-w','vienna']
		areaNumbers = [i+1 for i in range(5)]
		areaList=[c+str(a) for c in cities for a in areaNumbers]
		pyList=[6 for i in range(5*len(cities))]
		pxList=[6 for i in range(5*len(cities))]
		syList=[5000 for i in range(5*len(cities))]
		sxList=[5000 for i in range(5*len(cities))]
		trList=[800 for i in range(5*len(cities))]
	elif args.set == 'test':
		returnLabels = False
		normalize = True
		imageSize = 1024
		cities = ['bellingham','bloomington','innsbruck','sfo','tyrol-e'] 
		areaNumbers = [i+1 for i in range(36)]
		areaList=[c+str(a) for c in cities for a in areaNumbers]
		pyList=[6 for i in range(len(cities)*36)]
		pxList=[6 for i in range(len(cities)*36)]
		syList=[5000 for i in range(len(cities)*36)]
		sxList=[5000 for i in range(len(cities)*36)]
		trList=[800 for i in range(len(cities)*36)]


#=============================== Loss Function ==============================
global criterion
global labelFormat
if args.criterion == '2dNLL':
	labelFormat='long'
	lastLayer = 'logsoftmax'
	labels_oneHot = False
	criterion = nn.NLLLoss2d()

elif args.criterion == 'BCE':
	labelFormat='float'
	lastLayer = 'sigmoid'
	labels_oneHot = True
	criterion = nn.BCELoss()

elif args.criterion == 'MSE':
	labelFormat='float'
	lastLayer = 'sigmoid'
	labels_oneHot = True
	criterion = nn.MSELoss()


if args.GPU:
	criterion = criterion.cuda()
	
#===================== TRAIN AND VAL DATALOADER =====================#
# Mutex lock for concurrent HDF5 reading when nThreads > 1
if args.set == 'train':
	lockImg = Lock()
	lockLabel = Lock()
	train_dataset = loader(args.fileNameData, lockImg, lockLabel, dataset='train', aug=True, imageSize= imageSize, patchSize=args.patchSize, nChannelsIn=args.nChannelsIn,nChannelsOut=args.nChannelsOut, labelFormat=labelFormat, preload=preload,areaBasedNorm=areaBasedNorm,normalize=normalize,returnLabels=returnLabels)
	val_dataset = loader(args.fileNameData, lockImg, lockLabel, dataset='val', aug=False, imageSize=args.patchSize, patchSize=args.patchSize, nChannelsIn=args.nChannelsIn,nChannelsOut=args.nChannelsOut, labelFormat=labelFormat, preload=preload,areaBasedNorm=areaBasedNorm,normalize=normalize,returnLabels=returnLabels)
	NumberOfSamples_train = train_dataset.__len__()
	NumberOfSamples_val = val_dataset.__len__()
	print ('train size : %d' % (NumberOfSamples_train))
	print ('val size : %d' % (NumberOfSamples_val))
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batchSize,shuffle=True,num_workers=args.nThreads)
	val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=args.batchSize, shuffle=False,num_workers=args.nThreads)

#=====================CNN====================================
print('depth = '+ str(args.depth))
model = encoder_decoder_resnet(args.nChannelsIn,args.nChannelsOut,args.depth,lastLayer)
print(model)

if args.testModule != None:
	print('loadning trained module of %s' % (args.testModule))
	model.load_state_dict(torch.load(args.testModule))

if args.finetuneModule != None:
	print('loadning trained module of %s' % (args.finetuneModule))
	model.load_state_dict(torch.load(args.finetuneModule))

if args.GPU:
	model.cuda()
#==============================train and val ===================================
train = trainSegmentation
val = valSegmentation


#======================= Optimizer and learning rates ==========================
if args.optim == 'SGD':
	if args.lr < 1:
		print('ERROR optimizer SGD uses lr as strategy index')
		sys.exit(1)
	
	optimizer = torch.optim.SGD (model.parameters(), lr = 0.1, momentum=0.9)
	if args.lr == 1:
		lr_strategy = {'epoch':[1,100,300],'lr':[5e-2,5e-3,5e-4],'wd':[0,0,0]}
	elif args.lr == 2:
		lr_strategy = {'epoch':[1,30,60,100,150],'lr':[5e-2,1e-2,5e-3,1e-3,5e-4],'wd':[1e-3,2e-4,1e-4,2e-5,1e-5]}
	print('learning strategy (SGD)')
	print(lr_strategy)
	
elif args.optim == 'adagrad':
	#http://pytorch.org/docs/master/_modules/torch/optim/adagrad.html
	# clr = lr / (((iter-1) * lr_decay) +1)
	optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=args.lrDecay, weight_decay=args.weightDecay)
	print('learning rate (Agagrad) LR ' + str(args.lr) + ' lr_decay ' + str(args.lrDecay) + ' weight_decay ' + str(args.weightDecay))
	
elif args.optim == 'adam':
	#http://pytorch.org/docs/master/optim.html#torch.optim.Adam
	# clr = lr / ...
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightDecay)
	print('learning rate (Adam) LR ' + str(args.lr) + ' b1 0.9 b2 0.999 weight_decay ' + str(args.weightDecay))
	
else:
	print('ERROR optimizer ' + args.optim + ' unsupported')
	sys.exit(1)

#=================================== main ======================================
#---TRAINING---
if args.set == 'train':
	print('training')
	numOfIterPerEpoch_train = int(ceil(NumberOfSamples_train/float(args.batchSize)))
	numOfIterPerEpoch_val   = int(ceil(NumberOfSamples_val/float(args.batchSize)))
	loss_val_best = 100
	loss_train_best = 100
	f1_val_best = 0
	overallAcc_val_best = 0
	for epoch in range(args.nEpochs):
		#==================== Learning rate adaptation (SGD) =========================
		if args.optim == 'SGD':
			if epoch+1 in lr_strategy['epoch']:
				lr = lr_strategy['lr'][lr_strategy['epoch'].index(epoch+1)]
				wd = lr_strategy['wd'][lr_strategy['epoch'].index(epoch+1)]
				print ( 'Epoch : %d LR : %f WD : %f ' % (epoch+1,lr,wd))
				optimizer = torch.optim.SGD (model.parameters(),weight_decay=wd, lr = lr, momentum=0.9)				
		#======================TRAINIG==================================#
		loss_train, f1_train, aveAcc_train, overallAcc_train, f1_classes_train, acc_classes_train = train(train_loader, model, criterion, optimizer, epoch, args.nEpochs, numOfIterPerEpoch_train, args.nChannelsOut,args.GPU, labels_oneHot)
		
		#======================VALIDATION===============================#
		loss_val, f1_val, aveAcc_val, overallAcc_val, f1_classes_val, acc_classes_val = val(val_loader, model, criterion, epoch, args.nEpochs, numOfIterPerEpoch_val, args.nChannelsOut,args.GPU, labels_oneHot)
		
		#======================Writning on log file====================#
		logEpoch(logFileName, epoch, loss_train, loss_val, f1_train, f1_val, aveAcc_train, aveAcc_val, overallAcc_train, overallAcc_val , f1_classes_train, f1_classes_val, acc_classes_train, acc_classes_val)

		# =========Saving the network =================
		if f1_val > f1_val_best:
			f1_val_best = f1_val
			fileName = 'ex%s_bestNet_valF1.pt' % (args.experiment)
			print('Saving val best model as ' + fileName)
			torch.save(model.state_dict(), fileName)

		if overallAcc_val > overallAcc_val_best:
			overallAcc_val_best = overallAcc_val
			fileName = 'ex%s_bestNet_valAcc.pt' % (args.experiment)
			print('saving val best model as ' + fileName)
			torch.save(model.state_dict(), fileName)

#---TESTING---
else:

		print('testing pretrained network')
		confusion_summed = np.zeros((args.nChannelsOut,args.nChannelsOut))
		F1_summed = 0
		F1_perClass_summed = [0 for i in range(args.nChannelsOut)]
		accaracy_summed = 0
		overallAccuracy_summed = 0
		lockImg = Lock()
		lockLabel = Lock()


		#===================== Fine Tunning BN ================#
		if args.finetunedBN:
			model.train()
			print('========= Fine Tunning BN ==========')
			if args.resetBN == True:
				print('reseting computed mean and var of batch norms')
				model.apply(reset_BN)

			for te in range(args.BNFinetuningEpochs):
				for index,area in enumerate(areaList):
					print('====================================')
					print(area)
					#===================== DATALOADER =====================#
					dataFile  = ('data/%s_test.h5') % area
					dataset   = loader(dataFile,dataFile, lockImg, lockLabel, dataset='test', aug=False, imageSize=imageSize, patchSize=imageSize, nChannelsIn=args.nChannelsIn,nChannelsOut=args.nChannelsOut, labelFormat=labelFormat, preload=preload,returnLabels=returnLabels,normalize=normalize)
					NumberOfSamples = dataset.__len__()
					print ('test size : %d' % (NumberOfSamples))			
					print(dataFile)
					test_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=args.batchSize,shuffle=True)
					numOfIterPerEpoch = int(ceil(NumberOfSamples/float(args.batchSize)))
					for iteration, (data,_,_) in enumerate(test_loader):
						print ("Epoch[%d] Iter [%d/%d]" %(te,iteration+1, numOfIterPerEpoch))
						if args.GPU:
							data = data.cuda()
						with torch.no_grad():
							outputs = model(data)

		for index,area in enumerate(areaList):
			print('====================================')
			print(area)
			#===================== DATALOADER =====================#
			dataFile  = ('%s_test.h5') % area
			dataset   = loader(dataFile, lockImg, lockLabel, dataset='test', aug=False, imageSize=imageSize, patchSize=imageSize, nChannelsIn=args.nChannelsIn,nChannelsOut=args.nChannelsOut, labelFormat=labelFormat, preload=preload,returnLabels=returnLabels,normalize=normalize)
			NumberOfSamples = dataset.__len__()
			print ('test size : %d' % (NumberOfSamples))			
			print(dataFile)

			#===================== Evaluating     ================#
			if args.finetunedBN:
				model.eval()	
			else:
				model.train()
			print('========= Evaluating ==========')
			test_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=args.batchSize,shuffle=False)
			numOfIterPerEpoch = int(ceil(NumberOfSamples/float(args.batchSize)))
			outputs_all = []
			for iteration, (data,_) in enumerate(test_loader):
				print (" Iter [%d/%d]" %(iteration+1, numOfIterPerEpoch))
				if args.GPU:
					data = data.cuda()

				with torch.no_grad():
					outputs = model(data)
					
				outputs = outputs.data
				outputs = outputs.cpu()
				if iteration == 0:
					outputs_all = outputs
				else:
					outputs_all = torch.cat((outputs_all,outputs),dim=0)

			if lastLayer == 'logsoftmax':
				outputs_all = torch.exp(outputs_all) # probability
			if args.set == 'val':
				outputs_all = stitchPatchesFull(outputs_all,pyList[index],pxList[index],trList[index],syList[index],sxList[index])
				if args.crf:
					print('Applying CRF')	
					outputs_all = crf(outputs_all,areaNumbers[index])
				F1, F1_perClass, accaracy , overallAccuracy, confusion = scores_test(outputs_all,area,args.experiment)
				F1_summed += F1
				accaracy_summed += accaracy
				overallAccuracy_summed += overallAccuracy
				confusion_summed += confusion
				F1_perClass_summed = [F1_perClass_summed[c] + F1_perClass[c] for c in range(args.nChannelsOut)]
			if args.set =='test':
				outputs_all = stitchPatchesFull(outputs_all,pyList[index],pxList[index],trList[index],syList[index],sxList[index])
				savePredictions(outputs_all,areaNumbers[index % len(areaNumbers)],args.experiment,args.dataset,areaList[index])	
			print('====================================')
		if args.set == 'val':
			F1_summed = F1_summed/len(areaList)
			accaracy_summed = accaracy_summed/len(areaList)
			overallAccuracy_summed = overallAccuracy_summed/len(areaList)
			confusion_summed = confusion_summed/len(areaList)
			F1_perClass_summed = [F1_perClass_summed[c]/len(areaList) for c in range(args.nChannelsOut)]
			fileName = 'testImages/'+args.experiment+'/results_.txt'
			print('+++++++++++++++++++++++++++++++++++++++')
			print('Results averaged over all areas')
			print('F1Score %f' % F1_summed)
			print('overallAccuracy %f' % overallAccuracy_summed)
			print('+++++++++++++++++++++++++++++++++++++++')











	


