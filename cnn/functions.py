import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from scipy.misc import imsave,imread
import os

#======================= Resetting Batch Norms ======================
def reset_BN(m):
	if type(m) == nn.BatchNorm2d:
		m.running_mean.zero_()
		m.running_var.fill_(1)
		m.momentum = 0.9
#=======================  Batch Norms Params ======================
def getBNparams(net):
	params_list=[]
	for module in net.modules():
		if type(module) == nn.BatchNorm2d:
			print('getting params of %s' % module)
			params_list.append({'params':module.parameters()})
	return params_list
#======================= Division ======================
def div(x,y):
	'''Division ,returning zeros for zero denominator'''
	if y == 0:
		return 0
	return x / y

#======================= Logging one epoch to report file ======================
def logEpoch(logFileName, epoch, loss_train, loss_val, f1_train, f1_val, aveAcc_train, aveAcc_val, overallAcc_train, overallAcc_val, f1_classes_train, f1_classes_val, acc_classes_train, acc_classes_val):
	'''Saving the perfromance metrics over a text log file'''
	logFile  = open(logFileName, 'a')
	logFile.write("Epoch %d trainLoss %2.6f valLoss %2.6f train_f1 %2.6f val_f1 %2.6f train_AveAcc %2.6f val_AveAcc %2.6f train_OAAcc %2.6f val_OAAcc %2.6f " % (epoch+1, loss_train, loss_val, f1_train, f1_val, aveAcc_train, aveAcc_val, overallAcc_train, overallAcc_val))
	for i in range(len(f1_classes_train)):
		logFile.write("F1_train_class_%d %2.6f "  % (i,f1_classes_train[i]) )
		logFile.write("Acc_train_class_%d %2.6f " % (i,acc_classes_train[i]))
		logFile.write("F1_val_class_%d %2.6f "    % (i,f1_classes_val[i])   )
		logFile.write("Acc_val_class_%d %2.6f " % (i,acc_classes_val[i])  )
	logFile.write("\n")
	logFile.close()
#============================== Minibatch training =============================
def trainSegmentation(train_loader, model, criterion, optimizer, epoch, n_Epochs, numOfIterPerEpoch_train, nClasses, GPU, oneHot):
	'''Function to train and measure network performance during the training'''
	model.train()
	loss = 0
	f1 = 0
	aveAcc = 0
	overallAcc = 0
	f1_classes  = [0 for i in range(nClasses)]
	acc_classes = [0 for i in range(nClasses)]
	start = timer()
	for iteration, (data, labels) in enumerate(train_loader):
		data = Variable(data)
		labels = Variable(labels)
		if GPU:
			data = data.cuda()
			labels = labels.cuda()
		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = model(data)
		loss_minibatch = criterion(outputs, labels)

		f1_minibatch, aveAcc_minibatch, overallAcc_minibatch, f1_classes_minibatch, acc_classes_minibatch = computeScore(outputs, labels ,oneHot)
		loss = loss + loss_minibatch.item()
		f1  = f1  + f1_minibatch
		aveAcc = aveAcc + aveAcc_minibatch
		overallAcc = overallAcc + overallAcc_minibatch
		f1_classes  = [f1_classes[i]  + f1_classes_minibatch[i] for i in range(nClasses)]
		acc_classes = [acc_classes[i] + acc_classes_minibatch[i] for i in range(nClasses)]
		loss_minibatch.backward()
		optimizer.step()
		print ("Epoch [%d/%d] Iter [%d/%d] Loss %.4f F1Score %.4f Ave Accuracy %.4f Overall Accuracy %.4f Time[s] %.3f" %(epoch+1, n_Epochs, iteration+1, numOfIterPerEpoch_train, loss_minibatch.item(), f1_minibatch, aveAcc_minibatch, overallAcc_minibatch, timer()-start))
		start = timer()
	loss = loss / float(numOfIterPerEpoch_train)
	f1 = f1 / float(numOfIterPerEpoch_train)
	aveAcc = aveAcc / float(numOfIterPerEpoch_train)
	overallAcc = overallAcc / float(numOfIterPerEpoch_train)
	f1_classes  =  [f1_classes[i] / float(numOfIterPerEpoch_train) for i in range(nClasses)]
	acc_classes  = [acc_classes[i] / float(numOfIterPerEpoch_train) for i in range(nClasses)]
	print('Loss  on train : %f ' % ( loss))
	return loss,f1,aveAcc,overallAcc,f1_classes,acc_classes

#============================= Minibatch validation ============================
def valSegmentation(val_loader, model, criterion, epoch, n_Epochs, numOfIterPerEpoch_val, nClasses, GPU, oneHot):
	'''Function to validate and measure network performance during the validation'''
	model.eval()
	loss = 0
	f1 = 0
	aveAcc = 0
	overallAcc = 0
	f1_classes  = [0 for i in range(nClasses)]
	acc_classes = [0 for i in range(nClasses)]  
	for iteration, (data, labels) in enumerate(val_loader):
		#data = Variable(data, volatile=True)
		#labels = Variable(labels, volatile=True)
		if GPU:
			data = data.cuda()
			labels = labels.cuda()

		with torch.no_grad():	
			outputs = model(data)

		loss_minibatch = criterion(outputs, labels)

		f1_minibatch, aveAcc_minibatch, overallAcc_minibatch, f1_classes_minibatch, acc_classes_minibatch = computeScore(outputs, labels, oneHot)
		loss = loss + loss_minibatch.item()
		f1  = f1  + f1_minibatch
		aveAcc = aveAcc + aveAcc_minibatch
		overallAcc = overallAcc + overallAcc_minibatch
		f1_classes  = [f1_classes[i]  + f1_classes_minibatch[i] for i in range(nClasses)]
		acc_classes = [acc_classes[i] + acc_classes_minibatch[i] for i in range(nClasses)]
		print ("Epoch [%d/%d] Iter [%d/%d] Loss %.4f F1Score %.4f Ave Accuracy %.4f Overall Accuracy %.4f" %(epoch+1, n_Epochs, iteration+1, numOfIterPerEpoch_val, loss_minibatch.item(), f1_minibatch, aveAcc_minibatch, overallAcc_minibatch))
	loss = loss / float(numOfIterPerEpoch_val)
	f1 = f1 / float(numOfIterPerEpoch_val)
	aveAcc = aveAcc / float(numOfIterPerEpoch_val)
	overallAcc = overallAcc / float(numOfIterPerEpoch_val)
	f1_classes  =  [f1_classes[i] / float(numOfIterPerEpoch_val) for i in range(nClasses)]
	acc_classes  = [acc_classes[i] / float(numOfIterPerEpoch_val) for i in range(nClasses)]
	print('Loss  on val : %f' % loss)
	return loss,f1,aveAcc,overallAcc,f1_classes,acc_classes

#==================== Computing Score in validation Areas ======================
def computeScore(outputs,labels,oneHot):
	'''Function to measure network performance on the test set'''
	num_classes = outputs.size(1)
	num_images = outputs.size(0)
	overallAccuracy_summed = 0
	precision_summed= 0 
	recall_summed = 0 
	accuracy_summed = 0
	IoU_summed   = 0
	F1_summed  = 0
	accuracyPerClass = [0 for i in range(num_classes)]
	F1PerClass = [0 for i in range(num_classes)]

	for img in range(num_images):
		tp_summed = 0
		output = outputs.data[img].squeeze()
		label  = labels.data[img].squeeze()


		_ , output = torch.max(output,0)
		output = output.squeeze()
		for cls in range(num_classes):
			if oneHot:
				mask  = label[cls].squeeze()
			else:
				mask = torch.eq(label,cls)
			prediction = torch.eq(output,cls)
			truePositive = torch.eq(prediction.byte(),mask.byte()) * torch.eq(prediction.byte(),1)
			tp = torch.sum(truePositive).item()
			falsePositive = torch.gt(prediction.byte(),mask.byte())
			fp = torch.sum(falsePositive).item()
			falseNegative = torch.lt(prediction.byte(),mask.byte())
			fn = torch.sum(falseNegative).item()
			trueNegative = torch.eq(prediction.byte(),mask.byte()) * torch.eq(prediction.byte(),0)
			tn = torch.sum(trueNegative).item()

			precision 	= div(tp, tp+fp)
			recall 		= div(tp, tp+fn)
			IoU 		= div(tp, tp+fn+fp)
			F1 		= div(2 * precision * recall, precision + recall)
			accuracy 	= div(tp+tn, tp+tn+fn+fp);
			tp_summed	= tp_summed + tp
		
			precision_summed = precision_summed + precision 
			recall_summed 	 = recall_summed + recall 
			accuracy_summed	 = accuracy_summed + accuracy
			IoU_summed 	 = IoU_summed + IoU
			F1_summed 	 = F1_summed + F1

			accuracyPerClass[cls] 	= accuracyPerClass[cls] + accuracy
			F1PerClass[cls] 	= F1PerClass[cls] + F1
		overallAccuracy = tp_summed / (output.size(0) * output.size(1))
		overallAccuracy_summed	= overallAccuracy_summed + overallAccuracy


	overallAccuracy 	= overallAccuracy_summed / float(num_images)
	accuracyPerClass	= [accuracyPerClass[i]/ float(num_images) for i in range(len(accuracyPerClass))] 
	F1PerClass 		= [F1PerClass[i]/ float(num_images) for i in range(len(F1PerClass))] 
	precision	 	= precision_summed / float(num_classes * num_images)
	recall			= recall_summed / float(num_classes * num_images)
	averagedAccuracy	= accuracy_summed / float(num_classes * num_images)
	IoU	 		= IoU_summed / float(num_classes * num_images)
	F1			= F1_summed / float(num_classes * num_images)
	
	return F1,averagedAccuracy,overallAccuracy,F1PerClass,accuracyPerClass
#=================================================================================
def stitchPatchesFull(networkOut_allClasses,num_patches_x,num_patches_y,transposiotn,size_x,size_y):
	print('stitching patches')
	nClasses = networkOut_allClasses.size(1)
	patchSize =  networkOut_allClasses.size(2)
	output_allClasses = torch.Tensor(nClasses,size_x,size_y)
	for cls in range(nClasses):
		print('class: '+str(cls))
		networkOut = networkOut_allClasses[:,cls].contiguous()
		networkOut = networkOut.view(num_patches_x,num_patches_y,networkOut.size(1),networkOut.size(2))
		remained_x = size_x - (((networkOut.size(0)-2) * transposiotn) + patchSize)
		remained_y = size_y - (((networkOut.size(1)-2) * transposiotn) + patchSize)
		shifted_x = transposiotn - remained_x 	
		shifted_y = transposiotn - remained_y
		output = torch.Tensor(size_x,size_y)

		a_1  = torch.Tensor(transposiotn,patchSize - transposiotn)
		d_1  = torch.Tensor(transposiotn,patchSize - transposiotn)
		for i in range(a_1.size(0)):
			a_1[i]  = torch.range(1,patchSize-transposiotn)/(patchSize-transposiotn)
			d_1[i] = 1 - torch.range(1,patchSize-transposiotn)/(patchSize-transposiotn) 

		a = torch.Tensor(2*transposiotn - patchSize,patchSize - transposiotn)
		d = torch.Tensor(2*transposiotn - patchSize,patchSize - transposiotn)
		for i in range(a.size(0)):
			a[i]  = torch.range(1,patchSize-transposiotn)/(patchSize-transposiotn)
			d[i] = 1 - torch.range(1,patchSize-transposiotn)/(patchSize-transposiotn) 
		

		a_2d  = torch.Tensor(patchSize - transposiotn,patchSize - transposiotn)
		d_2d  = torch.Tensor(patchSize - transposiotn,patchSize - transposiotn)
		for i in range(a_2d.size(0)):
			for j in range(a_2d.size(1)):
				d_2d[i][j] = (d_2d.size(0) + d_2d.size(0) - i - j)/(d_2d.size(0) + d_2d.size(0) - 2)
				a_2d[i][j] = 1 - d_2d[i][j]

		for i in range(networkOut.size(0)):
			for j in range(networkOut.size(1)):

				#==========Unshared Areas=====================	
				os_1 = patchSize + (i-1)*transposiotn 
				oe_1 = (i+1)*transposiotn
				os_2 = patchSize + (j-1)*transposiotn
				oe_2 = (j+1)*transposiotn 
				ns_1 = patchSize - transposiotn
				ne_1 = transposiotn 
				ns_2 = patchSize - transposiotn
				ne_2 = transposiotn 
					  
				if i == 0:
					os_1 = 0
					oe_1 = transposiotn
					ns_1 = 0
					ne_1 = transposiotn			
		
				if j == 0:
					os_2 = 0
					oe_2 = transposiotn
					ns_2 = 0
					ne_2 = transposiotn			

				if i == networkOut.size(0)-1:
					oe_1 = size_x
					ns_1 = patchSize - remained_x #+ 1
					ne_1 = patchSize 	


				if j == networkOut.size(1)-1:
					oe_2 = size_y
					ns_2 = patchSize  - remained_y #+ 1 
					ne_2 = patchSize 	
				output[os_1:oe_1,os_2:oe_2] = networkOut[i,j,ns_1:ne_1,ns_2:ne_2]

				#==Vertical Shared Areas
				if i < networkOut.size(0)-2 :
					if j == 0:
						output[(i+1)*transposiotn:i*transposiotn + patchSize,:transposiotn] = networkOut[i, j, transposiotn:patchSize, :transposiotn] * d_1.transpose(0,1) +  networkOut[i + 1, j, :patchSize-transposiotn, :transposiotn] * a_1.transpose(0,1)
					elif j == networkOut.size(1)-1:
						output[(i+1)*transposiotn:i*transposiotn + patchSize,patchSize + (j-1)*transposiotn:size_y] = networkOut[i, j, transposiotn:patchSize, patchSize - remained_y:patchSize] * d_1[:remained_y,:patchSize - transposiotn].transpose(0,1) +  networkOut[i+1,j,:patchSize-transposiotn,patchSize - remained_y:patchSize] * a_1[:remained_y,:patchSize - transposiotn].transpose(0,1)
					else:
						output[(i+1)*transposiotn : i*transposiotn + patchSize,patchSize + (j-1)*transposiotn : (j+1)*transposiotn] = networkOut[i,j, transposiotn:patchSize, patchSize-transposiotn:transposiotn] * d.transpose(0,1) + networkOut[i+1 ,j,:patchSize-transposiotn, patchSize -transposiotn : transposiotn] * a.transpose(0,1)


				#==Vertical Shared Areas last
				if i == networkOut.size(0)-2 :
					if j == 0:
						output[(i+1)*transposiotn:i*transposiotn + patchSize,:transposiotn] = networkOut[i, j, transposiotn:patchSize, :transposiotn] * d_1.transpose(0,1) +  networkOut[i + 1, j, shifted_x:patchSize-transposiotn+shifted_x, :transposiotn] * a_1.transpose(0,1)
					elif j == networkOut.size(1)-1:
						output[(i+1)*transposiotn:i*transposiotn + patchSize,patchSize + (j-1)*transposiotn:size_y] = networkOut[i, j, transposiotn:patchSize, patchSize - remained_y:patchSize] * d_1[:remained_y,:patchSize - transposiotn].transpose(0,1) +  networkOut[i+1,j,shifted_x:patchSize-transposiotn+shifted_x,patchSize - remained_y:patchSize] * a_1[:remained_y,:patchSize - transposiotn].transpose(0,1)
					else:
						output[(i+1)*transposiotn : i*transposiotn + patchSize,patchSize + (j-1)*transposiotn : (j+1)*transposiotn] = networkOut[i,j, transposiotn:patchSize, patchSize-transposiotn:transposiotn] * d.transpose(0,1) + networkOut[i+1 ,j,shifted_x:patchSize-transposiotn+shifted_x, patchSize -transposiotn : transposiotn] * a.transpose(0,1)







				#==Horizontal shared Aresa
				if j < networkOut.size(1)-2 :
					if i == 0:
						output [:transposiotn,(j+1)*transposiotn:j*transposiotn + patchSize] = networkOut[i,j,:transposiotn,transposiotn:patchSize] * d_1 +  networkOut[i,j+1,:transposiotn,:patchSize-transposiotn] * a_1
					elif i == networkOut.size(0)-1:
						output [patchSize + (i-1)*transposiotn:size_x,(j+1)*transposiotn:j*transposiotn + patchSize] = networkOut[i,j,patchSize - remained_x:patchSize,transposiotn:patchSize] * d_1[:remained_x,:] +  networkOut[i,j+1,patchSize -remained_x:patchSize,:patchSize-transposiotn] * a_1[:remained_x,:]
					else:
						output [patchSize + (i-1)*transposiotn : (i+1)*transposiotn,(j+1)*transposiotn:j*transposiotn + patchSize] = networkOut[i,j,patchSize -transposiotn : transposiotn,transposiotn:patchSize] * d +  networkOut[i,j+1,patchSize - transposiotn:transposiotn,:patchSize-transposiotn] * a


				#==Horizontal shared Aresa
				if j == networkOut.size(1)-2 :
					if i == 0:
						output [:transposiotn,(j+1)*transposiotn:j*transposiotn + patchSize] = networkOut[i,j,:transposiotn,transposiotn:patchSize] * d_1 +  networkOut[i,j+1,:transposiotn,shifted_y:patchSize-transposiotn+shifted_y] * a_1
					elif i == networkOut.size(0)-1:
						output [patchSize + (i-1)*transposiotn:size_x,(j+1)*transposiotn:j*transposiotn + patchSize] = networkOut[i,j,patchSize - remained_x:patchSize,transposiotn:patchSize] * d_1[:remained_x,:] +  networkOut[i,j+1,patchSize -remained_x:patchSize,shifted_y:patchSize-transposiotn+shifted_y] * a_1[:remained_x,:]
					else:
						output [patchSize + (i-1)*transposiotn : (i+1)*transposiotn,(j+1)*transposiotn:j*transposiotn + patchSize] = networkOut[i,j,patchSize -transposiotn : transposiotn,transposiotn:patchSize] * d +  networkOut[i,j+1,patchSize - transposiotn:transposiotn,shifted_y:patchSize-transposiotn+shifted_y] * a
					
				

				#==Squared Shared Areas

				if i < networkOut.size(0)-1 and j < networkOut.size(1)-1 :
					output_s_x = (i+1)*transposiotn
					output_e_x = patchSize + i*transposiotn 
					output_s_y = (j+1)*transposiotn
					output_e_y = patchSize + j*transposiotn
					network_pr_s_x = transposiotn
					network_pr_e_x = patchSize
					network_pr_s_y = transposiotn
					network_pr_e_y = patchSize
					network_nx_s_x = 0
					network_nx_e_x = patchSize - transposiotn
					network_nx_s_y = 0
					network_nx_e_y = patchSize - transposiotn
					output [output_s_x:output_e_x,output_s_y:output_e_y] = networkOut[i,j,network_pr_s_x:network_pr_e_x,network_pr_s_y:network_pr_e_y]*d_2d + networkOut[i+1,j+1,network_nx_s_x:network_nx_e_x,network_nx_s_y:network_nx_e_y]*a_2d
					if (i == networkOut.size(0)) or (j == networkOut.size(1)):
						output[output_s_h:output_e_h,output_s_w:output_e_w] = networkOut[i,j,network_pr_s_h:network_pr_e_h,network_pr_s_w:network_pr_e_w]
		output_allClasses[cls] = output
		print('Done')
	return output_allClasses 
#==================================Function to perform measurements over test tiles===========================================

def scores_test(output_allClasses,area,experiment,minus=0):
	if not os.path.exists('testImages/'+experiment):
		os.makedirs('testImages/'+experiment)
	nClasses = output_allClasses.size(0)
	_,output = torch.max(output_allClasses,0) 
	imsave('testImages/'+experiment+'/output_%s.tif' % (area), output.numpy())
	output = output.squeeze()
	print('fullMask_'+area+'.mat')
	mask_allClasses = sio.loadmat('fullMask_'+area+'.mat')
	mask_allClasses = torch.from_numpy(mask_allClasses['m'])
	mask_allClasses = mask_allClasses - minus
	prediction_allClasses = torch.Tensor(3,output.size(0),output.size(1)).fill_(0).float()
	prediction_allClasses_withFalsePos = torch.Tensor(3,output.size(0),output.size(1)).fill_(0).float()
	F1_total = 0
	iou_total = 0
	accaracy_total = 0
	n_clasess_f1 = 0
	tp_summed = 0
	F1_perClass = [0 for c in range(nClasses)]
	confusion = torch.Tensor(nClasses,nClasses)
	for cls in range(nClasses): 
		mask = torch.eq(mask_allClasses,cls)
		mask = mask[:output.size(0),:output.size(1)]
		print('Calculation overal metrics')
		prediction=torch.eq(output,cls)
		if cls == 0:
			prediction_allClasses[0] = prediction_allClasses[0] + prediction.float()
			prediction_allClasses[1] = prediction_allClasses[1] + prediction.float()
			prediction_allClasses[2] = prediction_allClasses[2] + prediction.float()
		elif cls == 1:
			prediction_allClasses[2] = prediction_allClasses[2] + prediction.float()
		elif cls == 2:
			prediction_allClasses[1] = prediction_allClasses[1] + prediction.float()
			prediction_allClasses[2] = prediction_allClasses[2] + prediction.float()
		elif cls == 3:
			prediction_allClasses[1] = prediction_allClasses[1] + prediction.float()
		elif cls == 4:
			prediction_allClasses[0] = prediction_allClasses[0] + prediction.float()
			prediction_allClasses[1] = prediction_allClasses[1] + prediction.float()
		elif cls == 5:
			prediction_allClasses[0] = prediction_allClasses[0] + prediction.float()		
			
		truePositive = torch.eq(prediction.byte(),mask.byte()) * torch.eq(prediction.byte(),1)
		tp = torch.sum(truePositive).item()
		falsePositive = torch.gt(prediction.byte(),mask.byte())
		fp = torch.sum(falsePositive).item()
		falseNegative = torch.lt(prediction.byte(),mask.byte())
		fn = torch.sum(falseNegative).item()
		trueNegative = torch.eq(prediction.byte(),mask.byte()) * torch.eq(prediction.byte(),0)
		tn = torch.sum(trueNegative).item()

		tp_summed	= tp_summed + tp
		precision 	= div(tp, tp+fp)
		recall 		= div(tp, tp+fn)
		IoU 		= div(tp, tp+fn+fp)
		F1 		= div(2 * precision * recall, precision + recall)
		accuracy 	= div(tp+tn, tp+tn+fn+fp);
		iou		= div(tp,tp+fp+fn)
		F1_perClass[cls]= F1
		

		if cls < 5:
			accaracy_total = accaracy_total + accuracy
			F1_total = F1_total + F1
			iou_total = iou_total + iou
			n_clasess_f1 = n_clasess_f1 + 1

		print('--------------------------------------')
		print('class:%d' % cls)
		print('F1 = %f' % F1)
		print('Accuracy = %f' % accuracy)
		print('IoU = %f' % iou)
		print('--------------------------------------')
		best_prediction=prediction;

		im=torch.Tensor(3,output.size(0),output.size(1));
		im[0][truePositive]=0 ;im[1][truePositive]=1 ;im[2][truePositive]=0 ;
		im[0][falsePositive]=1;im[1][falsePositive]=0;im[2][falsePositive]=0;
		im[0][falseNegative]=0;im[1][falseNegative]=0;im[2][falseNegative]=1;
		im[0][trueNegative]=1 ;im[1][trueNegative]=1 ;im[2][trueNegative]=1 ;
		im = im.transpose(2,0)
		im = im.transpose(1,0)
		im = im.numpy()
		imsave('testImages/'+experiment+'/prediction_%d_%s.tif' % (cls,area), im)


		output_class = output_allClasses[cls].squeeze()
		#output_class = torch.exp(output_class)		
		heatMap=torch.Tensor(3,output_class.size(0),output_class.size(1)).fill_(0)
		heatMap[0]=torch.sqrt(output_class.float())
		heatMap[1]=torch.pow(output_class.float(),3)
		heatMap[2]=torch.sin(output_class.float() * np.pi)
		heatMap = heatMap.transpose(2,0)
		heatMap = heatMap.transpose(1,0)
		heatMap = heatMap.numpy() 
		NaNs = np.isnan(heatMap)
		heatMap[NaNs] = 0
		#print(np.sum(1*np.isnan(heatMap)))
		imsave('testImages/'+experiment+'/HeatMap_%d_%s.tif' % (cls,area), heatMap)



		if cls == 0:
			falsePositive_summedAllclasses = falsePositive.clone().float()
		else:
			falsePositive_summedAllclasses = falsePositive_summedAllclasses + falsePositive.float()

		if cls == nClasses - 1:
			prediction_allClasses_withFalsePos = prediction_allClasses.clone()
			prediction_allClasses_withFalsePos[0][falsePositive_summedAllclasses.byte()] = 1			
			prediction_allClasses_withFalsePos[1][falsePositive_summedAllclasses.byte()] = 0			
			prediction_allClasses_withFalsePos[2][falsePositive_summedAllclasses.byte()] = 0

		for kk in range(nClasses):
			mask = torch.eq(mask_allClasses,cls)
			mask = mask[:output.size(0),:output.size(1)]
			confusion[cls][kk] = div(torch.sum(torch.eq(output,kk).float() * mask.float()) , (torch.sum(mask.float())))
	print(confusion)
	overallAccuracy = tp_summed / (output.size(0) * output.size(1))
	F1_total = F1_total / n_clasess_f1
	iou_total = iou_total / n_clasess_f1
	accaracy_total = accaracy_total / n_clasess_f1
	print('--------------------------------------')
	print('F1(mean over classes) = %f' % F1_total);
	print('Accaracy(mean over classes)= %f' % accaracy_total);
	print('Accaracy(overall)= %f' % overallAccuracy);
	print('IoU= %f' % iou_total);
	print('--------------------------------------')

	prediction_allClasses = prediction_allClasses.float()
	prediction_allClasses = prediction_allClasses.transpose(2,0)
	prediction_allClasses = prediction_allClasses.transpose(1,0)
	imsave('testImages/'+experiment+'/prediction_allClasses_%s.tif' % (area), prediction_allClasses.numpy())

	prediction_allClasses_withFalsePos = prediction_allClasses_withFalsePos.float()
	prediction_allClasses_withFalsePos = prediction_allClasses_withFalsePos.transpose(2,0)
	prediction_allClasses_withFalsePos = prediction_allClasses_withFalsePos.transpose(1,0)
	prediction_allClasses_withFalsePos = prediction_allClasses_withFalsePos.numpy()
	NaNs = np.isnan(prediction_allClasses_withFalsePos)
	prediction_allClasses_withFalsePos[NaNs] = 0
	imsave('testImages/'+experiment+'/prediction_allClassesFalsePos_%s.tif' % (area), prediction_allClasses_withFalsePos)
	return F1_total,F1_perClass,accaracy_total,overallAccuracy,confusion.numpy()

#====================================================================
def savePredictions(output_allClasses,areaNumber,experiment,dataset,areaName):
	if not os.path.exists('testImages/'+experiment):
		os.makedirs('testImages/'+experiment)
	nClasses = output_allClasses.size(0)
	_,output = torch.max(output_allClasses,0) 
	output = output.squeeze()
	if dataset == 'isprs':
		prediction_allClasses = torch.Tensor(3,output.size(0),output.size(1)).fill_(0).float()
		for cls in range(nClasses): 
			prediction=torch.eq(output,cls)
			if cls == 0:
				prediction_allClasses[0] = prediction_allClasses[0] + prediction.float()
				prediction_allClasses[1] = prediction_allClasses[1] + prediction.float()
				prediction_allClasses[2] = prediction_allClasses[2] + prediction.float()
			elif cls == 1:
				prediction_allClasses[2] = prediction_allClasses[2] + prediction.float()
			elif cls == 2:
				prediction_allClasses[1] = prediction_allClasses[1] + prediction.float()
				prediction_allClasses[2] = prediction_allClasses[2] + prediction.float()
			elif cls == 3:
				prediction_allClasses[1] = prediction_allClasses[1] + prediction.float()
			elif cls == 4:
				prediction_allClasses[0] = prediction_allClasses[0] + prediction.float()
				prediction_allClasses[1] = prediction_allClasses[1] + prediction.float()
			elif cls == 5:
				prediction_allClasses[0] = prediction_allClasses[0] + prediction.float()

		prediction_allClasses = prediction_allClasses.float()
		prediction_allClasses = prediction_allClasses.transpose(2,0)
		prediction_allClasses = prediction_allClasses.transpose(1,0)
		imsave('testImages/'+experiment+'/top_mosaic_09cm_area%s_class.tif' % areaNumber,prediction_allClasses)
	if dataset == 'inria':
		prediction = torch.eq(output,1)
		prediction = prediction.float()*255
		#prediction = prediction.transpose(2,0)
		#prediction = prediction.transpose(1,0)
		imsave('testImages/'+experiment+'/%s.tif' % areaName,prediction)
			



#======================================================================
def crf(prob,area):
	import pydensecrf.densecrf as dcrf
	from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
	s_x = 16
	s_y = 16
	s_x_Gaussian = 4
	s_y_Gaussian = 4
	s_ch = 0.01
	W = prob.shape[2]
	H = prob.shape[1]
	NLABELS = prob.shape[0]
	image_path = ('top/top_mosaic_09cm_area%s.tif') % area
	dsm_path = ('dsm/dsm_09cm_matching_area%s.tif') % area
	image_rgb  = imread(image_path)
	image_dsm  = imread(dsm_path)
	image_rgb = image_rgb[:H,:W,:]
	image_dsm = image_dsm[:H,:W]
	#input_image = input_image.transpose(2,0,1)
	U = unary_from_softmax(prob)
	input_image_1 = image_rgb[:,:,0:1]
	pairwise_energy_1 = create_pairwise_bilateral(sdims=(s_x,s_y), schan=(s_ch,), img=input_image_1, chdim=2)

	input_image_2 = image_rgb[:,:,1:2]
	pairwise_energy_2 = create_pairwise_bilateral(sdims=(s_x,s_y), schan=(s_ch,), img=input_image_2, chdim=2)

	input_image_3 = image_rgb[:,:,2:3]
	pairwise_energy_3 = create_pairwise_bilateral(sdims=(s_x,s_y), schan=(s_ch,), img=input_image_3, chdim=2)

	input_image_4 = image_dsm[:,:]
	pairwise_energy_4 = create_pairwise_bilateral(sdims=(s_x,s_y), schan=(s_ch,), img=input_image_4, chdim=2)


	d = dcrf.DenseCRF2D(W, H, NLABELS)
	d.setUnaryEnergy(U)
	d.addPairwiseEnergy(pairwise_energy_1, compat=10, kernel=dcrf.FULL_KERNEL)
	d.addPairwiseEnergy(pairwise_energy_2, compat=10, kernel=dcrf.FULL_KERNEL)
	d.addPairwiseEnergy(pairwise_energy_3, compat=10, kernel=dcrf.FULL_KERNEL)
	d.addPairwiseEnergy(pairwise_energy_4, compat=10, kernel=dcrf.FULL_KERNEL)

	d.addPairwiseGaussian(sxy=(s_x_Gaussian,s_y_Gaussian), compat=1)

	Q = d.inference(5)
	out_crf = np.argmax(Q, axis=0).reshape((H,W))
	out_crf_expand = np.zeors(NLABELS,H,W)
	for i in range(NLABELS):
		out_crf_expand[i] = 1*(out_crf==i)
	return out_crf_expand
		
		







