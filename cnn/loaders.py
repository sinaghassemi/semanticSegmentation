import torch.utils.data as data
import h5py
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import interpolation
import scipy.stats as st
def random_crop(img,label,size,lossWeight=None,useWeightedLoss=False):
	diff_h = img.shape[1] - size
	diff_w = img.shape[2] - size
	pos_h = np.random.randint(1,diff_h)
	pos_w = np.random.randint(1,diff_w)
	img =   img[:,pos_h:pos_h + size,pos_w:pos_w + size]
	label = label[pos_h:pos_h + size,pos_w:pos_w + size]
	if useWeightedLoss:
		lossWeight = lossWeight[:,pos_h:pos_h + size,pos_w:pos_w + size]
	return img,label,lossWeight	

def random_rot(img,label,size,lossWeight=None,useWeightedLoss=False):
	degrees = [30,60,90,120,150,180,210,240,270,300,330]
	index = np.random.randint(0,len(degrees))
	angel = degrees[index]
	img = interpolation.rotate(img,angel,axes=(1,2),order=0)
	label = interpolation.rotate(label,angel,axes=(0,1),order=0)
	diff_h = img.shape[1] - size
	diff_w = img.shape[2] - size
	pos_h = int(diff_h/2)
	pos_w = int(diff_w/2)
	img =   img[:,pos_h:pos_h + size,pos_w:pos_w + size]
	label = label[pos_h:pos_h + size,pos_w:pos_w + size]
	if useWeightedLoss:
		lossWeight = interpolation.rotate(lossWeight,angel,axes=(1,2),order=0)
		lossWeight = lossWeight[:,pos_h:pos_h + size,pos_w:pos_w + size]	
	return img,label,lossWeight
 
# producing a kernel which values are changing from center_value to border_value according to its distance from center 			
def kernel(size=(5,5),center_value=0,border_value=1,sigma = 0.7):
	k=np.ones(size) 
	center_x = (size[0]+1) / 2
	center_y = (size[1]+1) / 2
	max_d = np.sqrt(size[0]**2 + size[1]**2)
	for i in range(size[0]):
		for j in range(size[1]):
			d = sigma * np.sqrt((i-center_x)**2 + (j-center_y)**2)
			k[i,j] = ((border_value - center_value) / max_d) * d + center_value
			
	return  k 

def addShadow(img):
	size_x = np.random.randint(150,230)
	size_y = np.random.randint(150,230)
	sigma=np.random.uniform(0.5,1)
	img[:3,:size_x,:size_y] =  kernel(size=(size_x,size_y),center_value=0,border_value=2,sigma=sigma)*img[:3,:size_x,:size_y] + kernel(size=(size_x,size_y),center_value=np.min(img),border_value=1,sigma=sigma)
	return img

def augment(img,label,size,lossWeight=None,useWeightedLoss=False):
	
	if np.random.uniform() > 0.5:
		#random crop
		img,label,lossWeight = random_crop(img,label,size,lossWeight=lossWeight,useWeightedLoss=useWeightedLoss)
	else:
		#random rotation
		img,label,lossWeight = random_rot(img,label,size,lossWeight=lossWeight,useWeightedLoss=useWeightedLoss)
	
	if np.random.uniform() > 0.5:
		#H flip
		img = np.flip(img,1).copy()
		label = np.flip(label,0).copy()
		if useWeightedLoss:
			lossWeight = np.flip(lossWeight,1).copy()			
	if np.random.uniform() > 0.5:
		#V flip
		img = np.flip(img,2).copy()
		label = np.flip(label,1).copy()
		if useWeightedLoss:
			lossWeight = np.flip(lossWeight,2).copy()
	'''if np.random.uniform() > 0.5:
		#Adding shadow
		img = addShadow(img)'''


		
	return [img,label,lossWeight]


class loader(data.Dataset):
	def __init__(self,fileNameData,fileNameMask,lockImg,lockLabel,**kwargs):

		self.fileData = h5py.File(fileNameData, 'r')
		self.fileMask = h5py.File(fileNameMask, 'r')
		self.lockImg = lockImg
		self.lockLabel = lockLabel
		self.dataset = kwargs.get('dataset') or 'Train'
		self.imageSize = kwargs.get('imageSize') or 364
		self.nChannelsIn = kwargs.get('nChannelsIn') or 4
		self.nChannelsOut = kwargs.get('nChannelsOut') or 2
		self.labelFormat = kwargs.get('labelFormat') or 'float'
		self.aug = kwargs.get('aug') or False
		self.patchSize = kwargs.get('patchSize') or 256
		self.preload = kwargs.get('preload') or False
		self.minus = kwargs.get('minus') or 0
		self.fileNameWeight = kwargs.get('fileNameWeight') or None
		self.returnLabels = kwargs.get('returnLabels') or False
		self.normalize = kwargs.get('normalize') or False
		self.areaBasedNorm = kwargs.get('areaBasedNorm') or False
		if self.fileNameWeight:
			self.useWeightedLoss = True
			self.fileWeight = h5py.File(fileNameWeight, 'r')
		else:
			self.useWeightedLoss = False

		if self.normalize:		
			self.mean = self.fileData['/mean'][:,:]
			self.mean = np.squeeze(self.mean)
			self.std = self.fileData['/std'][:,:]
			self.std = np.squeeze(self.std)
		# Optionally preloading the train/val data/mask into memory
		if self.preload:
			if self.dataset == 'train':
				self.data = self.fileData['/train_data'][:,:,:,:]
				self.mask = self.fileMask['/train_mask'][:,:,:]
				if self.useWeightedLoss:
					self.lossWeight = self.fileWeight['/train_weight'][:,:,:,:]
			elif self.dataset == 'val':
				self.data = self.fileData['/val_data'][:,:,:,:]
				self.mask = self.fileMask['/val_mask'][:,:,:]
				if self.useWeightedLoss:
					self.lossWeight = self.fileWeight['/val_weight'][:,:,:,:]
			elif self.dataset == 'test':
				self.data = self.fileData['/data'][:,:,:,:]
				if self.returnLabels:
					self.mask = self.fileMask['/mask'][:,:,:]
		if self.areaBasedNorm:
			if self.dataset == 'train': 
				self.areas= self.fileData['/train_area'][:,:]
			elif self.dataset == 'val': 
				self.areas= self.fileData['/val_area'][:,:]
				

	
	def __getitem__(self, index):
		# loading imeg from the backend
		img, label, lossWeight = self.__loadSample (index)
		img = np.swapaxes(img,0,3)
		img = np.swapaxes(img,1,2)
		img = np.squeeze(img) # 4 256 256

		if self.normalize:
			img = img.astype(float)
			if self.areaBasedNorm:
				area = self.areas[0,index]
				for ch in range(img.shape[0]):
					img[ch] = img[ch] - self.mean[ch,area]
					img[ch] = img[ch] / self.std[ch,area]
			else:
				for ch in range(img.shape[0]):
					img[ch] = img[ch] - self.mean[ch]
					img[ch] = img[ch] / self.std[ch]

		if self.returnLabels:
			label = np.swapaxes(label,0,2)
			label = np.squeeze(label) # 256 256
		if self.useWeightedLoss:
			lossWeight = np.swapaxes(lossWeight,0,2)
			lossWeight = np.squeeze(lossWeight) # 256 256
			
		
		if self.aug:
			img,label,lossWeight = augment(img,label,self.patchSize,lossWeight=lossWeight,useWeightedLoss=self.useWeightedLoss) 
		
		img = torch.from_numpy(img)
		img = img.float()
		if self.returnLabels:
			label = torch.from_numpy(label)
			if self.labelFormat == 'float':
				#unsqueezing
				num_classes = self.nChannelsOut
				l = torch.Tensor(num_classes,label.shape[0],label.shape[1])
				for cls in range(num_classes):
					l[cls] = torch.eq(label,cls)
				label = l.float()
			if self.labelFormat == 'long':
				label = label.long()
		if self.useWeightedLoss:
			lossWeight = torch.from_numpy(lossWeight)			

		return img,label,lossWeight
	
	def __len__(self):
		if self.dataset == 'train':
			 return self.fileMask['/train_mask'].shape[2]
		elif self.dataset == 'val':
			 return self.fileMask['/val_mask'].shape[2]
		elif self.dataset == 'test':
			 return self.fileMask['/data'].shape[3]
			
	# Loads a sample image and mask either from file or from preloaded memory area
	def __loadSample (self, index):
		
		# Everything is preloaded into memory
		if self.preload:
			img = self.data[:, :, :, index:index+1]	#256 256 4 1
			if self.returnLabels:
				label = self.mask[:, :, index:index+1] - self.minus #for isprs
			else:
				label = 0

			lossWeight = 0
			if self.useWeightedLoss:
				lossWeight = self.lossWeight[:, :, :, index:index+1]
		
		# Accessing everything from HDF5 file
		else:
			if self.dataset == 'train':
				datasetPrefix = '/train_'
			elif self.dataset == 'val':
				datasetPrefix = '/val_'
			elif self.dataset == 'test':
				datasetPrefix = ''
			
			self.lockImg.acquire()
			img = self.fileData[datasetPrefix + 'data'][0:self.imageSize, 0:self.imageSize, 0:self.nChannelsIn, index:index+1]	#256 256 4 1
			self.lockImg.release()
			self.lockLabel.acquire()
			if self.returnLabels:
				label = self.fileMask[datasetPrefix + 'mask'][0:self.imageSize, 0:self.imageSize, index:index+1] - self.minus
			else:
				label = 0			



			self.lockLabel.release()
			lossWeight = 0
			if self.useWeightedLoss:
				self.lockLabel.acquire()
				lossWeight = self.fileMask['/lossWeights'][:, :, index:index+1]
				self.lockLabel.release()
		
		return img, label, lossWeight




class loader_domainTest(data.Dataset):
	def __init__(self,fileNameData,dataset='Train',imageSize=364,nChannelsIn=4):
		self.fileData = h5py.File(fileNameData, 'r')
		self.dataset = dataset
		self.imageSize = imageSize
		self.nChannelsIn = nChannelsIn
		
		# Optionally preloading the train/val data/mask into memory
		if self.dataset == 'train':
			self.data = self.fileData['/train_data'][:,:,:,:]
			self.mask = self.fileData['/train_mask'][:,:]
		elif self.dataset == 'val':
			self.data = self.fileData['/val_data'][:,:,:,:]
			self.mask = self.fileData['/val_mask'][:,:]
	
	def __getitem__(self, index):
		img = self.data[:, :, :, index:index+1]	
		label = self.mask[0, index:index+1] 
		img = np.swapaxes(img,0,3)
		img = np.swapaxes(img,1,2)
		img = np.squeeze(img) 
		label = np.squeeze(label)
		img = torch.from_numpy(img)
		img = img.float()
		label = torch.from_numpy(label)
		label = label.long()
		return img,label
	
	def __len__(self):
		if self.dataset == 'train':
			 return self.fileData['/train_mask'].shape[1]
		elif self.dataset == 'val':
			 return self.fileData['/val_mask'].shape[1]


class loader_DA(data.Dataset):
	def __init__(self,fileNameData,filenameTarget,lockImg,lockLabel,**kwargs):

		self.fileData = h5py.File(fileNameData, 'r')
		self.fileTarget = h5py.File(filenameTarget, 'r')
		self.lockImg = lockImg
		self.lockLabel = lockLabel
		self.dataset = kwargs.get('dataset') or 'Train'
		self.imageSize = kwargs.get('imageSize') or 364
		self.nChannelsIn = kwargs.get('nChannelsIn') or 4
		self.nChannelsOut = kwargs.get('nChannelsOut') or 2
		self.labelFormat = kwargs.get('labelFormat') or 'float'
		self.aug = kwargs.get('aug') or False
		self.patchSize = kwargs.get('patchSize') or 256
		self.preload = kwargs.get('preload') or False
		self.minus = kwargs.get('minus') or 0
		self.returnLabels = kwargs.get('returnLabels') or False
		self.normalize = kwargs.get('normalize') or False
		self.areaBasedNorm = kwargs.get('areaBasedNorm') or False
		self.numTargets = self.fileTarget['/data'].shape[3]
		print('number of targets=%d' % self.numTargets )

		if self.normalize:		
			self.mean = self.fileData['/mean'][:,:]
			self.mean = np.squeeze(self.mean)
			self.std = self.fileData['/std'][:,:]
			self.std = np.squeeze(self.std)
			self.meanTarget = self.fileTarget['/mean'][:,:]
			self.meanTarget = np.squeeze(self.meanTarget)
			self.stdTarget = self.fileTarget['/std'][:,:]
			self.stdTarget = np.squeeze(self.stdTarget)
		# Optionally preloading the train/val data/mask into memory
		if self.preload:
			if self.dataset == 'train':
				self.data = self.fileData['/train_data'][:,:,:,:]
				self.mask = self.fileData['/train_mask'][:,:,:]
				self.dataTarget = self.fileTarget['/data'][:,:,:,:]
			elif self.dataset == 'val':
				self.data = self.fileData['/val_data'][:,:,:,:]
				self.mask = self.fileData['/val_mask'][:,:,:]
			elif self.dataset == 'test':
				self.data = self.fileData['/data'][:,:,:,:]
				if self.returnLabels:
					self.mask = self.fileData['/mask'][:,:,:]
		if self.areaBasedNorm:
			if self.dataset == 'train': 
				self.areas= self.fileData['/train_area'][:,:]
				self.areasTarget= self.fileTarget['/area'][:,:]
			elif self.dataset == 'val': 
				self.areas= self.fileData['/val_area'][:,:]
				

	
	def __getitem__(self, index):
		# loading imeg from the backend
		index_target = index % self.numTargets
		img, imgTarget, label = self.__loadSample(index,index_target)
		img = np.swapaxes(img,0,3)
		img = np.swapaxes(img,1,2)
		img = np.squeeze(img) # 4 256 256
		imgTarget = np.swapaxes(imgTarget,0,3)
		imgTarget = np.swapaxes(imgTarget,1,2)
		imgTarget = np.squeeze(imgTarget) # 4 256 256

		if self.normalize:
			img = img.astype(float)
			if self.areaBasedNorm:
				area = self.areas[0,index]
				areaTarget = self.areasTarget[0,index_target]
				for ch in range(img.shape[0]):
					img[ch] = img[ch] - self.mean[ch,area]
					img[ch] = img[ch] / self.std[ch,area]
					imgTarget[ch] = imgTarget[ch] - self.meanTarget[ch,areaTarget]
					imgTarget[ch] = imgTarget[ch] / self.stdTarget[ch,areaTarget]
			else:
				for ch in range(img.shape[0]):
					img[ch] = img[ch] - self.mean[ch]
					img[ch] = img[ch] / self.std[ch]
					imgTarget[ch] = imgTarget[ch] - self.meanTarget[ch]
					imgTarget[ch] = imgTarget[ch] / self.stdTarget[ch]

		if self.returnLabels:
			label = np.swapaxes(label,0,2)
			label = np.squeeze(label) # 256 256
		
		if self.aug:
			img,label,_   = augment(img,label,self.patchSize,lossWeight=0,useWeightedLoss=False)
			imgTarget,_,_ = augment(imgTarget,np.zeros((imgTarget.shape[1],imgTarget.shape[2],1)),self.patchSize,lossWeight=0,useWeightedLoss=False)  
		
		img = torch.from_numpy(img)
		img = img.float()
		imgTarget = torch.from_numpy(imgTarget)
		imgTarget = imgTarget.float()
		if self.returnLabels:
			label = torch.from_numpy(label)
			if self.labelFormat == 'float':
				#unsqueezing
				num_classes = self.nChannelsOut
				l = torch.Tensor(num_classes,label.shape[0],label.shape[1])
				for cls in range(num_classes):
					l[cls] = torch.eq(label,cls)
				label = l.float()
			if self.labelFormat == 'long':
				label = label.long()			

		return img,imgTarget,label
	
	def __len__(self):
		if self.dataset == 'train':
			 return self.fileData['/train_mask'].shape[2]
		elif self.dataset == 'val':
			 return self.fileData['/val_mask'].shape[2]
		elif self.dataset == 'test':
			 return self.fileData['/data'].shape[3]
			
	# Loads a sample image and mask either from file or from preloaded memory area
	def __loadSample (self, index,index_target):
		
		# Everything is preloaded into memory
		if self.preload:
			img = self.data[:, :, :, index:index+1]	#256 256 4 1
			if self.dataset == 'train':
				imgTarget = self.dataTarget[:, :, :, index_target:index_target+1]	#256 256 4 1
			else:
				imgTarget = np.zeros((img.shape))
			if self.returnLabels:
				label = self.mask[:, :, index:index+1] - self.minus #for isprs
			else:
				label = 0

		
		# Accessing everything from HDF5 file
		else:
			if self.dataset == 'train':
				datasetPrefix = '/train_'
			elif self.dataset == 'val':
				datasetPrefix = '/val_'
			elif self.dataset == 'test':
				datasetPrefix = ''
			
			self.lockImg.acquire()
			img = self.fileData[datasetPrefix + 'data'][:,:,:,index:index+1]	#256 256 4 1
			if self.dataset == 'train':
				imgTarget = self.fileTarget['/data'][:,:,:,index_target:index_target+1]	#256 256 4 1
			else:
				imgTarget = np.zeros((img.shape))
			self.lockImg.release()
			self.lockLabel.acquire()
			if self.returnLabels:
				label = self.fileData[datasetPrefix + 'mask'][0:self.imageSize, 0:self.imageSize, index:index+1] - self.minus
			else:
				label = 0			



			self.lockLabel.release()
		
		return img, imgTarget, label


	
'''
class isprs_loader(data.Dataset):
	def __init__(self,dataFile,train=True,transform=None,imageSize=256,nChannelsIn=4,labelFormat='float'):
		self.file = h5py.File(dataFile, 'r')
		self.train = train
		self.imageSize = imageSize
		self.nChannelsIn = nChannelsIn
		self.labelFormat = labelFormat
	def __getitem__(self, index):
		if self.train:
			img = self.file['/train_data'][0:self.imageSize,0:self.imageSize,0:self.nChannelsIn,index:index+1]
			img = np.swapaxes(img,0,3)
			img = np.swapaxes(img,1,2)
			label = self.file['/train_mask'][0:self.imageSize,0:self.imageSize,index:index+1] #- 1
			label = np.swapaxes(label,0,2)
		else:
			img = self.file['/val_data'][0:self.imageSize,0:self.imageSize,0:self.nChannelsIn,index:index+1]
			img = np.swapaxes(img,0,3)
			img = np.swapaxes(img,1,2)
			label = self.file['/val_mask'][0:self.imageSize,0:self.imageSize,index:index+1] #- 1
			label = np.swapaxes(label,0,2)


		img = np.squeeze(img)
		img = torch.from_numpy(img)
		img = img.float()
		label = np.squeeze(label)
		label = torch.from_numpy(label)
		if self.labelFormat == 'float':
			label = label.float()
		if self.labelFormat == 'long':
			label = label.long()
		return img, label
	def __len__(self):
		if self.train:
			 return self.file['/train_mask'].shape[2]
		else:
			 return self.file['/val_mask'].shape[2]
class isprs_loader(data.Dataset):
	def __init__(self,dataFile,train=True,transform=None,imageSize=256,nChannelsIn=4,labelFormat='float'):
		self.file = h5py.File(dataFile, 'r')
		self.train = train
		self.imageSize = imageSize
		self.nChannelsIn = nChannelsIn
		self.labelFormat = labelFormat
	def __getitem__(self, index):
		if self.train:
			img = self.file['/train_data'][0:self.imageSize,0:self.imageSize,0:self.nChannelsIn,index:index+1]
			img = np.swapaxes(img,0,3)
			img = np.swapaxes(img,1,2)
			label = self.file['/train_mask'][0:self.imageSize,0:self.imageSize,index:index+1] #- 1
			label = np.swapaxes(label,0,2)
		else:
			img = self.file['/val_data'][0:self.imageSize,0:self.imageSize,0:self.nChannelsIn,index:index+1]
			img = np.swapaxes(img,0,3)
			img = np.swapaxes(img,1,2)
			label = self.file['/val_mask'][0:self.imageSize,0:self.imageSize,index:index+1] #- 1
			label = np.swapaxes(label,0,2)
		img = np.squeeze(img)
		img = torch.from_numpy(img)
		img = img.float()
		label = np.squeeze(label)
		label = torch.from_numpy(label)
		if self.labelFormat == 'float':
			label = label.float()
		if self.labelFormat == 'long':
			label = label.long()
		return img, label
	def __len__(self):
		if self.train:
			 return self.file['/train_mask'].shape[2]
		else:
			 return self.file['/val_mask'].shape[2]

class isprs_loader_testset(data.Dataset):
	def __init__(self,dataFile,imageSize=512,nChannelsIn=4,minus=1):
		self.file = h5py.File(dataFile, 'r')
		self.mean = self.file['/mean'][0:4,0:1]
		self.mean = np.squeeze(self.mean)
		self.std = self.file['/std'][0:4,0:1]
		self.std = np.squeeze(self.std)
		self.imageSize = imageSize
		self.nChannelsIn = nChannelsIn
	def __getitem__(self, index):
		img = self.file['/data'][0:self.imageSize,0:self.imageSize,0:self.nChannelsIn,index:index+1]
		img = np.swapaxes(img,0,3)
		img = np.swapaxes(img,1,2)
		img = np.squeeze(img)  # 4 360 360
		for ch in range(4):
			img[ch] = img[ch] - self.mean[ch]
			img[ch] = img[ch] / self.std[ch]
		img = torch.from_numpy(img)
		img = img.float()
		#label=0
		label = self.file['/mask'][0:self.imageSize,0:self.imageSize,index:index+1]
		label = np.swapaxes(label,0,2)
		label = np.squeeze(label)
		label = torch.from_numpy(label)
		label = label.float()
		return img, label
	def __len__(self):
		return self.file['/data'].shape[3]
'''
