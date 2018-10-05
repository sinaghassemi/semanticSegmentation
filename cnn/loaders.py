import torch.utils.data as data
import h5py
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import interpolation
import scipy.stats as st


def random_crop(img,label,size):
	diff_h = img.shape[1] - size
	diff_w = img.shape[2] - size
	pos_h = np.random.randint(1,diff_h)
	pos_w = np.random.randint(1,diff_w)
	img =   img[:,pos_h:pos_h + size,pos_w:pos_w + size]
	label = label[pos_h:pos_h + size,pos_w:pos_w + size]
	return img,label	

def random_rot(img,label,size):
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
	return img,label

def augment(img,label,size):
	
	if np.random.uniform() > 0.5:
		#random crop
		img,label = random_crop(img,label,size)
	else:
		#random rotation
		img,label = random_rot(img,label,size)
	if np.random.uniform() > 0.5:
		#H flip
		img = np.flip(img,1).copy()
		label = np.flip(label,0).copy()			
	if np.random.uniform() > 0.5:
		#V flip
		img = np.flip(img,2).copy()
		label = np.flip(label,1).copy()
	
	return [img,label]


class loader(data.Dataset):
	def __init__(self,fileNameData,lockImg,lockLabel,**kwargs):

		self.fileData = h5py.File(fileNameData, 'r')
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
		self.returnLabels = kwargs.get('returnLabels') or False
		self.normalize = kwargs.get('normalize') or False
		self.areaBasedNorm = kwargs.get('areaBasedNorm') or False


		if self.normalize:		
			self.mean = self.fileData['/mean'][:,:]
			self.mean = np.squeeze(self.mean)
			self.std = self.fileData['/std'][:,:]
			self.std = np.squeeze(self.std)
		# Optionally preloading the train/val data/mask into memory
		if self.preload:
			if self.dataset == 'train':
				self.data = self.fileData['/train_data'][:,:,:,:]
				self.mask = self.fileData['/train_mask'][:,:,:]
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
			elif self.dataset == 'val': 
				self.areas= self.fileData['/val_area'][:,:]
				

	
	def __getitem__(self, index):
		# loading imeg from the backend
		img, label = self.__loadSample (index)
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
			
		
		if self.aug:
			img,label = augment(img,label,self.patchSize) 
		
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

		return img,label
	
	def __len__(self):
		if self.dataset == 'train':
			 return self.fileData['/train_mask'].shape[2]
		elif self.dataset == 'val':
			 return self.fileData['/val_mask'].shape[2]
		elif self.dataset == 'test':
			 return self.fileData['/data'].shape[3]
			
	# Loads a sample image and mask either from file or from preloaded memory area
	def __loadSample (self, index):
		
		# Everything is preloaded into memory
		if self.preload:
			img = self.data[:, :, :, index:index+1]	#256 256 4 1
			if self.returnLabels:
				label = self.mask[:, :, index:index+1] 
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
			img = self.fileData[datasetPrefix + 'data'][0:self.imageSize, 0:self.imageSize, 0:self.nChannelsIn, index:index+1]	#256 256 4 1
			self.lockImg.release()
			self.lockLabel.acquire()
			if self.returnLabels:
				label = self.fileData[datasetPrefix + 'mask'][0:self.imageSize, 0:self.imageSize, index:index+1] 
			else:
				label = 0			



			self.lockLabel.release()

		
		return img, label


