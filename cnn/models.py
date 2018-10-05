import torch
import torch.nn as nn
import torchvision.models as models
import sys
import copy
from resnetModule import Bottleneck,BasicBlock

class encoder_decoder_resnet(nn.Module):
	def __init__(self, nChannelsIn=4, nChannelsOut=2,depth=18,lastLayer='logsoftmax'):
		super(encoder_decoder_resnet, self).__init__()
		self.lastLayer = lastLayer
		#=========Encoder=========#
		if depth == 18:
			resnet = models.resnet18(pretrained=True)
		elif depth == 34:
			resnet = models.resnet34(pretrained=True)
		elif depth == 50:
			resnet = models.resnet50(pretrained=True) 
		elif depth == 101:
			resnet = models.resnet101(pretrained=True) 
		elif depth == 152:
			resnet = models.resnet152(pretrained=True) 
		elif depth == 200:
			resnet = models.resnet200(pretrained=True)
		else:
			sys.stderr.write('Not valid depth for encoder!')

		resnet.conv1 = nn.Conv2d(nChannelsIn,64,7,2,3,bias=False)
		self.encoderBlocks = nn.ModuleList()

		self.encoderBlocks.append(nn.Sequential(
		resnet.conv1,
		resnet.bn1,
		resnet.relu
		))
		self.encoderBlocks.append(nn.Sequential(
		resnet.maxpool,
		resnet.layer1,
		))
		self.encoderBlocks.append(resnet.layer2)
		self.encoderBlocks.append(resnet.layer3)
		self.encoderBlocks.append(resnet.layer4)

		#=========Decoder=========#
		if depth <=34:
			num_feats = [256,128,64,64,64] 
		else:
			num_feats = [1024,512,256,64,64] 

		self.decoderBlocks = nn.ModuleList()

		for i in range(5):
			if i==0:
				self.decoderBlocks.append(nn.Sequential(
						     nn.ConvTranspose2d(num_feats[i]*2,num_feats[i],4,2,1),
						     nn.BatchNorm2d(num_feats[i]),
						     nn.ReLU(),
						     nn.Conv2d(num_feats[i],num_feats[i],3,1,1),
						     nn.BatchNorm2d(num_feats[i]),
						     nn.ReLU(),
						     nn.Conv2d(num_feats[i],num_feats[i],3,1,1),
						     nn.BatchNorm2d(num_feats[i]),
						     nn.ReLU(),
						     ))
			else:
				self.decoderBlocks.append(nn.Sequential(
						     nn.ConvTranspose2d(num_feats[i-1]*2,num_feats[i],4,2,1),
						     nn.BatchNorm2d(num_feats[i]),
						     nn.ReLU(),
						     nn.Conv2d(num_feats[i],num_feats[i],3,1,1),
						     nn.BatchNorm2d(num_feats[i]),
						     nn.ReLU(),
						     nn.Conv2d(num_feats[i],num_feats[i],3,1,1),
						     nn.BatchNorm2d(num_feats[i]),
						     nn.ReLU(),
						     ))
		self.lastConv = nn.Sequential(
		nn.ReLU(),
		nn.Conv2d(64,nChannelsOut,1,1)
		) 
		self.sofmax2d =	nn.Softmax2d()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):

		out_encoder = {}

		for i in range(5):
			if i == 0:
				out_encoder[i] = self.encoderBlocks[i](x)
			else:
				out_encoder[i] = self.encoderBlocks[i](out_encoder[i-1])

		
		for i in range(5):
			if i == 0:
				out = self.decoderBlocks[i](out_encoder[len(out_encoder)-1])
			else:
				encoder_skips = out_encoder[4-i]
				out = self.decoderBlocks[i](torch.cat((out,encoder_skips),1))

		out = self.lastConv(out)
		if self.lastLayer == 'logsoftmax':
			out = self.sofmax2d(out)
			out = out.clamp(min=1e-8)
			out = torch.log(out)
		elif self.lastLayer == 'sigmoid':
			out = self.sigmoid(out)


		return out
