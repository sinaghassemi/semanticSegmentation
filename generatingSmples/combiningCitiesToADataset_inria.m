% Copyright 2018 Telecom Italia S.p.A.

% Redistribution and use in source and binary forms,
% with or without modification, are permitted provided that the following conditions are met:

% Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
% Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation and/or other materials provided
% with the distribution.
% Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
% derived from this software without specific prior written permission.

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
% EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
% OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
% IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
% INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
% BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
% STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


filenames = {'austin','chicago'}%,'kitsap','tyrol-w','vienna'};
nAreas    = size(filenames,2);



train_data          = hdf5read(filenames{1},'/train_data');
train_size          = size(train_data,1) * size(filenames,2);
imageSize_train     = size(train_data,3);
nChannels           = size(train_data,2);
clear train_data

val_data        = hdf5read(filenames{1},'/val_data');
val_size        = size(val_data,1) * size(filenames,2);
imageSize_val   = size(val_data,3);
clear val_data

hdf5Filename    = 'inria.h5';


train_patches=[];
val_patches=[];
train_masks=[];
val_masks=[];
area_indexes_train=[];
area_indexes_val=[];
means=[];
stds=[];



 h5create(hdf5Filename,'/train_data', [train_size nChannels imageSize_train imageSize_train],'Datatype','uint8','ChunkSize',[32 nChannels imageSize_train imageSize_train])
 h5create(hdf5Filename,'/train_mask',[train_size imageSize_train imageSize_train],'Datatype','uint8','ChunkSize',[32 imageSize_train imageSize_train])
 h5create(hdf5Filename,'/train_area',[train_size 1],'Datatype','uint8','ChunkSize',[32 1])
 h5create(hdf5Filename,'/val_data', [val_size nChannels imageSize_val imageSize_val],'Datatype','uint8','ChunkSize',[1 nChannels imageSize_val imageSize_val])
 h5create(hdf5Filename,'/val_mask', [val_size imageSize_val imageSize_val],'Datatype','uint8','ChunkSize',[1 imageSize_val imageSize_val])
 h5create(hdf5Filename,'/val_area',[val_size 1],'Datatype','uint8','ChunkSize',[32 1])

 
 h5create(hdf5Filename,'/mean', [nAreas nChannels],'Datatype','double')
 h5create(hdf5Filename,'/std' , [nAreas nChannels],'Datatype','double')

for i=1:size(filenames,2)
    filenames{i}
    sprintf('reading data:')

    train_data  = hdf5read(filenames{i},'/train_data');
    train_patches=cat(1,train_patches,train_data);
    size_train = size(train_data,1);
    clear train_data
    
     train_mask  = hdf5read(filenames{i},'/train_mask');
     train_masks=  cat(1,train_masks  ,train_mask);
     clear train_mask
    
     val_data    = hdf5read(filenames{i},'/val_data');
     val_patches=cat(1,val_patches,val_data);
     size_val= size(val_data,1);
     clear val_data

     val_mask    = hdf5read(filenames{i},'/val_mask');
     val_masks=cat(1,val_masks,val_mask);
     clear val_mask
     
     
     mean    = hdf5read(filenames{i},'/mean');
     std    = hdf5read(filenames{i},'/std');
     means=cat(1,means,mean);
     stds=cat(1,stds,std);
     
     area_index_train = ones(size_train,1)*(i-1);
     area_indexes_train = cat(1,area_indexes_train,area_index_train);
     area_index_val = ones(size_val,1)*(i-1);
     area_indexes_val = cat(1,area_indexes_val,area_index_val);
end

 h5write(hdf5Filename, '/train_data', train_patches);
 h5write(hdf5Filename, '/train_mask', train_masks);
 h5write(hdf5Filename, '/val_data', val_patches);
 h5write(hdf5Filename, '/val_mask', val_masks);
 h5write(hdf5Filename, '/mean', means);
 h5write(hdf5Filename, '/std', stds);
 h5write(hdf5Filename, '/train_area', area_indexes_train);
 h5write(hdf5Filename, '/val_area', area_indexes_val);




