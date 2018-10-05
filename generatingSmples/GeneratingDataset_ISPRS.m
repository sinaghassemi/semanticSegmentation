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


clearvars -except patchMean patchSTD
folderPath ='/home/sina/satteliteImage/matlab/ISPRS_semantic_labeling_Vaihingen'; %Path to the data 
hdf5Filename ='vaihingen.h5';
set = 'test'                                                                    ; %'val' | 'train' | 'test' ;%Extracting samples for 'val' | 'train' | 'test' set



nClasses = 6;        
NrTemp = 500;        
DataFormat = 'uint8'; 
weights=[];
patches=[];
information=[];
masks=[];

if strcmp(set,'train')
    area = [1,3]%,5,7,13,17,21,23,26,32,37];
    hdf5_data='/train_data';
    hdf5_mask='/train_mask';
    patchSize=364;          
    translation=108;
    chunkSize=32;
    withAnnotation = 1;
end

if strcmp(set,'val')
    area = [11]%,15,28,30,34];
    hdf5_data='/val_data';
    hdf5_mask='/val_mask';
    patchSize=256;
    translation = 256;
    chunkSize=32;
    withAnnotation = 1;
end

if strcmp(set,'test')
    area = [2] %[2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38];
    hdf5_data='/data';
    hdf5_mask='/mask';
    patchSize=512; 
    translation = 400;
    NrTemp = 100; 
    chunkSize=1;
    withAnnotation = 0;
end

chunkSizeData=[chunkSize 4 patchSize patchSize];
chunkSizeMask=[chunkSize patchSize patchSize];


for area_index=1:size(area,2) 
    sprintf('reading Channels:')
    top_path = strcat(folderPath,'/top/top_mosaic_09cm_area',int2str(area(area_index)),'.tif')
    dsm_path = strcat(folderPath,'/dsm/dsm_09cm_matching_area',int2str(area(area_index)),'.tif')
    top = imread(top_path);
    dsm = imread(dsm_path);
    a=dsm;
    r=top(:,:,1);
    g=top(:,:,2);
    b=top(:,:,3);
    if withAnnotation
        gt_path  = strcat(folderPath,'/gts_for_participants/top_mosaic_09cm_area',int2str(area(area_index)),'.tif')
        m_all=imread(gt_path);
        %---fixing mask for building-----------------------------------------------
        sprintf('converting mask to proper format:')
        m=fix_mask(m_all,nClasses);
        sprintf('done!')
    else
        m=zeros(size(r,1),size(r,2)); 
    end
    sprintf('done!')
    im(:,:,1)=r; 
    im(:,:,2)=g;
    im(:,:,3)=b;
    imshow(im)
    %pause()
    clear im

    %%%%%%%%%%%%%%%%%%%% computing translations to cover whole image %%%%%%%%%
    heightofArea = size(m,1); 
    widthofArea  = size(m,2);
    numebrOfpathces_height = ceil(((heightofArea - patchSize) / translation) + 1); 
    numebrOfpathces_width = ceil(((widthofArea - patchSize) / translation) + 1);
    sprintf('numebrOfpathces_height:%d numebrOfpathces_width:%d heightofArea:%d widthofArea:%d',numebrOfpathces_height,numebrOfpathces_width,heightofArea,widthofArea)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rot = 0;
    translation_h = translation;
    translation_w = translation;
    %%%  extracting whole image
    [ patches_temp,masks_temp,information] = ExtractPatches({r,g,b,a},m,patchSize,translation_h,translation_w,NrTemp,DataFormat,DataFormat);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Show image
    im(:,:,1)=r; 
    im(:,:,2)=g;
    for nc = 1:nClasses
         im(:,:,3)= b + 128*uint8((m==nc));
         im=uint8(im);
         showPatches( im,information,patchSize)
         %pause()
    end
    clear im
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    patches=cat(1,patches,patches_temp);
    masks=cat(1,masks,masks_temp);
    % Test Patches%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    if strcmp(set,'test')
        if withAnnotation
            name = strcat('fullMask_isprs_vaihingen',int2str(area(area_index)),'.mat');
            save(name,'m')
        end
        % Test Patches
            patchMean = [0 0 0 0];
            patchSTD = [1 1 1 1];

        % Computing patchMean and Patchstd over test area
            for i=1:4
                p=single(patches(:,i,:,:));
                patchMean(i)=mean(p(:))
                patchSTD(i)= std(p(:))
            end

        %  Writing to HDF5 data and std and mean
            datatype=DataFormat;
            hdf5Filename=strcat('isprs_vaihingen',int2str(area(area_index)),'_test.h5');%isprs_vaihingen28_test.h5
            h5create(hdf5Filename,hdf5_data, size(patches),'Datatype',datatype,'ChunkSize',chunkSizeData)
            h5create(hdf5Filename,hdf5_mask, size(masks),'Datatype',datatype,'ChunkSize',chunkSizeMask)
            h5create(hdf5Filename,'/mean', size(patchMean),'Datatype',datatype)
            h5create(hdf5Filename,'/std' , size(patchSTD),'Datatype',datatype)
            h5write(hdf5Filename, hdf5_data, patches)
            h5write(hdf5Filename, hdf5_mask, masks)
            h5write(hdf5Filename, '/mean', patchMean)
            h5write(hdf5Filename, '/std', patchSTD)   
    end
    %%%%%%%%%%%%%%%%%%%%
    clear r g b m a
    clear im temp_data temp_mask temp_information
%unique(masks)
end


% Train and Val%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(set,'train') || strcmp(set,'val')

% Computing Mean and STD
    if strcmp(set,'train')
        patchMean = [0 0 0 0];
        patchSTD = [1 1 1 1];
        for i=1:4
            p=single(patches(:,i,:,:));
            patchMean(i)=mean(p(:))
            patchSTD(i)=std(p(:))
        end
    
% writing mean and std
	h5create(hdf5Filename,'/mean', size(patchMean),'Datatype','double')
	h5create(hdf5Filename,'/std' , size(patchSTD),'Datatype','double')
	h5write(hdf5Filename, '/mean', patchMean)
	h5write(hdf5Filename, '/std', patchSTD)  
   end

    clear temp_data temp_mask temp_information

% Random Permutation
    NrSampels=size(patches,1);
    rndIndex=randperm(NrSampels);
    patches=patches(rndIndex,:,:,:);
    masks=masks(rndIndex,:,:);
    
    
%  Writing to HDF5
    h5create(hdf5Filename,hdf5_data, size(patches),'Datatype',DataFormat,'ChunkSize',chunkSizeData)
    h5create(hdf5Filename,hdf5_mask, size(masks),'Datatype',DataFormat,'ChunkSize',chunkSizeMask)
    h5write(hdf5Filename, hdf5_data, patches)
    h5write(hdf5Filename, hdf5_mask, masks) 
        
end