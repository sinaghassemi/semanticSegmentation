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

path = '/AerialImageDataset/test/'    ; %Path to the data 
set  = 'test'                         ; %Extracting samples for 'val' | 'train' | 'test' set 

city = 'bellingham'                   ;%train and val = {'austin','chicago','kitsap','tyrol-w','vienna'};test = {'bellingham','bloomington','innsbruck','sfo','tyrol-e'}
hdf5Filename = strcat(city,'.h5');
NrTemp = 1500;        
datatype_patch = 'uint8'; 
datatype_label = 'uint8'; 
patches=[];
information=[];
masks=[];

if strcmp(set,'train')
    areas=[6,7]%,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36];
    hdf5_data='/train_data';
    hdf5_mask='/train_mask';
    patchSize=364;% 360         
    translation=364;% 360
    chunkSize=32; 
    withAnnotation = 1;
    numberOfIteration = 1;
end

if strcmp(set,'val')
    areas=[1]%,2,3,4,5];
    hdf5_data='/val_data';
    hdf5_mask='/val_mask';
    patchSize=256; 
    translation = 256;
    chunkSize=1;
    withAnnotation = 1;
    numberOfIteration = 1;
end


if strcmp(set,'test')
    areas=[1,2,3]%,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36];
    hdf5_data='/data';
    hdf5_mask='/mask';
    patchSize=1024; 
    translation = 800; % 
    NrTemp = 1000; 
    chunkSize=1; 
    withAnnotation = 0;
    numberOfIteration = 2;
end

chunkSizeData=[chunkSize 3 patchSize patchSize];
chunkSizeMask=[chunkSize patchSize patchSize];

if strcmp(set,'test') || strcmp(set,'train')
    patchMean = [0 0 0];
    patchSTD = [1 1 1];
end
sprintf('reading Channels:')



for iteration = 1:numberOfIteration




    for area_index=1:size(areas,2)
    area=areas(area_index)


    [imageData,~] = geotiffread(strcat(path,'images/',city,int2str(area),'.tif'));

    if ~ withAnnotation 
         m=zeros(5000,5000);
         m=uint8(m);
    else
         [m,~] = geotiffread(strcat(path,'gt/',city,int2str(area),'.tif'));
    end

    m = m/255; %convert 0 and 255 to 0 and 1

    sprintf('reading done.')

    % Selecting the labeled portion of images
    im(:,:,:)=imageData(:,:,1:3);
    im(:,:,3)=im(:,:,3) + m*256;
    im=uint8(im);
    imshow(im)
    % pause()
    clear im

    %%%%%%%%%%%%%%%%%%%% computing translations to cover whole image %%%%%%%%%
    heightofArea = size(imageData,1); 
    widthofArea  = size(imageData,2);
    numebrOfpathces_height = ceil(((heightofArea - patchSize) / translation) + 1); 
    numebrOfpathces_width = ceil(((widthofArea - patchSize) / translation) + 1);
    sprintf('numebrOfpathces_height:%d numebrOfpathces_width:%d heightofArea:%d widthofArea:%d',numebrOfpathces_height,numebrOfpathces_width,heightofArea,widthofArea)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rot = 0;
    translation_h = translation;
    translation_w = translation;
    %%%  extracting whole image
    [ patches_temp,masks_temp,information] = ExtractPatches({imageData(:,:,1),imageData(:,:,2),imageData(:,:,3)},m,patchSize,translation_h,translation_w,NrTemp,datatype_patch,datatype_label);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Show image
    im(:,:,:)=imageData(:,:,1:3);
    im(:,:,3)=im(:,:,3) + m*256;
    im=uint8(im);
    showPatches( im,information,patchSize)
    clear im
    % pause()
    patches=cat(1,patches,patches_temp);
    masks=cat(1,masks,masks_temp);

    % Test Patches%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(set,'test')

        % First iteration : Computing patchMean and Patchstd over test area
        if area_index==size(areas,2) && iteration == 1
            sprintf('The end of first iteration: Computing mean and std')
            for i=1:3
                p=single(patches(:,i,:,:));
                patchMean(i)=mean(p(:))
                patchSTD(i)= std(p(:))
            end
        end
        
        clear p


        %  Second iteration : Writing to HDF5 data and std and mean
        if iteration == 2
            if  withAnnotation 
                name = strcat('fullMask_',city,int2str(area),'.mat');
                save(name,'m')
            end
            hdf5Filename = strcat(city,int2str(area),'_test.h5')
            h5create(hdf5Filename,hdf5_data, size(patches_temp),'Datatype','uint8','ChunkSize',chunkSizeData)
            h5create(hdf5Filename,'/mean', size(patchMean),'Datatype','double')
            h5create(hdf5Filename,'/std' , size(patchSTD),'Datatype','double')
            h5write(hdf5Filename, hdf5_data, patches_temp)
            h5write(hdf5Filename, '/mean', patchMean)
            h5write(hdf5Filename, '/std', patchSTD)
        end
    end



    end


end


% Train and Val%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(set,'train') || strcmp(set,'val')

    % Computing Mean and STD
      if strcmp(set,'train')
            for i=1:3
                p=single(patches(:,i,:,:));
                patchMean(i)=mean(p(:))
                patchSTD(i)=std(p(:))
            end
            clear p
        % writing mean and std
        h5create(hdf5Filename,'/mean', size(patchMean),'Datatype','double')
        h5create(hdf5Filename,'/std' , size(patchSTD),'Datatype','double')
        h5write(hdf5Filename, '/mean', patchMean)
        h5write(hdf5Filename, '/std', patchSTD)  
            
      end
      

    % Random Permutation
        NrSampels=size(patches,1);
        rndIndex=randperm(NrSampels);
        patches=patches(rndIndex,:,:,:);
        masks=masks(rndIndex,:,:,:);
        
        size(patches)
        size(masks)

    %  Writing to HDF5
        h5create(hdf5Filename,hdf5_data, size(patches),'Datatype',datatype_patch,'ChunkSize',chunkSizeData)
        h5create(hdf5Filename,hdf5_mask, size(masks),'Datatype',datatype_label,'ChunkSize',chunkSizeMask)
        h5write(hdf5Filename, hdf5_data, patches)
        h5write(hdf5Filename, hdf5_mask, masks)    
        
end




    
