clearvars -except patchMean patchSTD
meanAndSTD_computed=0;

path = '/home/sina/satteliteImage/AerialImageDataset/test/' ; %Path to the data 
set  = 'train'                                              ; %Extracting samples for 'val' | 'train' | 'test' set 

city = 'tyrol-e'                                            ;%train and val = {'austin','chicago','kitsap','tyrol-w','vienna'};test = {'bellingham','bloomington','innsbruck','sfo','tyrol-e'}
hdf5Filename = strcat(city,'.h5');
NrTemp = 1500;        
datatype_patch = 'uint8'; 
datatype_label = 'uint8'; 
patches=[];
information=[];
masks=[];

if strcmp(set,'train')
    areas=[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36];
    hdf5_data='/train_data';
    hdf5_mask='/train_mask';
    patchSize=360;% 360         
    translation=360;% 360
    chunkSize=32; 
end

if strcmp(set,'val')
    areas=[1,2,3,4,5];
    hdf5_data='/val_data';
    hdf5_mask='/val_mask';
    patchSize=256; 
    translation = 256;
    chunkSize=1; 
end


if strcmp(set,'test')
    areas=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36];
    hdf5_data='/data';
    hdf5_mask='/mask';
    patchSize=1024; 
    translation = 800; % 
    NrTemp = 1000; 
    chunkSize=1; 
end

chunkSizeData=[chunkSize 3 patchSize patchSize];
chunkSizeMask=[chunkSize patchSize patchSize];

if meanAndSTD_computed==0 && strcmp(set,'test')
    patchMean = [0 0 0];
    patchSTD = [1 1 1];
end
sprintf('reading Channels:')

for area_index=1:size(areas,2)
area=areas(area_index)


[imageData,~] = geotiffread(strcat(path,'images/',city,int2str(area),'.tif'));
if strcmp(set,'test')
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
pause()
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
[ patches_temp,masks_temp,information] = ExtractPatches_inria(imageData(:,:,1),imageData(:,:,2),imageData(:,:,3),m,patchSize,translation_h,translation_w,NrTemp,datatype_patch,datatype_label);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Show image
im(:,:,:)=imageData(:,:,1:3);
im(:,:,3)=im(:,:,3) + m*256;
im=uint8(im);
showPatches( im,information,patchSize)
clear im
pause()
patches=cat(1,patches,patches_temp);
masks=cat(1,masks,masks_temp);

% Test Patches%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(set,'test')
   
    % Computing patchMean and Patchstd over test area
    if area_index==size(areas,2) && ~meanAndSTD_computed
        for i=1:3
            p=patches(:,i,:,:);
            patchMean(i)=mean(p(:))
            patchSTD(i)= std(p(:))
        end
    end
      

    %  Writing to HDF5 data and std and mean
    if meanAndSTD_computed==1
        name = strcat('fullMask_',city,int2str(area),'.mat');
        save(name,'m')
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






% Train and Val%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(set,'train') || strcmp(set,'val')

    % Computing Mean and STD
      if strcmp(set,'train')
            patchMean = [0 0 0];
            patchSTD = [1 1 1];
            for i=1:3
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




    
