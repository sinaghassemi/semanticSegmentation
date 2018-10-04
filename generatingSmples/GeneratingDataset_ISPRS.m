clearvars -except patchMean patchSTD

folderPath ='/home/sina/satteliteImage/matlab/ISPRS_semantic_labeling_Vaihingen'; %Path to the data 
hdf5Filename ='vaihingen.h5';
set = 'val'                                                                     ; %'val' | 'train' | 'test' ;%Extracting samples for 'val' | 'train' | 'test' set

nClasses = 6;        
NrTemp = 500;        
DataFormat = 'single'; 
weights=[];
patches=[];
information=[];
masks=[];

if strcmp(set,'train')
    area = [1,3,5,7,13,17,21,23,26,32,37];
    hdf5_data='/train_data';
    hdf5_mask='/train_mask';
    patchSize=364;          
    translation=108;
    chunkSize=32;
    maxUncovered = 100;
end

if strcmp(set,'val')
    area = [11,15,28,30,34];
    hdf5_data='/val_data';
    hdf5_mask='/val_mask';
    patchSize=256;
    translation = 256;
    chunkSize=32;
    maxUncovered = 100;
end

if strcmp(set,'test')
    area = [2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38];
    hdf5_data='/data';
    hdf5_mask='/mask';
    patchSize=512; 
    translation = 400; 
    NrTemp = 100; 
    chunkSize=1; 
end

chunkSizeData=[chunkSize 4 patchSize patchSize];
chunkSizeMask=[chunkSize patchSize patchSize];


for area_index=1:size(area,2) 
    sprintf('reading Channels:')
    top_path = strcat(folderPath,'/top/top_mosaic_09cm_area',int2str(area(area_index)),'.tif')
    dsm_path = strcat(folderPath,'/dsm/dsm_09cm_matching_area',int2str(area(area_index)),'.tif')
    gt_path  = strcat(folderPath,'/gts_for_participants/top_mosaic_09cm_area',int2str(area(area_index)),'.tif')
    top = imread(top_path);
    dsm = imread(dsm_path);
    a=dsm;
    r=top(:,:,1);
    g=top(:,:,2);
    b=top(:,:,3);
    m_all=imread(gt_path);
    sprintf('done!')
    im(:,:,1)=r; 
    im(:,:,2)=g;
    im(:,:,3)=b;
    imshow(im)
    pause()
    
%---fixing mask for building-----------------------------------------------
    sprintf('converting mask to proper format:')
    m=fix_mask(m_all,nClasses);
    sprintf('done!')
%---Validation Mask for each area -----------------------------------------
    valPixels=zeros(size(a));
    if strcmp(set,'val')
        valPixels=ones(size(a));
    end
%---Extrating subtiles-----------------------------------------------------
    rot = 0;
    translation_w = translation;
    translation_h = translation;
    [ temp_data,temp_mask,temp_information] = ExtractPatchesFull( {r,g,b,a},m,patchSize,translation_h,translation_w,maxUncovered,NrTemp);
    patches=cat(1,patches,temp_data);
    masks=cat(1,masks,temp_mask);
    information=cat(1,information,temp_information);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Show image
    infrm=temp_information;
    im(:,:,1)=r; 
    im(:,:,2)=g;
    im(:,:,3)=b;
    im=uint8(im);
    showPatches( im,infrm,patchSize)
    if strcmp(set,'test')
        name = strcat('fullMask_new',area,'.mat');
        save(name,'m')
    end
    %%%%%%%%%%%%%%%%%%%%
    clear r g b m a
    clear im temp_data temp_mask temp_information
%unique(masks)
end


% Test Patches
if strcmp(set,'test')
    
    patchMean = [0 0 0 0];
    patchSTD = [1 1 1 1];
    
% Computing patchMean and Patchstd over test area
    for i=1:4
        p=patches(:,i,:,:);
        patchMean(i)=mean(p(:))
        patchSTD(i)= std(p(:))
    end
    
%  Writing to HDF5 data and std and mean
    datatype=DataFormat;
    h5create(hdf5Filename,hdf5_data, size(patches),'Datatype',datatype,'ChunkSize',chunkSizeData)
    h5create(hdf5Filename,hdf5_mask, size(masks),'Datatype',datatype,'ChunkSize',chunkSizeMask)
    h5create(hdf5Filename,'/mean', size(patchMean),'Datatype',datatype)
    h5create(hdf5Filename,'/std' , size(patchSTD),'Datatype',datatype)
    h5write(hdf5Filename, hdf5_data, patches)
    h5write(hdf5Filename, hdf5_mask, masks)
    h5write(hdf5Filename, '/mean', patchMean)
    h5write(hdf5Filename, '/std', patchSTD)
    
    
% Train/Validation Patches  
% for training patches we compute the mean and std and then we use these
% for validation patches

else

% Computing Mean and STD
    if strcmp(set,'train')
        patchMean = [0 0 0 0];
        patchSTD = [1 1 1 1];
        for i=1:4
            p=patches(:,i,:,:);
            patchMean(i)=mean(p(:))
            patchSTD(i)=std(p(:))
        end
    end
    
 % Normalizing Pathces

    for i=1:4
        patches(:,i,:,:)=patches(:,i,:,:) - patchMean(i);
        patches(:,i,:,:)=patches(:,i,:,:) / patchSTD(i);
    end

    clear temp_data temp_mask temp_information

% Random Permutation
    NrSampels=size(patches,1);
    rndIndex=randperm(NrSampels);
    patches=patches(rndIndex,:,:,:);
    masks=masks(rndIndex,:,:);
    
    h5create(strcat('data_',hdf5Filename),hdf5_data, size(patches),'Datatype','single','ChunkSize',chunkSizeData)
    h5write(strcat('data_',hdf5Filename), hdf5_data, patches)
    
    h5create(strcat('mask_',hdf5Filename),hdf5_mask, size(masks),'Datatype','uint8','ChunkSize',chunkSizeMask)
    h5write(strcat('mask_',hdf5Filename), hdf5_mask, masks)
        
end