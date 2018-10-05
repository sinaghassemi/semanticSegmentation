function [ temp_data,temp_mask,temp_information ] = ExtractPatches(data,m,patchSize,translation_h,translation_w,NrTemp,type_patch,type_mask)
%extract patches of a given image
% args->>>>>>>>
% data channel
% m mask
% NrTemp number of temporary memory patch allocation
% translation step size between two consecutive patches
% >>>>>>>>>>>
h=size(m,1);
w=size(m,2);
temp_data = zeros(NrTemp,size(data,2),patchSize,patchSize,type_patch);
temp_mask = zeros(NrTemp,patchSize,patchSize,type_mask);
temp_information=zeros(NrTemp,3);
nrSamples=1;
numebrOfpathces_height = ceil(((h - patchSize) / translation_h) + 1); 
numebrOfpathces_width = ceil(((w - patchSize) / translation_w) + 1);
    for nH=1:numebrOfpathces_height
        i = 1 + (nH-1)*translation_h; 
      if i + patchSize > h 
          i = i - translation_h;%undo translation
          i = h - patchSize;
      end
        for nW=1:numebrOfpathces_width
            j = 1 + (nW-1)*translation_w; 
            if j + patchSize > w 
                j = j - translation_w;%undo translation
                j = w - patchSize;
            end
            
            for chls=1:size(data,2)
                temp_data(nrSamples,chls,:,:)=data{chls}(i:i+patchSize-1,j:j+patchSize-1);
            end
            temp_mask(nrSamples,:,:)  =m(i:i+patchSize-1,j:j+patchSize-1);
            temp_information(nrSamples,:)=[i j 0];
            nrSamples=nrSamples+1;
        end
    end
temp_data=temp_data(1:nrSamples-1,:,:,:); 
temp_mask=temp_mask(1:nrSamples-1,:,:,:); 
temp_information=temp_information(1:nrSamples-1,:,:,:);      
end

