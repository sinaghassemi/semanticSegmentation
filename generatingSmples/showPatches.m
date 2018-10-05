function [ output_args ] = showPatches( im,infrm,patchSize)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
NrSamples = size(infrm,1);
for kk = 1:NrSamples
 lx=infrm(kk,1);
 ly=infrm(kk,2);
 im(max(lx-1,1):lx+1           ,ly:ly+patchSize-1            ,:)=0;
 im(lx+patchSize-2:lx+patchSize,ly:ly+patchSize-1            ,:)=0;
 im(lx:lx+patchSize-1          ,max(ly-1,1):ly+1             ,:)=0;
 im(lx:lx+patchSize-1          ,ly+patchSize-2:ly+patchSize  ,:)=0;
 im(max(lx-1,1):lx+1           ,ly:ly+patchSize-1            ,1)=255;
 im(lx+patchSize-2:lx+patchSize,ly:ly+patchSize-1            ,1)=255;
 im(lx:lx+patchSize-1          ,max(ly-1,1):ly+1             ,1)=255;
 im(lx:lx+patchSize-1          ,ly+patchSize-2:ly+patchSize  ,1)=255;
end
imshow(im)

end

