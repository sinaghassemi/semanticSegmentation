function [ m ] = fix_mask(m_all,nClasses)
%     Impervious surfaces (RGB: 255, 255, 255)
%     Building (RGB: 0, 0, 255)
%     Low vegetation (RGB: 0, 255, 255)
%     Tree (RGB: 0, 255, 0)
%     Car (RGB: 255, 255, 0)
%     Clutter/background (RGB: 255, 0, 0)
   m=zeros(size(m_all,1), size(m_all,2));
    if nClasses == 1 
        for i=1:size(m,1)
            for j=1:size(m,2)
                color_mask = squeeze(m_all(i,j,:));
                if color_mask(1) == 0 &&  color_mask(2) == 0 &&  color_mask(3) == 255
                    m(i,j)=1;
                end
            end
        end
    else
        for i=1:size(m,1)
            for j=1:size(m,2)
                color_mask = squeeze(m_all(i,j,:));
                if color_mask(1) == 255 &&  color_mask(2) == 255 &&  color_mask(3) == 255     
                    m(i,j)=1;
                elseif color_mask(1) == 0 &&  color_mask(2) == 0 &&  color_mask(3) == 255
                    m(i,j)=2;
                elseif color_mask(1) == 0 &&  color_mask(2) == 255 &&  color_mask(3) == 255
                    m(i,j)=3;
                elseif color_mask(1) == 0 &&  color_mask(2) == 255 &&  color_mask(3) == 0
                    m(i,j)=4;
                elseif color_mask(1) == 255 &&  color_mask(2) == 255 &&  color_mask(3) == 0
                    m(i,j)=5;
                elseif color_mask(1) == 255 &&  color_mask(2) == 0 &&  color_mask(3) == 0
                    m(i,j)=6;
                end
            end
        end
    end

end

