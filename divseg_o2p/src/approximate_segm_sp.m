function [sp_apps, sp_masks] = approximate_segm_sp(sp, masks, MAX_FRACTION_OUT, MIN_FRACTION_OBJ_IN)
    % approximate segments by superpixels (remove those that
    % have more than MAX_FRACTION_OUT% of their elements outside the segment)
    DefaultVal('*MAX_FRACTION_OUT', '0.7');
    DefaultVal('*MIN_FRACTION_OBJ_IN', '0.1');
    
    un_sp = unique(sp);
    sp_apps = false(size(un_sp,1), size(masks,3));
    sp_masks = false(size(masks));
    
    sp_area = regionprops(sp, 'Area');    
    sp_area = [sp_area.Area]';
       
    obj_area = zeros(size(masks,3),1);
    for i=1:size(masks,3)
        obj_area(i) = sum(sum(masks(:,:,i)));
    end
        
    bin = labels2binary(sp);
    bin = reshape(bin, [size(sp) size(bin,2)]);
    bin = ndSparse(bin);
    
    for i=1:size(masks,3)
        % how many pixels in mask
        count_in = histc(sp(masks(:,:,i)), 1:numel(un_sp));      
        
        if(size(count_in, 2)>size(count_in, 1))
            count_in = count_in';
        end
        sp_apps(:,i) = count_in>(sp_area*MAX_FRACTION_OUT);
        sp_apps(:,i) = sp_apps(:,i) | (count_in>(obj_area(i)*MIN_FRACTION_OBJ_IN) & count_in>(sp_area*0.3)); % black magic 
        
        sp_masks(:,:,i) = sum(bin(:,:,sp_apps(:,i)),3);
    end
end