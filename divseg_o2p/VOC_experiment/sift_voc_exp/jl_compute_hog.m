function D = jl_compute_hog(I, masks, nbins, space_around)
    DefaultVal('*space_around', '10');
    DefaultVal('*nbins', '[10 10]');
    DefaultVal('*dilat_width', '10');    
    
    artif_nbins = nbins+2; % to get it right
    
    sbin = 8;
    bg_color = 255;
    
    D = zeros(nbins(1)*nbins(2)*32,size(masks,3));            

    [bbox] = square_boxes_from_masks(masks, I, space_around);

    sz_Img = size(I);
    up = abs(min(0,min(bbox(1,:))));
    down = max(0,max(bbox(2,:)) - sz_Img(1));
    left = abs(min(0,min(bbox(3,:))));
    right = max(0,max(bbox(4,:)) - sz_Img(2));
    I = grow_it(I, up, down, left, right);
    masks = grow_it(masks, up, down, left, right);
    bbox(1,:) = bbox(1,:) + up + 1;
    bbox(2,:) = bbox(2,:) + up;
    bbox(3,:) = bbox(3,:) + left + 1;
    bbox(4,:) = bbox(4,:) + left;

    for i=1:size(masks,3)
        mask = masks(:,:,i);

        %%% resize image in bounding box to have square aspect ratio            
        %range1 = round(max(1,bbox(2)-space_around):min(size(I,1), bbox(2)+bbox(4)+space_around));
        %range2 = round(max(1,bbox(1)-space_around):min(size(I,2), bbox(1)+bbox(3) +space_around));
 
        range1 = round(max(1,bbox(1, i)):min(size(I,1), bbox(2, i)));
        range2 = round(max(1,bbox(3, i)):min(size(I,2), bbox(4, i)));

        mask = mask(range1, range2);
        I_cm = I(range1, range2,:);

        if 1 % to mask background 
            I_cm = mask_img(I_cm, mask, dilat_width, bg_color);
        end

        dim = sbin*max(artif_nbins);        
        I_cm = imresize(I_cm, [dim dim]);
        
        % now if it's not square pick the cells to match the right
        % dimension

        % HOG computed using TTI-C package of part-based models
        
        try
            f = features(double(I_cm), sbin);
        catch
            disp(['If you want to run this code, get the HOG implementation' ...
                 ' in the "features.cc" file from the TTI-C part-based ' ...
                 'detector package and compile it.']);
        end
        
        if(~(nbins(1)==nbins(2)))
            % must be even
            assert(mod(max(nbins) - min(nbins), 2)==0);
            
            % how many cells to remove from each side ?
            n_to_remove = (max(nbins) - min(nbins))/2;
            
            if(nbins(1)>nbins(2))
                f(1:n_to_remove,:,:) = [];
                f(end-n_to_remove+1:end,:,:) = [];
            else
                f(:,1:n_to_remove,:) = [];
                f(:,end-n_to_remove+1:end,:) = [];
            end
        end
        
        %sc(I_cm);
        %figure;
        %visualizeHOG(f);

        f = f(:);
        D(:,i) = f;
    end
    D = single(D);
end

function I_cm = mask_img(I_cm, mask, width, bg_color)
    mask = imdilate(mask, ones(width,width));
    for j=1:3
        duh = I_cm(:,:,j);
        duh(~mask) = bg_color;
        I_cm(:,:,j) = duh;
    end
end


function newI = grow_it(I, up, down, left, right)
    new_size = [(size(I,1) + up + down) size(I,2) + left + right size(I,3)];
    
    newI = zeros(new_size);
    newI(up+1:end-down, left+1:end-right,:) = I;
end
