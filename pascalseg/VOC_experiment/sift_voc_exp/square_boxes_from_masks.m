function [the_bboxes] = square_boxes_from_masks(masks, I, IND_MARGIN)
    DefaultVal('*IND_MARGIN', '10');
    n_masks = size(masks,3);
    the_bboxes = zeros(4, n_masks);    
    
    for k=1:n_masks    
        % creates fixed aspect ratio bounding boxes (setting it as  a square here)
        % bbox -- (ytop,ybottom,xleft,xright)
        
        %center = [(bbox(1)+bbox(2)) (bbox(3)+bbox(4))]./2;
        %imshow(I); hold on;plot(center(:,1), center(:,2), 'o')
        
        props = regionprops(double(masks(:,:,k)), 'BoundingBox');     
        if(isempty(props))
            bbox = [1 2 3 4];
        else            
            bbox(1) = props.BoundingBox(2); %ymin
            bbox(2) = bbox(1) + props.BoundingBox(4); %ymax
            bbox(3) = props.BoundingBox(1); % xmin
            bbox(4) = bbox(3) + props.BoundingBox(3); % xmax
            bbox = round(bbox);
        end
        
        % adds some extra space
        MARGIN = [IND_MARGIN IND_MARGIN IND_MARGIN IND_MARGIN];        
        bbox(1) = max(bbox(1) - MARGIN(1), 1);
        bbox(2) = min(bbox(2) + MARGIN(2), size(I,1));
        bbox(3) = max(bbox(3) - MARGIN(3), 1);
        bbox(4) = min(bbox(4) + MARGIN(4), size(I,2));
             
        yaxis = bbox(2) - bbox(1);
        xaxis = bbox(4) - bbox(3);
        
        to_add = ceil(abs(xaxis-yaxis)/2);
        if(xaxis>yaxis)
            bbox(2) = bbox(2)+to_add;
            bbox(1) = bbox(1)-to_add;
        else
            bbox(4) = bbox(4)+to_add;
            bbox(3) = bbox(3)-to_add;
        end
        
        the_bboxes(:,k) = bbox';
    end
end