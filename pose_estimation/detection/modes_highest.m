function boxes = modes_highest(boxes_modes)
    % for each mode take the highest scoring configuration
    % return these in 2D array in mode order

    nummodes = size(boxes_modes,3);
    boxes = zeros(nummodes,size(boxes_modes,2));
    count = 0;
    for mode = 1:nummodes
        [score level] = max(boxes_modes(:,end,mode));
        if score > -inf
            count = count + 1;
            boxes(count,:) = boxes_modes(level,:,mode);
        end
    end
    
    boxes = boxes(1:count,:);
end

