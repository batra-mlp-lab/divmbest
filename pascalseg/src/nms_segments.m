function pick = nms_segments(scores, overlap_thresh, overlap_matrix, MAX_SEGMS)
DefaultVal('*MAX_SEGMS', 'inf');
% pick = nms(boxes, overlap) 
% Non-maximum suppression.
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected detection.

if isempty(scores)
  pick = [];
else
  s = scores;    

  [vals, I] = sort(s);
  pick = [];
  counter = 0;
  while ~isempty(I)    
    last = length(I);
    i = I(last);
    pick = [pick; i];
    suppress = [last];
    for pos = 1:last-1
      j = I(pos);
      % compute overlap
      o = overlap_matrix(i,j);
      if o > overlap_thresh
          suppress = [suppress; pos];
      end
    end
    I(suppress) = [];
    
    counter = counter + 1;
    if(counter==MAX_SEGMS) 
      break;
    end
  end  
end
