function [segm_ids, labels, their_scores, whole_ids_cell] = nms_inference_simplicity_bias(nPBM, scores, type, whole_ids, MAX_OVER, MAX_SEGMS, SIMP_FACTOR, return_bground)
  % the basic idea is that it should be easier adding a first
  % non-background segment than two, and easier two than three, etc.

  DefaultVal('*SIMP_FACTOR', '0.02'); % threshold increases by SIMP_FACTOR from k to k+1 rank

  DefaultVal('*MAX_OVER', '0.5');
  DefaultVal('*MAX_SEGMS', 'inf');
  DefaultVal('*return_bground', 'false');
  if(strcmp(type, 'bbox'))
    boxes = nPBM.get_bboxes_img_ids(1:numel(nPBM.img_names));          
  else
    boxes = [];
  end
  
  BACKGROUND = numel(nPBM.categories); % last one assumed to be background
  
  img_ids = unique(nPBM.whole_2_img_ids(whole_ids));
    
  %n = cellfun(@numel, nPBM.img_2_whole_ids(img_ids));  
  n = hist(nPBM.whole_2_img_ids(whole_ids), numel(nPBM.img_names));
  scores_cell = mat2cell(scores, numel(nPBM.categories), n);
  
  segm_ids = cell(numel(scores_cell),1);
  their_scores = segm_ids;
  labels = segm_ids;

  whole_ids_cell = mat2cell(whole_ids, 1, n);
  
  %parfor i=1:numel(scores_cell)  
  for i=1:numel(scores_cell)
    bground_score = max(scores_cell{i}(BACKGROUND,:));
    [img_scores, these_labels] = max(scores_cell{i},[],1);
    
    if(strcmp(type, 'bbox'))
      boxes1 = [boxes{i} img_scores'];
      the_I = nms(boxes1, MAX_OVER);
    elseif(strcmp(type, 'segment'))
      %nPBM.img_names{img_ids(i)}
      overlap_mat = myload([nPBM.exp_dir 'MyOverlaps/' nPBM.mask_type '/' nPBM.img_names{img_ids(i)} '.mat'], 'overlap_mat');    
      the_I = nms_segments(img_scores, MAX_OVER, overlap_mat, inf);
    end
    
    if 0 && strcmp(nPBM.imgset, 'val') % debug
      range = 1:min(numel(the_I), MAX_SEGMS);
      nPBM.show_wholes(whole_ids_cell{i}(the_I(range)))
      img_scores(the_I(range))
      for j=1:numel(range)
        fprintf('%s ', VOC09_id_to_classname(these_labels(the_I(range(j)))));        
      end
      fprintf('\n');
      figure;
      nPBM.show_best_masks(img_ids(i))
      close all;
    end
    
    if(MAX_SEGMS~=inf)
      if 0
        % MAX number of non-background segments
        c = cumsum(these_labels(the_I) ~= BACKGROUND);
        the_I = the_I(1:min(numel(the_I),find(c==MAX_SEGMS)));
      else
        % MAX number of any segments
        the_I = the_I(1:min(numel(the_I),MAX_SEGMS));
      end            
    end
    
    ;
    
    if(~return_bground)
      % retain only non-background segments    
      the_I(these_labels(the_I)==BACKGROUND) = [];
    end
    
    % simplicity bias
    to_keep = (img_scores(the_I) > [bground_score + [0 SIMP_FACTOR*(1:((numel(the_I)-1)))]]);
    the_I = the_I(to_keep);
    
    segm_ids{i} = the_I;
    whole_ids_cell{i} = whole_ids_cell{i}(the_I);
    
    their_scores{i} = img_scores(the_I);
    labels{i} = these_labels(the_I);
    
    if 0 % debug      
      nPBM.show_wholes(whole_ids_cell{i}(84))
    end
  end
end