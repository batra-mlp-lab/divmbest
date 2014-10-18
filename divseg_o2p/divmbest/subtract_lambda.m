function scores = subtract_lambda (nPBM, lambda, scores, global_ids, labels, whole_ids, THRESH, type)
  % subtract lambda from segment labels that were chosen in previous solution, set the background label to
  % THRESH and reduce THRESH by lambda for those segments that were assigned to background 
  n = hist (nPBM.whole_2_img_ids (whole_ids), numel(nPBM.img_names));
  whole_ids_cell = mat2cell (whole_ids, 1, n);
  
  scz = size (scores);

  for i = 1:length (n),
    foresegids = global_ids{i};
    backsegids = setdiff (whole_ids_cell{i}, foresegids); 
    foreindx = sub2ind (scz, labels{i}, foresegids);
   
    scores(21,:) = THRESH;
    if(strcmp(type, 'divmbest'))
	scores(foreindx) = scores(foreindx) - lambda;
	if ~isempty(backsegids),
	      scores(21,backsegids) = scores(21,backsegids) - lambda;
	end
    elseif(strcmp(type, 'perturb'))
	U = rand(scz);
	gumbel = log(-log(U));
	gumbel = lambda.*gumbel;
	scores = scores - gumbel; % Domain agnostic perturbation
	% scores(1:20,:) = scores(1:20,:) - gumbel(1:20,:); % Domain aware perturbation
    end
  end
end

