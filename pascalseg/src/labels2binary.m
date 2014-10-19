% function binary = labels2binary(labels)
%   n_labels = length(unique(labels));
%   binary = zeros(length(labels),n_labels);
%   
%   for i=1:n_labels
%     binary((labels==i),i) = 1;
%   end

function binary = labels2binary(labels)
  un_l = unique(labels);
  n_labels = length(un_l);
  
  binary = false(numel(labels),n_labels);
  for i=1:n_labels     % this is as fast as it gets
      binary(labels == un_l(i), i) = true;
  end
end
