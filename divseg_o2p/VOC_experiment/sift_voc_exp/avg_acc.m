function a = avg_acc(labels, pred_labels, n_labels)
  un_labels = unique(labels);
  
  for i=1:numel(un_labels)
    a(i) = mean(labels(labels==un_labels(i)) == pred_labels(labels==un_labels(i)));
  end
end