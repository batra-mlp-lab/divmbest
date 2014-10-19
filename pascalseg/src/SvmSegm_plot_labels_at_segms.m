function SvmSegm_plot_labels_at_segms(segms, labels, cmap)
    if(~iscell(labels))
        new_labels = {};
        for i=1:numel(labels)
            new_labels{i} = sprintf('%s', labels{i});
        end
        labels = new_labels;
    end
    
   if((size(segms,3) == 1) && ~islogical(segms))    
       u = unique(segms);
       new_segms = labels2binary(segms);
       segms = reshape(new_segms, size(segms,1), size(segms,2), numel(u));
   end
   
   %hold on
   for i=1:size(segms,3)
       s  = regionprops(segms(:,:,i), 'centroid');
       centroids = cat(1, s.Centroid);
       for j=1:size(centroids,1)
           h = text(centroids(j,1)-10, centroids(j,2), labels{i}, 'FontSize', 25, 'FontWeight', 'bold');
           %set(h, 'FontName', 'Helvetica'); % choose your favorite font
           %type
           if(exist('cmap', 'var'))
               set(h, 'Color', cmap(i,:));
           end
       end
   end
   %plot(centroids(:,1), centroids(:,2), 'b*')
   %hold off
end