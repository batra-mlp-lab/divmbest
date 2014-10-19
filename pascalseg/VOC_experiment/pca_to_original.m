function new_vectors = pca_to_original(pca_basis, vectors)    
    if(~isfield(pca_basis, 'range_split'))
        new_vectors = vectors*pca_basis.pca_basis(:,1:size(vectors,2))';
    else
        new_vectors_split = [];
        range_split = pca_basis.range_split;
        n_per_split = size(vectors,2)/numel(range_split);
        counter_split = 0;
        for j=1:numel(range_split)
            this_range = counter_split+1:(counter_split+n_per_split);
            new_vectors_split{j} = vectors(:, this_range)*pca_basis.pca_basis(pca_basis.range_split{j},1:n_per_split)';
            counter_split = counter_split+n_per_split;
        end

        new_vectors = cell2mat(new_vectors_split);                
        % verify
        if 0
            newD = [];
            for k=1:numel(range_split)
                newD{k} = project_pca(new_vectors(:,pca_basis.range_split{k})', {pca_basis.pca_basis(pca_basis.range_split{k},:)});
                newD{k} = newD{k}(1:n_per_split,:);
            end
            newD = cell2mat(newD')';                        
            mean(mean(abs(newD - vectors)))
        end
    end        
    
    if(isfield(pca_basis, 'the_mean'))
        new_vectors = bsxfun(@plus, new_vectors, pca_basis.the_mean');
    end
end