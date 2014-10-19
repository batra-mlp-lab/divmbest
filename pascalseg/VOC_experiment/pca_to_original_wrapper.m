function [beta] = pca_to_original_wrapper(exp_dir, beta, pca_training_mask_type, feats, ndims)
    DefaultVal('*caching', 'false');
    counter = 0;
    for i=1:numel(feats)
        var.pca_basis = load([exp_dir 'DIM_REDUC/pca_basis_' pca_training_mask_type '_' feats{i} '_noncent.mat']);
       
        range = counter+1:counter+ndims(i);
        
        new_beta{i} = pca_to_original(var.pca_basis, beta(:, range));
        
        counter = counter + ndims(i);
        clear var;
    end
    beta = cell2mat(new_beta);
end