function K = compute_K_train(browser, MAX_ELEMS, feats, weights, scaling_type, PowerNorm, whole_ids)    
    DefaultVal('*whole_ids', '[]');    
    
    if(isempty(whole_ids))
        whole_ids = 1:numel(browser.whole_2_img_ids);
    end

    [Feats_one_sample, dims] = browser.get_whole_feats(whole_ids(1), feats, scaling_type, weights);
    K = zeros(numel(whole_ids), numel(whole_ids), 'single');

    total_dims = sum(dims);
    total_elems = numel(whole_ids)*total_dims;
    n_chunks = ceil(total_elems/MAX_ELEMS);

    if(size(whole_ids,1) > size(whole_ids,2))
        whole_ids = whole_ids';
    end
    
    whole_ids_chunks = chunkify(whole_ids, n_chunks);
    K_whole_ids_chunks = chunkify(1:numel(whole_ids), n_chunks);
    
    for i=1:n_chunks
        [Feats1, dims] = browser.get_whole_feats(whole_ids_chunks{i}, feats, scaling_type, weights);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(PowerNorm)
            Feats1 = squash_features(Feats1, 'power');        
        end

        for j=i:n_chunks
            % libsvm dual
            if(i~=j)
                [Feats2, dims] = browser.get_whole_feats(whole_ids_chunks{j}, feats, scaling_type, weights);
                if(PowerNorm)
                    Feats2 = squash_features(Feats2, 'power');
                end
            else
                Feats2 = Feats1;
            end
            
            K(K_whole_ids_chunks{i}, K_whole_ids_chunks{j}) = Feats1'*Feats2;

            if(i~=j)
                K(K_whole_ids_chunks{j},K_whole_ids_chunks{i}) = K(K_whole_ids_chunks{i}, K_whole_ids_chunks{j})';
            end
        end
    end
end

