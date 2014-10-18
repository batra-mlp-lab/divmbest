function o2p_pca_noncent(exp_dir, mask_type_train, feat_collection)
    imgset_train = {'all_gt_segm_mirror', 'all_gt_segm'};

    [feats, power_scaling, scaling_type, weights, dim_div] = feat_config(feat_collection);
    selection = 1:numel(feats);

    for h=1:numel(selection)
        D = [];

        if(isempty(scaling_type))
            basis_file = [exp_dir 'DIM_REDUC/pca_basis_' mask_type_train '_'  feats{selection(h)} '_nonorm_noncent.mat'];
        else
            basis_file = [exp_dir 'DIM_REDUC/pca_basis_' mask_type_train '_'  feats{selection(h)} '_noncent.mat'];
        end

        if(~exist([exp_dir 'DIM_REDUC/'], 'dir'))
            mkdir([exp_dir 'DIM_REDUC/']);
        end

        if(~exist(basis_file, 'file'))
            Feats = [];
            for i=1:numel(imgset_train)
                browser_train = SegmBrowser(exp_dir, mask_type_train, imgset_train{i});

                [Feats1, dims] = browser_train.get_whole_feats(1:numel(browser_train.whole_2_img_ids), feats(selection(h)), scaling_type, weights(selection(h)));

                if(strcmp(scaling_type, 'norm_2'))
                    Feats = [Feats Feats1./sqrt(sum(weights.^2))]; % ensure global l2 norm = 1
                else
                    Feats = [Feats Feats1];
                end
            end
            
            clear Feats1;
            if(power_scaling)
                Feats = squash_features(Feats, 'power');
            end

            if(isempty(dim_div{selection(h)}))
                n_splits = 1;
                range_split = {1:size(Feats,1)};
            else
                n_splits = numel(dim_div{selection(h)});
                counter = 1;
                for k=1:n_splits
                    range_split{k} = counter:counter+dim_div{selection(h)}(k)-1;
                    counter = counter + dim_div{selection(h)}(k);
                end
            end

            latent = [];
            pca_basis = [];
            for k=1:n_splits
                t = tic();
                [pca_basis{k},~,latent{k}] = princomp_noncent(Feats(range_split{k},:)', 'econ');
                toc(t)
            end
            
            min_j = min(cellfun(@(a) size(a,2), pca_basis));
            for k=1:n_splits
                pca_basis{k} = pca_basis{k}(:,1:min_j);
                latent{k} = latent{k}(1:min_j);
            end
            pca_basis = cell2mat(pca_basis');

            save(basis_file, 'pca_basis', 'latent', 'range_split');
        end
    end
end