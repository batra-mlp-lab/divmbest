function o2p_project_pca_noncent(exp_dir, img_names, mask_type_train, mask_type_test, feat_collection, N_TO_KEEP)
    [feats, power_scaling, scaling_type, weights] = feat_config(feat_collection);
    selection = 1:numel(feats);

    for h=1:numel(selection)
        if(isempty(scaling_type))
            basis_file = [exp_dir 'DIM_REDUC/pca_basis_' mask_type_train '_'  feats{selection(h)} '_nonorm_noncent.mat'];
        else
            basis_file = [exp_dir 'DIM_REDUC/pca_basis_' mask_type_train '_'  feats{selection(h)} '_noncent.mat'];
        end

        load(basis_file);
                
        if(~exist('range_split', 'var'))
            range_split = {1:size(pca_basis,1)};
        end
        
        real_n_to_keep = ceil(N_TO_KEEP(selection(h))/numel(range_split));
        pca_basis = pca_basis(:,1:min(size(pca_basis,2),real_n_to_keep));
        
        if(strcmp(scaling_type, 'norm_2'))
            the_dir = [exp_dir 'MyMeasurements/' mask_type_test '_'  feats{selection(h)} '_pca_' int2str(N_TO_KEEP(selection(h))) '_noncent/'];
        elseif(isempty(scaling_type))
            the_dir = [exp_dir 'MyMeasurements/' mask_type_test '_'  feats{selection(h)} '_pca_' int2str(N_TO_KEEP(selection(h))) '_nonorm_noncent/'];
        else
            error('not ready');
        end
        
        if(~exist(the_dir, 'dir'))
            mkdir(the_dir);
        end
        
        parfor (i=1:numel(img_names))
        %for (i=1:numel(img_names))
            if(~exist([the_dir img_names{i} '.mat'], 'file'))                
                D = myload([exp_dir 'MyMeasurements/' mask_type_test '_' feats{selection(h)} '/' img_names{i} '.mat'], 'D');
                
                if(~isempty(D))
                    if(~isempty(scaling_type))
                        D = scale_data(D, 'norm_2'); % scale individual descriptor
                    end
                    
                    D = D*weights(h);

                    if(~isempty(scaling_type))
                        D = D./sqrt(sum(weights.^2)); % ensure global norm_2 = 1                        
                    end
                    
                    if(power_scaling)
                        D = squash_features(D, 'power');
                    end
                    
                    newD = cell(numel(range_split),1);
                    for k=1:numel(range_split)
                        newD{k} = project_pca(D(range_split{k},:), {pca_basis(range_split{k},:)});
                    end
                    D = cell2mat(newD);
                end
                
                mysave([the_dir img_names{i} '.mat'], 'D', D);
            end
        end
    end           
