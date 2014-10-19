function [F, D, speedup_struct] = pooling_local_feats_v3(I, type, masks, pb_path, pars, speedup_struct_in, SPEEDUP)
    %global precomp_shape_feats;
    
    N_SHAPE_DIMS = 8;
    DefaultVal('*speedup_struct_in', '[]');
    DefaultVal('*SPEEDUP', 'true');

    if(~isfield(pars, 'sum_type') || isempty(pars.sum_type) || ~isfield(pars, 'sp_app'))
        SPEEDUP = false;
        pars.sp_app = [];
    end
    
    speedup_struct = [];
    no_speedup_in = isempty(speedup_struct_in);
    
    if(~isfield(pars, 'conditioning_sigma'))
        pars.conditioning_sigma = 0.001;
    end

    if(~isempty(pars.enrichments) && ~iscell(pars.enrichments{1}))
        pars.enrichments = {pars.enrichments};
    end
         
    if(~isfield(pars, 'pca_basis'))
        pca_basis = [];
    end
    
    if(isempty(speedup_struct_in))
        [D,F, feat_ranges, variable_grids] = compute_shape_invariant_feats(I, pars.main_feat, pars.enrichments, pars.mode, pars.color_type, pca_basis, pars.STEP, pars.base_scales, [], [], pb_path);
        speedup_struct.F = F;
        speedup_struct.feat_ranges = feat_ranges;
        speedup_struct.variable_grids = variable_grids;
        speedup_struct.D = D;
    else
        F = speedup_struct_in.F;
        feat_ranges = speedup_struct_in.feat_ranges;
        variable_grids = speedup_struct_in.variable_grids;
        D = speedup_struct_in.D;
        nfeats_in_superp = speedup_struct_in.nfeats_in_superp;
    end
    
    pars.pooling_weights = ones(1,size(F,2));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %t_speed = tic();
    all_masks = [];
    all_sp_app = logical([]);
    counter = 1;

    for a=1:numel(pars.fg)
        [new_masks, new_sp_app] = process_masks(masks, pars.fg{a}, pars.sp_app);
        range_masks{a} = counter:counter+size(new_masks,3)-1;
        all_masks = cat(3, logical(all_masks), new_masks);
        all_sp_app = [all_sp_app new_sp_app];
        counter = counter + size(new_masks,3);
    end
    clear new_masks;
    pars.sp_app = all_sp_app;
    
    if SPEEDUP
        if(no_speedup_in)
            if(strcmp(pars.sum_sp_type, 'ucm') && ~isfield(pars, 'sp'))
                error('ucm superpixels should be precomputed and include in pars as .sp field');
            end

            [mask_spds, ~, feats_in_masks, superp_spds, nfeats_in_superp] = speedup_structs_approx(F, all_masks, D, pars, variable_grids, pars.sum_approx, pars.sum_type, pars.sum_sp_type, I);        
        else
            mask_spds = speedup_struct_in.mask_spds;
            feats_in_masks = speedup_struct_in.feats_in_masks;
            superp_spds =  speedup_struct_in.sp_spds;
        end
    else
        lin_ids = sub2ind([size(all_masks,1) size(all_masks,2)], F(2,:), F(1,:));
        feats_in_masks = false(size(all_masks,3), size(F,2));
        for b=1:size(all_masks,3)
            m = all_masks(:,:,b);
            feats_in_masks(b,:) = m(lin_ids);
        end
    end

    all_D = [];
    for a=1:numel(pars.fg)    
        n_shape_varying_feats = 0;
        xy = false;
        scale = false;
        shape = false;
        internal_shape = false;
        extra_xy = false;
        
        for b=1:numel(pars.enrichments{a})
            if(strcmp(pars.enrichments{a}{b}, 'xy'))
                xy = true;
            elseif(strcmp(pars.enrichments{a}{b}, 'extra_xy'))
                extra_xy = true;
            elseif(strcmp(pars.enrichments{a}{b}, 'scale'))
                scale = true;
            elseif(strcmp(pars.enrichments{a}{b}, 'shape'))
                shape = true; % not used in default implementation, slow
            elseif(strcmp(pars.enrichments{a}{b}, 'internal_shape'))
                internal_shape = true; % not used in default implementation, it's more of an open slot than really anything concrete
            end
        end

        if(xy)
            n_shape_varying_feats = n_shape_varying_feats + 4;
        end

        if(extra_xy)
            n_shape_varying_feats = n_shape_varying_feats + 16;
        end
        
        if(scale)
            n_shape_varying_feats = n_shape_varying_feats + 2;
        end

        if(shape)
            n_shape_varying_feats = n_shape_varying_feats + 2*N_SHAPE_DIMS;
        end

        if(internal_shape)
            n_shape_varying_feats = n_shape_varying_feats + 2*N_SHAPE_DIMS;
        end
        
        speedup_struct.n_shape_varying_feats(a) = n_shape_varying_feats;

        masks = all_masks(:,:,range_masks{a});

        feat_dim = max(feat_ranges{a});

        total_dims = feat_dim + n_shape_varying_feats;
        
        duh = true(total_dims);
        in_triu = triu(duh);
        if(~strcmp(pars.pooling_type, 'avg') && ~strcmp(pars.pooling_type, 'max'))
            % second order pooling
            N_DIMS = (total_dims^2 + total_dims)/2;
        else
            % first order pooling
            N_DIMS = total_dims;
        end

        n_masks = size(masks,3);

        finalD = zeros(N_DIMS, n_masks, 'single');

        bbox = zeros(size(masks,3), 4);   
        F1 = cell(1,size(masks,3));
        jump = 1;

        mask_counter = 1;
        for i=1:jump:size(masks,3)
            %the_time = tic();             
            in_mask = feats_in_masks(range_masks{a}(i),:);
            in_mask_id = find(in_mask);
            F1{i} = single(F(:,in_mask));

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%% select local features to pool over %%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            new_feats_in_masks = true(numel(in_mask_id),1);

            global_new_feats_in_masks = false(1, size(feats_in_masks,2));
            global_new_feats_in_masks(in_mask_id(new_feats_in_masks)) = true;   
            
            F1{i} = single(F1{i}(:, new_feats_in_masks));
            if 0
                min(F1{i}(4,:))
                sc(I); hold on;
                F_to_show = F1{i};
                F_to_show(3,:) = F_to_show(4,:)*2;
                F_to_show(4,:) = [];
                vl_plotframe(F_to_show)
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            [new_D, bbox(i,:)] = compute_shape_varying_feats(masks(:,:,i), F1{i}, ...
                xy, scale, shape, internal_shape, extra_xy, N_SHAPE_DIMS, variable_grids, pars.base_scales, I);%, D(1:128,in_mask_id(new_feats_in_masks)));
            if(n_shape_varying_feats>0 || ~SPEEDUP)                           
                D1 = zeros(numel(feat_ranges{a})+n_shape_varying_feats, size(F1{i},2), 'single');
                D1(1:numel(feat_ranges{a}),:) = D(feat_ranges{a},in_mask_id(new_feats_in_masks));
                if(~isempty(new_D))
                    D1((feat_dim+1):(feat_dim+n_shape_varying_feats),:) = new_D;  
                end
            else                
                D1 = [];
            end
            
            %ind = feats_in_masks(range_masks{a}(i),:);
            
            if(numel(in_mask_id)>1)
                if(SPEEDUP)
                    if(n_shape_varying_feats==0)
                        D1 = [];
                    end
                    finalD(:,mask_counter) = pool_features_fast(pars.pooling_type, D1, in_triu, pars.conditioning_sigma, pars.pooling_weights(in_mask_id), SPEEDUP, mask_spds(feat_ranges{a},feat_ranges{a},range_masks{a}(i)));
                else
                    finalD(:,mask_counter) = pool_features_fast(pars.pooling_type, D1, in_triu, pars.conditioning_sigma, pars.pooling_weights(in_mask_id), false);
                end
            end
            mask_counter = mask_counter + 1;
            %toc(the_time)
        end 
        %time_summing = toc(t1)

        all_D = [all_D; finalD];
    end
    D = all_D;
    
    if SPEEDUP        
        speedup_struct.sp_spds = superp_spds;
        speedup_struct.mask_spds = mask_spds;
        speedup_struct.feats_in_masks = feats_in_masks;
        speedup_struct.nfeats_in_superp = nfeats_in_superp;
    end
    
end
