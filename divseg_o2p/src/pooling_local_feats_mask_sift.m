function [F, D, speedup_struct] = pooling_local_feats_mask_sift(I, type, masks, pb_path, pars, enclosing_masks, speedup_struct_in)
    %global precomp_shape_feats;
    DefaultVal('*enclosing_masks', '[]');       
       
    N_SHAPE_DIMS = 8;
    DefaultVal('*speedup_struct_in', '[]');
    speedup_struct = [];

    if(~isfield(pars, 'conditioning_sigma'))
        pars.conditioning_sigma = 0.001;
    end

    if(~iscell(pars.enrichments{1}))
        pars.enrichments = {pars.enrichments};
    end

    if(~isfield(pars, 'pca_basis'))
        pca_basis = [];
    end

    n_shape_varying_feats = 0;
    n_nonshape_feats = 0;
    xy = false;
    scale = false;
    shape = false;
    internal_shape = false;
    extra_xy = false;
    rgb = false;
    hsv = false;
    lab = false;
    
    a = 1;
    for b=1:numel(pars.enrichments{a})
        if(strcmp(pars.enrichments{a}{b}, 'xy'))
            xy = true;
        elseif(strcmp(pars.enrichments{a}{b}, 'extra_xy'))
            extra_xy = true;
        elseif(strcmp(pars.enrichments{a}{b}, 'scale'))
            scale = true;
        elseif(strcmp(pars.enrichments{a}{b}, 'shape'))
            shape = true;
        elseif(strcmp(pars.enrichments{a}{b}, 'internal_shape'))
            internal_shape = true;
        elseif(strcmp(pars.enrichments{a}{b}, 'rgb'))
            rgb = true;
        elseif(strcmp(pars.enrichments{a}{b}, 'hsv'))            
            hsv = true;
        elseif(strcmp(pars.enrichments{a}{b}, 'lab'))                        
            lab = true;
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
    
    if(rgb)
        n_nonshape_feats = n_nonshape_feats + 3;
    end
    
    if(hsv)
        n_nonshape_feats = n_nonshape_feats + 3;
    end
    
    if(lab)
        n_nonshape_feats = n_nonshape_feats + 3;
    end
    
    variable_grids = false;
       
    if(strcmp(pars.main_feat, 'really_dense_sift'))
        if(strcmp(pars.color_type, 'gray'))
            main_feat_dim = 128;
        else
           main_feat_dim = 384;
        end
    elseif(strcmp(pars.main_feat, 'really_dense_lbp'))
        main_feat_dim = 58;
    end
    
    total_dims = main_feat_dim + n_nonshape_feats + n_shape_varying_feats;        
   
    speedup_struct.feat_ranges = {[]};
    speedup_struct.n_shape_varying_feats = total_dims; % set them all as shape-varying as simplification

    duh = true(total_dims);
    in_triu = triu(duh);
    
    if(isempty(enclosing_masks))
        enclosing_masks = masks;
    end
    
    if(~strcmp(pars.pooling_type, 'avg') && ~strcmp(pars.pooling_type, 'max'))
        % second order pooling
        N_DIMS = (total_dims^2 + total_dims)/2;
    else
        % first order pooling
        N_DIMS = total_dims;
    end
       
    if(pars.spatial_pyr)
        pyramid = [1 2];
        n_masks = size(masks,3);
    else
        n_masks = 1;
        pyramid = 1;
    end
    pgrid = pyramid.^2;
    sgrid = sum(pgrid);
    weights = (1./pgrid); % divided by the number of grids at the coresponding level
    weights = weights/sum(weights);
    
    finalD = zeros(sgrid*N_DIMS, n_masks, 'single');
    
    F = cell(size(masks,3),1);
    D = F;
    for i=1:size(masks,3)        
        % every mask should have K pixels height
        
        assert(numel(pars.fg) == 1);
        mask = process_masks(masks(:,:,i), pars.fg{1});
        
        K = 75;        
        [a,b] = find(mask);
        max_a = max(a);
        max_b = max(b);
        min_a = min(a);
        min_b = min(b);
        mask = mask(min_a:max_a,min_b:max_b);
        
        if(numel(mask)<2)
            res = 1;
        else            
            res = K/(max_a - min_a);
        end
        
        if(res==inf)
            res = 1;
        end
        
        mask = imresize(mask, res, 'bilinear');
        thisI_orig = imresize(I(min_a:max_a,min_b:max_b,:), size(mask), 'bilinear');                        
    
        fuh = cell(1,numel(pars.base_scales));
        duh = fuh;
        mask = single(mask);
        
        if(pars.do_img_masking)
            GAP = 0.4;  % 0.3/0.4 seem best
            thisI = thisI_orig*(1-GAP) + GAP*255.0;
            theI = single(thisI) .* single(repmat(mask, [ 1 1 size(thisI,3)]));
        else
            theI = single(thisI_orig);
        end
        
        if(strcmp(pars.main_feat, 'really_dense_sift'))
            total_pad = 0;
            theI_pad = im2single(theI/255.0);
            for k=1:numel(pars.base_scales)
                if(k~=1)
                    pad = ceil(1.5*(pars.base_scales(k)-pars.base_scales(k-1)));
                    total_pad = total_pad + pad;
                    if(mod(pars.base_scales(k)-pars.base_scales(k-1), 2) ~= 0)
                        theI_pad = padarray(theI_pad, repmat(pad+1,1,2), 0, 'pre');
                        total_pad = total_pad - 0.5;
                    else
                        theI_pad = padarray(theI_pad, repmat(pad,1,2), 0, 'both');
                    end
                end

                %[fuh{k}, duh{k}] = vl_phow(im2single(theI), 'FloatDescriptors', true, 'Sizes', base_scales(k), 'Step', STEP, 'Magnif', inf, 'Color', color_type, 'ContrastThreshold', 0); % 0
                [fuh{k}, duh{k}] = vl_phow(theI_pad, 'FloatDescriptors', true, 'Sizes', pars.base_scales(k), 'Step', pars.STEP, 'WindowSize', 1.5, 'Magnif', 6, 'Color', pars.color_type, 'ContrastThreshold', 0);

                fuh{k}(1:2,:) = fuh{k}(1:2,:) - total_pad;
            end
            var.F = ceil(cell2mat(fuh));
            F{i} = var.F;
            var.D = cell2mat(duh);
            D{i} = var.D/512.0;        
        elseif(strcmp(pars.main_feat, 'really_dense_lbp'))            
            variable_grids = true;

            if(size(theI,3)==1)
                theI = repmat(theI, [1 1 3]);
            end
            
            counter = 1;
            for j=1:3
                Isingle = theI(:,:,j);
                for k=1:numel(pars.base_scales)
                    Ilbp = vl_lbp(Isingle, pars.base_scales(k));

                    [a,b] = find(Ilbp(:,:,1)~=inf); % just want the index, hope there's never any inf

                    newD = reshape(Ilbp, numel(a), size(Ilbp,3))';

                    duh{k} = [duh{k} newD];
                    a = a*pars.base_scales(k);
                    b = b*pars.base_scales(k);
                    fuh{k} = [b'; a'; pars.base_scales(k)*ones(1,numel(a))];                
                    counter = counter + 1;
                end
            end

            to_ignore = cellfun(@isempty, fuh);
            F{i} = cell2mat(fuh(~to_ignore));
            D{i} = cell2mat(duh(~to_ignore));
            if(~isempty(D{i}))                
                D{i} = scale_data(D{i}, 'norm_1');
            end            
        end
        
        if(size(F{i},2)~=0)
            % select those descriptors inside the mask        
            in_mask = logical(mask(sub2ind(size(mask), F{i}(2,:), F{i}(1,:))));

            F{i} = F{i}(:,in_mask);
            D{i} = D{i}(:,in_mask);

            D{i} = sqrt(D{i});                        
            if(~isempty(F{i})) 
                [newD,newF, feat_ranges] = compute_shape_invariant_feats(uint8(thisI_orig), {}, pars.enrichments, pars.mode, pars.color_type, pca_basis, pars.STEP, pars.base_scales, F{i}, variable_grids);
                [new_D_shape, bbox] = compute_shape_varying_feats(mask, F{i}, ...
                        xy, scale, shape, internal_shape, extra_xy, N_SHAPE_DIMS, variable_grids, pars.base_scales, theI);

                D{i} = [D{i}; newD; new_D_shape];
            end            
        else
            bbox = [1 1 1 1];
        end
        
        pars.pooling_weights = ones(1,size(F{1},2));
        
        %masks = all_masks(:,:,range_masks{a});
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% select local features to pool over %%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        counter = 1;
        y = F{i}(1,:);
        x = F{i}(2,:);
        y = y - min(y) +1 ;
        x = x - min(x) + 1;
        
        for s = 1:length(pyramid)
            wleng = bbox(3)/pyramid(s);
            hleng = bbox(4)/pyramid(s);
            xgrid = ceil(x/wleng);
            ygrid = ceil(y/hleng);
            allgrid = (ygrid -1 )*pyramid(s) + xgrid;
            
            for t = 1:pgrid(s)                
                ind = find(allgrid == t);
                range = counter:counter+N_DIMS-1;
                
                finalD(range,i) = pool_features_fast(pars.pooling_type, D{i}(:,ind), in_triu, pars.conditioning_sigma, ones(1,numel(ind)), false);
                finalD(range,i) = weights(s) * finalD(range,i);
            end
            
            counter = counter + N_DIMS;
        end
    end    
    
    D = finalD;    
    F = []; % save space
end
