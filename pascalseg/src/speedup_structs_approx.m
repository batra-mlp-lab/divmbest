function [pooled_spd, mask_superp_tmp, superp_ids_in, superp_spd, nfeats_in_superp] = speedup_structs_approx(F, masks, D, pars, variable_grids, MIN_SP_FEATS, sum_type, sum_sp_type, I)
    if(~variable_grids)
        range_unique = 1:size(F,2)/numel(pars.base_scales);
    else
        range_unique = 1:size(F,2);
    end

    % check how many
    lin_ids = sub2ind([size(masks,1) size(masks,2)], F(2,range_unique), F(1,range_unique));
    ids_in = false(size(masks,3), numel(range_unique));
    for i=1:size(masks,3)
        m = masks(:,:,i);
        ids_in(i,:) = m(lin_ids);
    end

    if(strcmp(sum_sp_type, 'slic'))
        if(MIN_SP_FEATS ~=0)
            spI = vl_slic(im2single(I), MIN_SP_FEATS, 0.01);
            superpixel_ids = double(spI(lin_ids))';
        else
            superpixel_ids = lin_ids';
        end
    elseif(strcmp(sum_sp_type, 'ucm'))
        % should come already in the params
        spI = pars.sp;
        superpixel_ids = spI(lin_ids)';
    else
        error('no such superpixel type');
    end
    
    % Approximate segments by superpixels, except if we have done that already.
    % Approximation simply removes those superpixels having more than 70% of their features outside the
    % segment.
    MAX_FRACTION_OUT = 0.7;
    un_sp = unique(spI);
        
    if(~isfield(pars, 'sp_app'))
        mask_superp = false(numel(un_sp), size(masks,3));
    else
        mask_superp = pars.sp_app;
    end
    
    nfeats_in_superp = zeros(size(mask_superp,1),1);
    superp_ids_in = false(size(ids_in));
    for i=1:numel(un_sp)
        sp_equal = superpixel_ids==un_sp(i);
        
        if(~variable_grids)
            nfeats_in_superp(i) = sum(sp_equal)*numel(pars.base_scales);
        else
            nfeats_in_superp(i) = sum(sp_equal);
        end
                        
        if(~isfield(pars, 'sp_app'))
            s = ids_in(:, sp_equal);
            sum_s = sum(s,2);        
            mask_superp(i,:) = sum_s>(size(s,2)*MAX_FRACTION_OUT);
        end
        
        superp_ids_in(mask_superp(i,:),sp_equal) = true;
    end       
    
    if 0 % visualize superpixels one by one
        un_sp = unique(superpixel_ids);
        sc(I); hold on;
        for i=1:numel(un_sp)
            plot( F(1,range_unique(superpixel_ids==i)), F(2,range_unique(superpixel_ids==i)), 'o', 'Color', rand(1, 3)); hold on;
            pause;
        end
    end

    if 0
        % visualize segment approximation by superpixels
        un_sp = unique(superpixel_ids);
        for i=1:size(mask_superp,2)
            subplot_auto_transparent(masks(:,:,i),I); hold on;
            sp_in = find(mask_superp(:,i));
            for j=1:numel(sp_in)
                plot( F(1,superpixel_ids==un_sp(sp_in(j))), F(2,superpixel_ids==un_sp(sp_in(j))), 'o', 'Color', rand(1, 3)); hold on;
            end
            pause;
        end
    end

    superp_spd = zeros(size(D,1), size(D,1), size(mask_superp,1), 'single');

    if(~variable_grids)
        superpixel_ids = repmat(superpixel_ids, numel(pars.base_scales),1);
        superp_ids_in =  repmat(superp_ids_in, 1, numel(pars.base_scales));
    end
        
    % total number of features in a superpixel approximation of a mask should be the same as the
    % sum of the number of features in the internal superpixels
    assert(sum(nfeats_in_superp(mask_superp(:,1))) == sum(superp_ids_in(1,:)))

    D = bsxfun(@times, D, sqrt(pars.pooling_weights));
    for i=1:size(mask_superp,1)
        % potential speed-up
        %if(strcmp(pars.pooling_type, 'log_avg'))
        %    all_outer_prod = multiprod(reshape(D, [1 size(D)]), reshape(D,[size(D,1) 1 size(D,2)]), [0 1], [1 0]);
        %end
        %superp_spd(:,:,i) = sum(all_outer_prod(:,:,superpixel_ids==i),3);
        to_include = superpixel_ids==un_sp(i);
        duh = D(:,to_include);
        superp_spd(:,:,i) = duh*duh';
    end

    n_sp = sum(mask_superp, 1);

    pooled_spd = zeros(size(D,1), size(D,1), size(masks,3), 'single');
    [~, sp_srt] = sort(n_sp, 'ascend');
    n_sums = 0;
    
    total_n_sp = size(mask_superp,1);
    mask_superp_tmp = mask_superp;    
                
    if strcmp(sum_type, 'cache_sp') 
        for i=1:size(masks,3)
            pooled_spd(:,:,i) = sum(superp_spd(:,:,mask_superp_tmp(:,i)),3);
        end
    elseif(strcmp(sum_type, 'cache_sp_v2'))
        full_sum = sum(superp_spd,3);
        for i=1:size(masks,3)
            if(n_sp(i)>ceil(total_n_sp/2))
                pooled_spd(:,:,i) = full_sum - sum(superp_spd(:,:,~mask_superp_tmp(:,i)),3);
            else
                pooled_spd(:,:,i) = sum(superp_spd(:,:,mask_superp_tmp(:,i)),3);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%% tried more complicated stuff which didn't seem
        %%%%%%%%%%%%%%%%%% to pay off 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    elseif strcmp(sum_type, 'inclusion') 
        %%%%% using inclusions %%%%%
        t = tic();
        n_sums = 0;
        for i=1:size(masks,3)
            this_mask_id = sp_srt(i);
            col = mask_superp_tmp(:,this_mask_id);
            if(sum(col)~=1) % one element
                containers = sum(mask_superp_tmp(col,:)) == sum(col);
                mask_superp_tmp(col, containers) = false;
                
                %sum(col)
                if(sum(col)~=0)
                    this_sum = sum(superp_spd(:,:,col),3);
                    pooled_spd(:,:,containers) = bsxfun(@plus, pooled_spd(:,:,containers), this_sum);
                    n_sums = n_sums + sum(containers) + sum(col);
                end
            else
                pooled_spd(:,:,this_mask_id) = pooled_spd(:,:,this_mask_id) + superp_spd(:,:,col);
                n_sums = n_sums + 1;
                mask_superp_tmp(:, this_mask_id) = false;
            end
        end
    elseif strcmp(sum_type, 'inclusion_v2')
        %%%%% using inclusions %%%%%
        t = tic();        
        full_sum = sum(superp_spd,3);
        for i=1:size(masks,3)
            this_mask_id = sp_srt(i);
            col = mask_superp_tmp(:,this_mask_id);
            if(sum(col)~=1)
                containers = sum(mask_superp_tmp(col,:)) == sum(col);
                mask_superp_tmp(col, containers) = false;
                
                s_col = sum(col);
                if(s_col~=0)
                    if(s_col>ceil(total_n_sp/2))
                        this_sum = full_sum - sum(superp_spd(:,:,~col),3);
                    else
                        this_sum = sum(superp_spd(:,:,col),3);
                    end
            
                    %this_sum = sum(superp_spd(:,:,col),3);
                    pooled_spd(:,:,containers) = bsxfun(@plus, pooled_spd(:,:,containers), this_sum);
                    n_sums = n_sums + sum(containers) + sum(col);
                end
            else % one element
                id = find(col);
                assert(numel(id)==1);
                pooled_spd(:,:,this_mask_id) = pooled_spd(:,:,this_mask_id) + superp_spd(:,:,id);
                n_sums = n_sums + 1;
                mask_superp_tmp(id, this_mask_id) = false;
            end
        end        
    elseif 0
        n_sums_best = 0;
        pooled_spd_tmp = zeros(size(superp_spd,1), size(superp_spd,2),80000, 'single');        
        last_elem = size(pooled_spd,3);
        mask_superp_tmp = [mask_superp_tmp; [false(80000-size(mask_superp_tmp,1), size(mask_superp_tmp,2))]];
        
        t_dp = tic();
        for i=1:size(masks,3)
            i
            this_mask_id = sp_srt(i);
            col = mask_superp_tmp(:,this_mask_id);
            if(sum(col)==1) % one element
                continue
            else
                inters = mask_superp_tmp(col,:);
                these_sp = find(col);
                un = unique(inters', 'rows')';
                
                % add one superpixel for each set of intersections, progressively,
                % starting with the smaller ones, so that later ones are composed
                % of earlier ones
                un_cansum = un(:, sum(un)>1);
                for j=1:size(un_cansum,2)
                    ids = find(sum(bsxfun(@and, inters, un_cansum(:,j))) == sum(un_cansum(:,j)));
                    sps = these_sp(un_cansum(:,j));
                    
                    mask_superp_tmp(last_elem+1, ids) = true;
                    mask_superp_tmp(sps, ids) = false;
                    pooled_spd_tmp(:,:,last_elem+1) = sum(pooled_spd_tmp(:,:,sps),3); % the real sum
                    
                    n_sums_best = n_sums_best + numel(sps);
                    inters(un_cansum(:,j),:) = [];
                    inters = [inters; false(1, size(inters,2))];
                    inters(end, ids) = true;
                    
                    these_sp(un_cansum(:,j)) = [];
                    these_sp = [these_sp; last_elem];
                    
                    last_elem = last_elem + 1;

                    un_cansum(un_cansum(:,j),:) = [];
                    un_cansum = [un_cansum; true(1, size(un_cansum,2))];
                end
            end
        end
        
        n_sums_best = n_sums_best + sum(mask_superp_tmp(:));
        
        savings = n_sums_best/n_sums_naive
    else
        error('no such type');
    end
end