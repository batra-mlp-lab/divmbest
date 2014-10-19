function SvmSegm_extract_measurements_on_masks(mask_type, type, pars, img_names, exp_dir, MAX_DIM, overwrite)    
    DefaultVal('*overwrite', 'false', '*MAX_DIM', 'inf');  
    
    codebook_dir = [exp_dir 'MyCodebooks/'];
    assert(~iscell(type));
    
    if(strcmp(type, 'mask_phog') || strcmp(type, 'back_mask_phog')) % aspect ratio invariant
       type_func = @run_exp_database_do_phog;        
        if(~pars.withpb)
            nopb_str = 'nopb_';
        else 
            nopb_str = '';
        end
        dir_name = [mask_type '_' type '_'  nopb_str int2str(pars.n_ori) '_orientations_' int2str(pars.n_levels) '_levels/'];
    elseif(strcmp(type, 'back_mask_phog_scale_inv') || strcmp(type, 'mask_phog_scale_inv')) % aspect ratio variant
        type_func = @run_exp_database_do_scale_inv_phog;
        dir_name = [mask_type '_' type '_'  int2str(pars.n_ori) '_orientations_' int2str(pars.n_levels) '_levels/'];
    elseif(strcmp(type, 'hog'))
        type_func = @run_exp_database_do_hog;
        dir_name = [mask_type '_' type '/'];    
    elseif(strcmp(type, 'pyramid_hog'))
        type_func = @run_exp_database_do_pyramid_hog;
        dir_name = [mask_type '_' type '/'];            
    elseif(strcmp(type, 'bbox_phog_scale_inv'))
        type_func = @run_exp_database_do_scale_inv_phog;
        dir_name = [mask_type '_' type '_' 'nopb_' int2str(pars.n_ori) '_orientations_' int2str(pars.n_levels) '_levels/'];
    elseif(strcmp(type, 'bbox_phog'))        
        type_func = @run_exp_database_do_phog;
        dir_name = [mask_type '_' type '_' 'nopb_' int2str(pars.n_ori) '_orientations_' int2str(pars.n_levels) '_levels/'];
    elseif(strcmp(type, 'tps_hog'))
        type_func = @run_exp_database_do_tps_hog;
        dir_name = [mask_type '_' type '_'  int2str(pars.n_ori) '_orientations_' int2str(pars.n_levels) '_levels/'];
    elseif(strcmp(type, 'signature'))
        type_func = @run_exp_database_do_signature;
        dir_name = [mask_type '_' type];
    elseif(strcmp(type, 'mask_zernike_moments'))
        type_func = @run_exp_database_do_zernike_moments;
    elseif(strcmp(type, 'mask_regionprops'))
        type_func = @run_exp_database_do_simple;
    elseif(strcmp(type, 'figure_color_histogram'))
        type_func = @run_exp_database_do_ch;
        pars.figure_ground = 'figure';
        dir_name = [mask_type '_' type '/'];
    elseif(strcmp(type, 'ground_color_histogram'))
        type_func = @run_exp_database_do_ch;
        pars.figure_ground = 'ground';
        dir_name = [mask_type '_' type '/'];
    elseif(strcmp(type, 'figure_texton_histogram'))
        type_func = @run_exp_database_do_texton_hist;
        pars.figure_ground = 'figure';
        dir_name =  [mask_type '_' type '/'];
    elseif(strcmp(type, 'simple_segment_feats'))
        type_func = @run_exp_database_do_simple_segment_feats;
        dir_name = [mask_type '_' type '/'];
        pars.pb_dir = [exp_dir 'PB/'];
    elseif(strcmp(type, 'extended_segment_feats'))
        type_func = @run_exp_database_do_extended_segment_feats;
        dir_name = [mask_type '_' type '/'];
        pars.pb_dir = [exp_dir 'PB/'];
    elseif(strcmp(type, 'composition_segment_feats'))
        type_func = @run_exp_database_do_composition_segment_feats;
        dir_name = [mask_type '_' type '/'];

        pars.local_feats_path = [exp_dir 'MyMeasurements'];
        pars.local_feats{1} = 'dense_color_sift_3_scales';
        pars.local_feats{2} = 'dense_sift_4_scales';

        pars.region_feats_path = [exp_dir 'MyFeatures/' mask_type '/'];
        pars.region_feats{1}.figure_file = 'bow_dense_sift_4_scales_figure_300';
        pars.region_feats{1}.ground_file = 'bow_dense_sift_4_scales_ground_300';
        pars.region_feats{2}.figure_file = 'bow_dense_color_sift_3_scales_figure_300';
        pars.region_feats{2}.ground_file = 'bow_dense_color_sift_3_scales_ground_300';
        pars.region_feats{3}.figure_file = 'figure_color_histogram';
        pars.region_feats{3}.ground_file = 'ground_color_histogram';

        error('fix this, don''t want to pass img_list as parameter');
        
      for i=1:length(pars.region_feats)
          load([pars.region_feats_path train_test_val '__' pars.region_feats{i}.figure_file], 'Meas');
          Meas = Meas(img_list);
          pars.region_feats{i}.figure = Meas;

          load([pars.region_feats_path train_test_val '__' pars.region_feats{i}.ground_file], 'Meas');
          Meas = Meas(img_list);
          pars.region_feats{i}.ground = Meas;
          clear Meas;
      end
    elseif(strcmp(type, 'back_mask_local_shape_contexts') || strcmp(type, 'local_shape_contexts_boundary'))
%        codebook_file = [codebook_dir 'kmeans_mask_local_shape_contexts_300_words.mat'];
%         if(exist(codebook_file, 'file'))
%             vars = load(codebook_file, 'codebook');
%             pars.codebook = vars.codebook;
%         else
%             pars.codebook = [];
%         end
        if(~isempty(pars.codebook))
            var = load([exp_dir 'MyCodebooks/' pars.codebook]);            
            pars.codebook = var.codebook;
            dir_name = [mask_type '_' 'bow_' type '/'];
        else
            dir_name = [mask_type '_' type '/'];
        end
        
        type_func = @run_exp_database_do_local_shape_contexts;        
    elseif(strcmp(type, 'ansig'))
        type_func = @run_exp_database_do_ansig;
        dir_name = [mask_type '_' type '/'];
    elseif(strcmp(type, 'scaled_dense_sifts'))
        vars = load([codebook_dir 'kmeans_ground_truth_scaled_dense_sifts_4000_words.mat'], 'codebook');
        pars.codebook = vars.codebook;
        type_func = @run_scaled_dense_sifts;
        dir_name = [mask_type '_' type '/'];
    elseif(strcmp(type, 'idsc_boundary')) % inner distance shape contexts on the boundary
        type_func = @run_idsc_boundary;
        dir_name = [mask_type '_' type '/'];
    elseif(strcmp(type, 'pyramid_lbp'))
        type_func = @pyramid_lbp;
        dir_name = [mask_type '_' type '/'];             
    elseif(strcmp(type, 'pooling_v3'))        
        if (isfield(pars, 'name'))
            dir_name = [mask_type '_' pars.name '/'];
        else
            dir_name = [mask_type '_' cell2mat(pars.feats) '_' pars.color_type '_' pars.mode '_' pars.fg '_' pars.pooling_type '_pooling/'];
        end
        type_func = @pooling_local_feats_v3;
        pars.exp_dir = exp_dir;        
    elseif(strcmp(type, 'pooling_masked'))
        if (isfield(pars, 'name'))
            dir_name = [mask_type '_' pars.name '/'];
        else
            dir_name = [mask_type '_' cell2mat(pars.feats) '_' pars.color_type '_' pars.mode '_' pars.fg '_' pars.pooling_type '_pooling/'];
        end
        type_func = @pooling_local_feats_mask_sift;
        pars.exp_dir = exp_dir;   
    else        
        error('no such type defined');
    end

    %t = tic();    
    
    %parfor i=1:numel(img_names)        
    for i=1:numel(img_names)     
        img_name = img_names{i};
        
        filename_to_save = [exp_dir 'MyMeasurements/' dir_name img_name '.mat'];
        
        if ~overwrite
            if(exist(filename_to_save, 'file'))
                continue;
            end
        end

        img_file = [exp_dir 'JPEGImages/' img_name];
        I = imread([img_file '.jpg']);

        max_dim = max(size(I));
        if(max_dim>MAX_DIM)
            MAX_DIM
            img_name
            I = imresize(I, MAX_DIM/max_dim);
        end
        
        masks_file = [exp_dir 'MySegmentsMat/'  mask_type '/' img_name '.mat'];
        var_masks = load(masks_file);
        masks = var_masks.masks;
        
        pb_path = [exp_dir 'PB/'  img_name '_PB.mat'];
    

        the_pars = add_j_to_pars(pars, 1, img_name); % is this ok ? need to check it out
        
        if(isfield(var_masks, 'sp'))
            the_pars.sp_app = var_masks.sp_app;
            the_pars.sp = var_masks.sp;
        end
        
        my_all(exp_dir, I, type, masks, pb_path, the_pars, dir_name, img_name, type_func); 
    end
    %feats_on_masks_time_all_imgs = toc(t)
end

function [F, D] = my_all(exp_dir, I, type, masks, pb_path, the_pars, dir_name, img_name, type_func)
    if(isempty(masks))
        F = [];
        D = [];
    else    
        [F,D] = type_func(I, type, masks, pb_path, the_pars);        
    end
        
    the_dir = [exp_dir 'MyMeasurements/' dir_name];    
    
    %%%% if save is false it assumes you only passed one image!
    if(~exist(the_dir, 'dir'))
        mkdir(the_dir);
    end
    %save([the_dir img_name '.mat'], 'F', 'D', '-V6');    
    save([the_dir img_name '.mat'], 'D', '-V6');    
end

% to be able to use parfor
function pars = add_j_to_pars(pars,j, img_name)
    pars.j = j;
    pars.img_name = img_name;	
end

function [F, D] = run_exp_database_do_tps_hog(I, type, masks, pb_path,pars)
    pb_file = pb_path;
    n_bins = pars.n_ori;
    angle = 180;  % no gradient direction
    n_pyramid_levels = pars.n_levels;
    
    rois = square_boxes_from_masks(masks, I);
    p = tps_phog_masked(masks,n_bins,angle,n_pyramid_levels,rois, pb_file);
end

function [F,D] = run_exp_database_do_translation_inv_phog(I, type, masks, pb_path, pars)
end

function [F,D] = run_exp_database_do_hog(I, type, masks, pb_path, pars)
    %D{1} = jl_compute_hog(I, masks, [10 6]);
    %D{2} = jl_compute_hog(I, masks, [6 10]);
    %D{3} = jl_compute_hog(I, masks, [8 8]);
    %D = cell2mat(D');
    D = jl_compute_hog(I, masks, [8 8]);    
    F = [];
end

function [F,D] = run_exp_database_do_pyramid_hog(I, type, masks, pb_path, pars)
    D = jl_compute_pyramid_hog(I, masks);
    F = [];
end

function [F,D] = run_exp_database_do_scale_inv_phog(I, type, masks, pb_path, pars)
    n_bins = pars.n_ori;
    angle = 180; % no gradient direction
    n_pyramid_levels = pars.n_levels;

    if(size(I,3) ~= 1)
        I = rgb2gray(I);
    end
    I = (double(I)/255.0);  %I = (double(I)/255.0).*mask;

    pb_file = pb_path;
         
    if iscell(masks)
        masks = cell2mat(masks);
    end
    
    [the_bboxes] = square_boxes_from_masks(masks, I);
    %assert(all([the_bboxes(2,:) - the_bboxes(1,:)] == [the_bboxes(4,:) - the_bboxes(3,:)]));

    if(strcmp(type, 'back_mask_phog_scale_inv') || strcmp(type, 'bbox_phog_scale_inv'))
        WITH_BBOX = false;
        if(strcmp(type, 'bbox_phog_scale_inv'))            
            WITH_BBOX = true;            
        end
        [D] = phog_backmasked(I, n_bins, angle, n_pyramid_levels, the_bboxes, masks, pb_file, WITH_BBOX);
    elseif(strcmp(type,'mask_phog_scale_inv'))
        [D] = phog_masked(masks, n_bins, angle, n_pyramid_levels, the_bboxes, pb_file);        
    end
    D = single(D);
    F = the_bboxes;
    F = single(F);
end

function [F,D] = run_exp_database_do_phog(I, type, masks, pb_path, pars)          
    n_bins = pars.n_ori;
    angle = 180; % no gradient direction
    n_pyramid_levels = pars.n_levels;

    if(size(I,3) ~= 1)
        I = rgb2gray(I);
    end
    I = (double(I)/255.0);  %I = (double(I)/255.0).*mask;

    pb_file = pb_path;
         
    if(isempty(masks))
        n_masks = 0;
    else
        n_masks = size(masks,3);
    end
    
    the_bboxes = zeros(4, n_masks);    
       
    MARGIN = [10 10 10 10];
    for k=1:n_masks    
        props = regionprops(double(masks(:,:,k)), 'BoundingBox');
        if(isempty(props))
            bbox(1:4) = [1 2 3 4];
        else
            bbox(1) = props.BoundingBox(2); %ymin
            bbox(2) = bbox(1) + props.BoundingBox(4); %ymax
            bbox(3) = props.BoundingBox(1); % xmin
            bbox(4) = bbox(3) + props.BoundingBox(3); % xmax
        end
        bbox = round(bbox);
        bbox(1) = max(bbox(1) - MARGIN(1), 1);
        bbox(2) = min(bbox(2) + MARGIN(2), size(I,1));
        bbox(3) = max(bbox(3) - MARGIN(3), 1);
        bbox(4) = min(bbox(4) + MARGIN(4), size(I,2));
                
        the_bboxes(:,k) = bbox';
    end

    if(strcmp(type, 'mask_phog'))
        [D] = phog_masked(masks, n_bins, angle, n_pyramid_levels, the_bboxes);
    elseif(strcmp(type, 'back_mask_phog') || strcmp(type, 'back_mask_phog_nopb') || strcmp(type, 'bbox_phog'))
        bbox = false;
        if(strcmp(type, 'bbox_phog'))
            bbox = true;
        end
        
        if(pars.withpb)
            [D] = phog_backmasked(I, n_bins, angle, n_pyramid_levels, the_bboxes, masks, pb_file, bbox);
        else
            [D] = phog_backmasked(I, n_bins, angle, n_pyramid_levels, the_bboxes, masks, [], bbox);
        end
    else
        error('no such type');
    end
    
    D = single(D);
    F = the_bboxes;
    F =  single(F);
end

function [F,D] = run_exp_database_do_ch(I, type, masks, pb_path, pars)
    % color histogram
    N_BINS = 14;
    n_masks = size(masks,3);
    
    if(size(I,3) == 1)
        I(:,:,2) = I(:,:,1);
        I(:,:,3) = I(:,:,1);
    end
    Ir = double(I(:,:,1));
    Ig = double(I(:,:,2));
    Ib = double(I(:,:,3));
    
    %newI = rgb2hsv(I);
    %H = newI(:,:,3);
  
    if(strcmp(pars.figure_ground, 'ground'))
        masks = ~masks;
    end

    D = zeros(3*N_BINS, n_masks);
    for k=1:n_masks
        mask = masks(:,:,k);
        D(1:N_BINS,k) = hist(Ir(mask), N_BINS);
        D(N_BINS+1:2*N_BINS,k) = hist(Ig(mask), N_BINS);
        D((2*N_BINS) +1:3*N_BINS,k) = hist(Ib(mask), N_BINS);
    end
    F = [];
end

function [F,D] = run_exp_database_do_texton_hist(I, type, masks, pb_path, pars)
    % texton histogram
    N_BINS = 64;
    n_masks = size(masks,3);
    
    % load pb
    textons = myload(pb_path, 'textons');
    
    D = zeros( N_BINS, n_masks);
    
    if(strcmp(pars.figure_ground, 'ground'))
        masks = ~masks;
    end

    for k=1:n_masks
        mask = masks(:,:,k);
        D(:,k) = hist(double(textons(mask)), N_BINS); 
    end
    D = single(D);
    F = [];
end

function [F,D] = run_exp_database_do_zernike_moments(I, type, mask, pb_path, pars)
%   PATCH_SIDE = 70;
   n_powers = 0:9;
   
   mask = ~mask;
%   props = regionprops(double(mask), 'BoundingBox');  
% 
%   bbox(1) = props.BoundingBox(2); %ymin
%   bbox(2) = bbox(1) + props.BoundingBox(4); %ymax
%   bbox(3) = props.BoundingBox(1); % xmin
%   bbox(4) = bbox(3) + props.BoundingBox(3); % xmax
%   bbox([1 3]) = ceil(bbox([1 3]));
%   bbox([2 4]) = floor(bbox([2 4]));
%     
%   mask = mask(bbox(1):bbox(2), bbox(3):bbox(4));
%   max_dim = max(size(mask));
%   
%   rs = PATCH_SIDE/max_dim;
%   mask = imresize(mask, rs);
%   %imshow(mask)
%   D = abs(lans_zmoment(mask, 1:n_powers))';
%   %D_previous = abs(zernike(~mask, 0:n_powers))';   % invert to get foreground region
%   
  
  %[names, D] = mb_zernike(mask, n_powers,max(props.BoundingBox(3:4)/2));   % invert to get foreground region
  %D = abs(D');
  F = [];
  As=10000;
  As = 4000;
    [rr,ss]=size(mask);
    m000 = w_imgmoments(mask, 0, 0) ;
    m100 = w_imgmoments(mask, 1, 0) ;
    m010 = w_imgmoments(mask, 0, 1) ;
    Xco=(ss+1)/2;
    Yco=(rr+1)/2;
    cofx =Xco- m100/m000 ;
    cofy =Yco- m010/m000 ;
    I1=w_imgshift(mask,round(cofx),round(cofy));
    F1=I1(:);
    Am=sum(F1);
    cc=fix(rr/sqrt(Am/As));  
    dd=fix(ss/sqrt(Am/As));
    I2=imresize(I1,[cc,dd]);

    F=I2(:);
    m00=sum(F);
    iimg=invariant(I2,'m00 -translation 0');
    [A_nm,zmlist,cidx,V_nm] = P_zernike(iimg,n_powers);
    D =abs(A_nm/m00)';
end

function [F,D] = run_exp_database_do_local_shape_contexts(I, type, masks, pb_path, pars)
    if(size(I,3) >1)
        I = rgb2gray(I);
    end
   
    %load(pb_path); % not doing back_mask_local_shape_contexts yet

    nbins_theta = pars.theta; % 20
    nbins_r = pars.r;    % 8
    r_inner = pars.r_inner; %0.05, 0.01
    r_outer = pars.r_outer; %0.5 0.4
    
    %if(isempty(pars.codebook))
    %    error('not ready for that');
    %end
    
    if(~isempty(pars.codebook))        
        D = zeros(size(pars.codebook,2), size(masks,3));
    else
        D = [];
    end
    
    % load quality
    if isfield(pars,'quality_dir')
      load([pars.quality_dir pars.img_name '.mat']);
      [quals inds] = sort(Quality.q,'descend');
    else
      inds = 1:size(masks,3);
    end
    
%     pb = load(pb_path);
%     GrayI = pb.gPb_thin>10;
    for i=1:size(masks,3)
        %I_this = I.*uint8(masks(:,:,i));        
        
        if(strcmp(type, 'back_mask_local_shape_contexts'))
            % sample from outer contour, but consider inner edges
            NumPts = pars.NumPts; 
            
            
%             EdgeI = masks(:,:,inds(i)) .* GrayI; % apply mask to pb
            GrayI = I.*(uint8(masks(:,:,inds(i)))); % apply mask
            GrayI(GrayI == 0) = 255 ; % don't know why
            
            % canny
            EdgeI = edge(GrayI,'canny'); % using PB so no need for it
            [PRows PCols] = find(EdgeI == 1);
            NumPixels = size(PRows,1);            
            
            Interval = NumPixels/NumPts ;
            interV = 1 ;
            for iCount=1:NumPts            
                Bsamp(1,iCount) = PRows(floor(interV),1) ;
                Bsamp(2,iCount) = PCols(floor(interV),1) ;
                interV = interV + Interval ;            
            end

            % Now Compute the Shape Context Histogram
            out_vec = zeros(1, NumPts) ;
            Tsamp = zeros(1, NumPts) ;
            mean_dist_global = [] ;
            [desc, mean_dist_l] = sc_compute(Bsamp,Tsamp,mean_dist_global,nbins_theta,nbins_r, r_inner,r_outer,out_vec);

            if(~isempty(pars.codebook))
              proj = vl_ikmeanspush(uint8(desc)', int32(pars.codebook));
              D(:,i) = vl_ikmeanshist(length(pars.codebook), proj);
            else
              % if we don't have a codebook just retrieve some of the shape contexts 
                D = [D desc'];
                if(i==10) % this means only 10 masks will be used
                  break;
                end
            end
            F = [];
        elseif(strcmp(type, 'local_shape_contexts_boundary'))
          coords = bwboundaries(masks(:,:,inds(i)));
          coords = coords{1}';
          coords = coords(:,1:pars.SAMPLING_RATE:end);

          
          Bsamp = coords;
          Tsamp = zeros(1,size(Bsamp,2));    
          out_vec = Tsamp;
            % sample from outer contour, don't consider inner edges
            desc = sc_compute(Bsamp,Tsamp,[],nbins_theta,nbins_r,r_inner,r_outer, out_vec);
            if(~isempty(pars.codebook))
                proj = vl_ikmeanspush(uint8(desc)', int32(pars.codebook));
                D(:,i) = vl_ikmeanshist(length(pars.codebook), proj);
            else
                % if we don't have a codebook just retrieve some of the shape contexts 
                D = [D desc'];
                %if(i==10) % up to 10
                %    break;
                %end
            end
            F = [];
        else
            error('no such type');
        end
        %imshow(I_this);
        %pause;
    end
end

function [F, D] = run_scaled_dense_sifts(I, type, masks, pb_path, pars)
    if(size(I,3) >1)
        I = rgb2gray(I);
    end
    
    load(pb_path);                

    if(~isempty(pars.codebook))
        D = zeros(size(pars.codebook,2), size(masks,3));
    else
        D = [];
    end
    
    for i=1:size(masks,3)
        %I_this = I.*uint8(masks(:,:,i));        

        % sample from outer contour, don't consider inner edges
        spacing = 4;
        time_image=tic();

        reg_props = [regionprops(masks(:,:,i), 'BoundingBox')];
        horizontal_width = reg_props(1).BoundingBox(3);
        vertical_width = reg_props(1).BoundingBox(4);
        reference = min(horizontal_width, vertical_width);
        
        % fractions of the mask max reference length (width or height)
        %square_edge = [0.2 0.3 0.4].*reference; 
        square_edge = [0.1 0.2 0.4 0.6].*reference; 
        bin_size = max(2,square_edge ./ 4);
        bin_size = unique(bin_size);
        
        desc = [];
        F = [];

        for j=1:numel(bin_size)
            [F1,D1] = vl_dsift(single(I), 'Step', spacing, 'Size', bin_size(j));
            desc = [desc D1];
            F = [F F1];                
        end
        % keep the ones inside the mask 
        
        f_lin = sub2ind(size(I), round(F(2,:)), round(F(1,:)));
        [f_lin_in, id] = intersect(f_lin, find(masks(:,:,i)));
        desc = desc(:, id);
        F = F(:, id);
        
       %imshow(masks(:,:,i))
       %hold on; vl_plotsiftdescriptor(desc(:,1), [F(:,1); bin_size(1)/3.0; 0], 'Magnif', 3.0);
        
        toc(time_image)
        if(~isempty(pars.codebook))
            proj = vl_ikmeanspush(uint8(desc), int32(pars.codebook));
            D(:,i) = vl_ikmeanshist(length(pars.codebook), proj);
        else
            % if we don't have a codebook just retrieve some of the feats
            D = [D desc];
        end
        F = [];
    end
end

function [F, D] = run_exp_database_do_signature(I, type, masks, pb_path, pars)
    error('not ready');
    SAMP = 200;
    
    D = zeros(SAMP*2, size(masks,3));
        
    for i=1:size(masks,3)
        centroid = regionprops(masks(:,:,i), 'Centroid');    
        [B,L] = bwboundaries(masks(:,:,i),'noholes');
        B = B{1}';
                
        B = B - repmat((centroid.Centroid)', 1, size(B,2));
        
        % normalize distances
        B = B/max(abs(B(:)));

        % if number of points along boundary is lower than SAMP,
        % interpolate to SAMP points
        if(SAMP > size(B,2))
            interp1();
        else
            % else sample SAMP points
            B = B(:,round(1:10));
        end

        % visualize
        plot(B(2,:), B(1,:)); axis equal; axis ij;               

        % compute angles and distances, create descriptor

        
        % normalize descriptor        

    end
end


function [F, D ] = run_exp_database_do_ansig(I, type, masks, pb_path, pars)
    D = zeros(512, size(masks,3));
    
    if(size(I,3)>1)
        I = rgb2gray(I);
    end
    
    max_n_points = 5000;
    for i=1:size(masks,3)
        [edges_img] = edge(I.*uint8(masks(:,:,i))); 
        [coords_x, coords_y] = find(edges_img); % the biggest one
        coords = [coords_x, coords_y];
        rp = randperm(min(max_n_points,size(coords,1)));        
        [y] = ansig_descriptor(coords(rp,:)');
        D(1:512,:,i) = real(y);
        D(513:1024,:,i) = imag(y);
    end    
    F = [];
end
    
function [F, D] = run_idsc_boundary(I, type, masks, pb_path, pars)
    D = cell(1, size(masks,3));
    F = cell(1, size(masks,3));
    
    if(size(I,3)>1)
        I = rgb2gray(I);
    end
    
    %-- shape context parameters
    n_dist		= 5;
    n_theta		= 12;
    bTangent	= 1;
    bSmoothCont	= 1;
    n_contsamp = 100;
        
    for i=1:size(masks,3)     
        F{i}	= extract_longest_cont(masks(:,:,i), n_contsamp);	
        [sc,V,E,dis_mat,ang_mat] = compu_contour_innerdist_SC( ...
            F{i},masks(:,:,i), ...
            n_dist, n_theta, bTangent, bSmoothCont,...
            0);
        D{i} = sc;
    end        
end

% convert to gray level first
function [F,D] = pyramid_lbp_V5(I, type, masks,pb_path, pars)
    %SP = cartprod([1 0 -1]', [1 0 -1], [1 0 -1]);
    %[trash, to_remove] = intersect(SP, [0 0 0], 'rows');
    %SP(to_remove, :) = [];
    error('this one is not working for some reason (rgb2gray)');
    SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];                 
          
    % this works
    for i=1:size(I,3)
        I2(:,:,i) = lbp(I(:,:,i),SP,0,'i');
    end
    I2 = round(max(I2, [], 3));
     
    % this doesn't work ??
    I = rgb2gray(I);
    I2= lbp(I,SP,0,'i');
         
     hist(double(I2(:)))
     
     I2 = padarray(double(I2),1,-1, 'both');
     I2 = padarray(double(I2'),1,-1, 'both')';
     
     I2 = I2 +1;
     I2(I2==256) = 255;
     
     [f1, f2] = find(ones(size(I2)));
     F2 = [f2'; f1'];
     D2 = I2(:);
     
     for i=1:size(masks,3)
        [a{i}, b{i}] = filter_feats_outside_roi(D2',F2,masks(:,:,i), 'points');     
     end
     
     D = cell2mat(form_spatial_pyramid(b, a, masks, 255+1, 2));
     F = [];
end


% set -1 along the border
function [F,D] = pyramid_lbp(I, type, masks,pb_path, pars)
     SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];   
     for i=1:size(I,3)
        I2(:,:,i) = lbp(I(:,:,i),SP,0,'i');
     end     
     I2 = round(max(I2, [], 3));
     I2 = padarray(I2,1,-1, 'both');
     I2 = padarray(I2',1,-1, 'both')';
     
     I2 = I2 +1;
     [f1, f2] = find(ones(size(I2)));
     F2 = [f2'; f1'];
     D2 = I2(:);
     
     for i=1:size(masks,3)
        [a{i}, b{i}] = filter_feats_outside_roi(D2',F2,masks(:,:,i), 'points');     
     end
     
     D = cell2mat(form_spatial_pyramid(b, a, masks, 255+1, false, 2));
     F = [];
end

% reducing the number of elements
function [F,D] = pyramid_lbp_V3(I, type, masks,pb_path, pars)
     SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];   
     for i=1:size(I,3)
        I2(:,:,i) = lbp(I(:,:,i),SP,0,'i');
     end     
     I2 = round(max(I2, [], 3));
     I2 = padarray(I2,1,0, 'both');
     I2 = padarray(I2',1,0, 'both')';
          
     K = 2;
    I2 = round(I2/K);
    
     [f1, f2] = find(ones(size(I2)));
     F2 = [f2'; f1'];
     D2 = I2(:);
     
     for i=1:size(masks,3)
        [a{i}, b{i}] = filter_feats_outside_roi(D2',F2,masks(:,:,i), 'points');     
     end
     
     D = cell2mat(form_spatial_pyramid(b, a, masks, ceil(255/K), 2));
     F = [];
end

function [F,D] = pyramid_lbp_V2(I, type, masks,pb_path, pars)
     SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];   
     for i=1:size(I,3)
        I2(:,:,i) = lbp(I(:,:,i),SP,0,'i');
     end     
     I2 = round(max(I2, [], 3));
     I2 = padarray(I2,1,0, 'both');
     I2 = padarray(I2',1,0, 'both')';
     [f1, f2] = find(ones(size(I2)));
     F2 = [f2'; f1'];
     D2 = I2(:);
     
     for i=1:size(masks,3)
        [a{i}, b{i}] = filter_feats_outside_roi(D2',F2,masks(:,:,i), 'points');     
     end
     
     D = cell2mat(form_spatial_pyramid(b, a, masks, 255, 2));
     F = [];
end

function [F,D] = pyramid_lbp_V1(I, type, masks,pb_path, pars)
     SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];   
     for i=1:size(I,3)
        I2(:,:,i) = lbp(I(:,:,i),SP,0,'i');
     end     
     I2 = round(mean(I2,3));
     I2 = padarray(I2,1,0, 'both');
     I2 = padarray(I2',1,0, 'both')';
     [f1, f2] = find(ones(size(I2)));
     F2 = [f2'; f1'];
     D2 = I2(:);
     
     for i=1:size(masks,3)
        [a{i}, b{i}] = filter_feats_outside_roi(D2',F2,masks(:,:,i), 'points');     
     end
     
     D = cell2mat(form_spatial_pyramid(b, a, masks, 255, 2));
     F = [];
end

% features for learning to segment
function [F, D] = run_exp_database_do_simple_segment_feats(I, type, masks, pb_path, pars)
    % A. pairwise affinity features:
    % 1 - cut ratio ( sum of edges across cut, divided by their number )
    % 2 - cut
    % 3 - normalized cut
    % 4 - unbalanced normalized cut
    % 
    % B. Area features
    % from regionprops (17 feats)
           
    img_name = pars.img_name 
    %img_name = pb_path(strfind(pb_path, 'PB/')+3:end);
    %img_name = pb_path(1:end-7);
    
    
    %s = LongRangeSegmenter(I,img_name);    
    %s = s.set_pb_path(pars.pb_dir);
    %s = s.initialize();
 
    %img_dgraph = s.p_hgraph.pairwise_to_dgraph();
    resh_masks = reshape(masks, size(masks,1) * size(masks,2), size(masks,3));

    no_separate = true;
    S = SegmentProcessor([], resh_masks, I, pb_path,  25, no_separate);
    get_all = true;
    S = S.compute_energies(get_all);

    %
    %%%% Cut features %%%%
    %
    D_cut = zeros(8, size(masks,3));
    D_cut(1,:) = [S.energies(:).cut_ratio];
    D_cut(2,:) = [S.energies(:).cut];
    D_cut(3,:) = [S.energies(:).normalized_cut];
    D_cut(4,:) = [S.energies(:).unbalanced_normalized_cut];
    tmp_var = [S.energies(:).fraction_of_healthy_boundary];
    D_cut(5:end,:) = reshape(tmp_var, 4,size(tmp_var,2)/4);
    
    %
    %%%% Coarse Region Shape and Location features %%%%%%
    %
    D_crsl = zeros(19, size(masks,3)); 
    
    % absolute quantities
    for i=1:size(masks,3)
        props = regionprops(masks(:,:,i), 'Area', 'Centroid', 'BoundingBox', 'MajorAxisLength', ...
                                            'MinorAxisLength', 'Eccentricity', 'Orientation', 'ConvexArea', 'EulerNumber', 'EquivDiameter', 'Solidity', 'Extent', 'Perimeter');
        props = props(1);  % might be more than one, describe the first
        D_crsl(1:17,i) = struct2array(props(1)); 
                
        % convexity
        D_crsl(18, i) = props.Area / props.ConvexArea; % solidity is the same as this one
        % absolute distance to center of image
        D_crsl(19, i) = norm([size(I,1) size(I,2)]./2 - props.Centroid);
    end    
    
    
    D = [D_cut; D_crsl];
    F = size(I)';
end

function [F, D] = run_exp_database_do_extended_segment_feats(I, type, masks, pb_path, pars)
    % 
    im = double(I)/255.0;

    var = load(pb_path, 'textons');
    textons = double(var.textons);

%    [ ...
%    textons, ...
%    bg_r3, bg_r5,  bg_r10,  cga_r5, cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20, tg_r5,  tg_r10,  tg_r20...
%    ] = mex_pb_parts_final_selected(im(:,:,1),im(:,:,2),im(:,:,3));

    %
    %%% histograms of textons
    %
    t = reshape(textons, numel(textons), 1);
    m = reshape(masks,numel(textons), size(masks,3));
    N = 65;
    hist_fg = zeros(N, size(m,2));
    hist_bg = zeros(N, size(m,2));
    t = t+1; % to avoid t==0, that has problem with int_hist
    for i=1:size(m,2)
        hist_fg(:,i) = int_hist(t(m(:,i)), N)';
        hist_bg(:,i) = int_hist(t(~m(:,i)), N)';
    end
    hist_fg = scale_data(hist_fg, 'norm_1');
    hist_bg = scale_data(hist_bg, 'norm_1');
    D = chi2_mex(single(hist_fg), single(hist_bg));
    
    %
    %%% histograms of brightness
    %
    N_BINS = 256;
    hist_fg_b = zeros(N_BINS, size(m,2));
    hist_bg_b = zeros(N_BINS, size(m,2));
    Igray = rgb2gray(I)+1;    
    Igray_r = double(reshape(Igray, numel(Igray), 1));
    for i=1:size(m,2)        
        hist_fg_b(:,i) = int_hist(Igray_r(m(:,i)), N_BINS)';
    end
    hist_fg_b = scale_data(double(hist_fg_b), 'norm_1');
    hist_bg_b = scale_data(double(hist_bg_b), 'norm_1');
    D_b = chi2_mex(single(hist_fg_b), single(hist_bg_b));          
    
    %%% contour energy
    load(pb_path); % gPb_thin
    gPb_thin_c = reshape(gPb_thin, numel(gPb_thin),1);
    
    all_bw = imdilate(masks, ones(3,3)) & ~masks;
    n = zeros(1,size(m,2));
    s = n;
    s_intra = n;
    for i=1:size(m,2)
%         t = tic();
%         bw{i} = bwboundaries(masks(:,:,i));
%         n(i) = size(bw{i}{1}, 1);
%         ids = bw{i}{1};
%         time_bwbound = toc(t)
        
        %t = tic();
        [ids1, ids2] = find(all_bw(:,:,i));
        ids = [ids1 ids2];
        n(i) = size(ids,1);
        %time_dilate = toc(t)
        
        s(i) = sum(gPb_thin( sub2ind(size(gPb_thin), ids(:,1), ids(:,2))));
        s_intra(i) = sum(gPb_thin_c(m(:,i)));         
    end
    %gPb = mean(gPb_orient,3);
    
    %%% curvilinear continuity 
    %%%( curvature using bruckstein discrete approximation )
    the_lines = cell(size(m,2),1);
    for i=1:size(m,2)
        %BREAK_FACTOR = 0.9;
        %thresh = BREAK_FACTOR*0.05*log(size(bw{i}{1},1))/log(1.1);
        thresh = 1;
        lines = lineseg({ids}, thresh);
        the_lines{i} = seglist2segarray(lines);        
        %jl_plot_lines(the_lines{i});
        %pause;
    end
    
    sum_curvature = zeros(size(m,2),1);
    for i=1:size(m,2) 
        if(size(the_lines{i},2) == 1)
            sum_curvature(i) = 0;
            continue;
        end
        
        curvatures = zeros(size(the_lines{i},2)-1,1);
        curvatures(1) = angle_between_linesegs(the_lines{i}(:,end), the_lines{i}(:,1));
        for j=1:size(the_lines{i},2)-1
            curvatures(j) = (angle_between_linesegs(the_lines{i}(:,j), the_lines{i}(:,j+1)))^2;      
            len_1 = norm(the_lines{i}(1:2,j) - the_lines{i}((3:4),j));
            len_2 = norm(the_lines{i}((1:2),j+1) - the_lines{i}(((3:4)),j+1));
            curvatures(j) = curvatures(j)/(min(len_1,len_2));
        end
        sum_curvature(i) = sum(curvatures); 
    end
        
    % 1. inter-region dissimilarity ( big is good )
    D_chi2_fg_bg = diag(D); % this seems a nice a feature, could grow it, by doing a new feature with the relative value to the best of the image
    
    % 2. intra-region similarity (simplicity, small is good)       
    D_fg_simplicity = sum(hist_fg>(1/300))'; 
    
    % 3. inter region brightness similarity
    D_chi2_fg_bg_b = diag(D_b);
    
    % 3. b) brigh
    % 4. intra-region brightness similarity (simplicity, small is good)
    D_fg_b_simplicity = sum(hist_fg_b>(1/N_BINS))';
    
    % 5. inter-region contour energy (similar to ratio cut)
    D_fg_bg_contour_energy = (s./n)';
    
    % 6. intra-region contour_energy
    D_fg_bg_intra_contour_energy = (s_intra ./sum(m))';
    
    % 7. curvilinear continuity
    D_cc = sum_curvature;
            
    F = [];
    D = [D_chi2_fg_bg D_fg_simplicity D_chi2_fg_bg_b D_fg_b_simplicity D_fg_bg_contour_energy D_fg_bg_intra_contour_energy D_cc]';
end

function [F, D] = run_exp_database_do_composition_segment_feats(I, type, masks, pb_path, pars)     
      counter = 1;
      n_region_feats = length(pars.region_feats);
      
      D = zeros(2*n_region_feats, size(masks,3)); % 2 for normalized version
      

      for i=1:n_region_feats
        % 1. sum of dissimilarity between a region and all others
        D(counter,:) = sum(chi2_mex(single(pars.region_feats{i}.figure{pars.j}), single(pars.region_feats{i}.figure{pars.j})));
        D(counter+1,:) = D(counter,:) / norm(D(counter,:));        
        counter = counter + 2;
        
        % 2. dissimilarity between the region and its background (doesn't
        % seem to help much)
%         D(counter,:) = diag(chi2_mex(single(pars.region_feats{i}.figure{pars.j}), single(pars.region_feats{i}.ground{pars.j})));
%         D(counter+1,:) = D(counter,:) / norm(D(counter,:));        
%         counter = counter + 2;
      end
      F = [];            
end
