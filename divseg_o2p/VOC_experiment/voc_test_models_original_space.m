function voc_test_models_original_space()
    addpath('../src/');
    addpath('../src/SegmBrowser/');
    addpath('../external_src/');
    addpath('../external_src/vlfeats/toolbox/mex/mexa64/');    
    addpath('../external_src/vlfeats/toolbox/sift/');
    addpath('../external_src/vlfeats/toolbox/misc/');
    addpath('../external_src/vlfeats/toolbox/mex/mexa64/');
    addpath('../external_src/ndSparse/');
    addpath('../external_src/VOCcode/');
    addpath('../src/liblinear-custom/matlab/');
    addpath('../external_src/immerge/');
    
    exp_dir = './VOC/';
    mask_type_ho = 'CPMC_segms_150_sp_approx';
    
    imgset_ho = 'val11';
    %imgset_ho = 'test11';
    
    feat_collection = 'all_feats';
    name = {'all_gt_segm_minus_val11_all_feats_pca_noncent_5000_3.000000'};
    
    cache_dir = [exp_dir '/Cache/'];

    classes = 1:20;
    SVM = true;

    [feats, power_scaling, input_scaling_type, feat_weights] = feat_config(feat_collection);
    if(all(feat_weights==1))
        feat_weights = [];
    end
    
    if(~strcmp(mask_type_ho, 'CPMC_150_segms'))
        ho_cache_file = [cache_dir  imgset_ho '_' feat_collection '_mask_' mask_type_ho '_ps_' int2str(power_scaling) '_scaling_' input_scaling_type];
    else
        ho_cache_file = [cache_dir imgset_ho '_' feat_collection '_sqrt_' int2str(power_scaling) '_scaling_' input_scaling_type];
    end

    MAX_INPUT_CHUNK = 10000000000000000;
        
    browser_ho = SegmBrowser(exp_dir, mask_type_ho, imgset_ho);
    whole_ho_ids = 1:numel(browser_ho.whole_2_img_ids);
    
    % create multiple threads (set how many you have)
    N_THREADS = 6;
    if(matlabpool('size')~=N_THREADS)
        matlabpool('open', N_THREADS);
    end    
    
    % computes pairwise overlaps, which can be helpful in inference if
    % you want to experiment with any form of non-maximum supression (it will be cached)
    % the default is to have no non-maximum supression.
    SvmSegm_compute_overlaps(exp_dir, browser_ho.img_names, mask_type_ho); 
    
    chunked_whole_ho_ids = chunkify(whole_ho_ids, ceil(numel(whole_ho_ids)/MAX_INPUT_CHUNK));
    
    for h=1:numel(name)
        % load models
        feat = browser_ho.get_whole_feats(1, feats, input_scaling_type, feat_weights);
        
        %beta = zeros(20, size(feat,1));
        for i=1:numel(classes)            
            var = load([exp_dir 'MODELS/' name{h} '/' browser_ho.categories{classes(i)} '.mat']);
            beta(i,:) = var.model.w;
        end

        [beta] = pca_to_original_wrapper(exp_dir, beta, 'ground_truth_sp_approx', feats, [5000 5000 2500]);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%% Get (class, segment) scores %%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t_predicting = tic();
        y_pred = cell(1,numel(chunked_whole_ho_ids));
        for i=1:numel(chunked_whole_ho_ids)
            vgg_progressbar('Testing on hold-out set. ', i/numel(chunked_whole_ho_ids));
            chunk_cache_file = [ho_cache_file '_chunk_' int2str(i)];
            feat_loading_wrapper_altered(browser_ho, chunked_whole_ho_ids{i}, feats, input_scaling_type, power_scaling, chunk_cache_file, [], feat_weights);
            [Feats_ho, dims] = feat_loading_wrapper_altered([], [], [], [], [], chunk_cache_file);

            y_pred{i} = predict_regressor(Feats_ho, beta', true);            
            Feats_ho = [];
        end        
        if(iscell(y_pred))
            y_pred = cell2mat(y_pred);
        end                                                

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%% Generate desired outputs %%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        pasting_mode = true;
        segms2pixel_algo = 'simple';

        n_segms_per_img = inf;
        if(numel(browser_ho.categories)~=21)
            browser_ho.categories = [browser_ho.categories 'Background'];        
        end

        if(pasting_mode)                        
            % simply paste top-scoring masks onto the image. Higher scoring
            % masks are pasted over lower scoring masks.
            nms_type = 'segment';            

            THRESH = 0.28; % any sufficiently small value to start search with 
            
            NMS_MAX_OVER = 1; % spatial non-maximum supression (1 means nms is not performed)
            NMS_MAX_SEGMS = 3; % max number of segments per image            
            SIMP_BIAS_STEP = 0.02; % background threshold increase for each additional segment above 1
            
            % max number of segments per image on average (used to set the background threshold)
            % we set this value to the average number of objects in the
            % training set. Of course, that is just a coincidence. ;-)
            MAX_AVG_N_SEGM = 2.2; 

            while(n_segms_per_img > MAX_AVG_N_SEGM)
                y_pred(21,:) = THRESH; % Background score
                y_pred(setdiff(1:20, classes),:) = -10000; % remove non-selected classes

                [local_ids, labels, scores, global_ids] = nms_inference_simplicity_bias(browser_ho, y_pred, nms_type, whole_ho_ids, NMS_MAX_OVER, NMS_MAX_SEGMS, SIMP_BIAS_STEP);                    
                n_segms_per_img = numel(cell2mat(labels')) / numel(labels)

                THRESH = THRESH                
                THRESH = THRESH+0.01;
            end

            all_THRESH(h) = THRESH-0.01;
            all_n_segms(h) = n_segms_per_img

            name{h}
            if(~strcmp(imgset_ho(1:4), 'test'))
                % classification results                  
                [categ_cls_ap{h}, categ_fp, categ_tp] = browser_ho.voc_cls_score(local_ids, global_ids, labels, scores);
                categ_cls_ap{h}
                fprintf('Classification: average precision: %f\n', mean(categ_cls_ap{h}))

                % detection results (set low NMS_MAX_OVER and increase
                % NMS_MAX_SEGMS for decent results).
                [categ_det_ap{h}, categ_fp, categ_tp] = browser_ho.voc_detection_score(local_ids, global_ids, labels, scores);
                fprintf('Detection: mean of average precision: %f\n', mean(categ_det_ap{h}))
            end
            
            browser_ho.VOCopts.testset = imgset_ho;
            this_browser_ho = browser_ho;
            this_browser_ho.voc_segm_outputs(1:numel(browser_ho.img_names), global_ids, labels, scores, name{h}, false, segms2pixel_algo);
            
            if(~strcmp(imgset_ho(1:4), 'test'))
                [accuracies{h},avacc,conf,rawcounts] = VOCevalseg(this_browser_ho.VOCopts, name{h});
            end
            
            if 0
                % this saves images with label transparencies in folder
                % "visuals"
                this_browser_ho.voc_segm_visuals(['./VOC/results/VOC2012/Segmentation/' name{h} '_' imgset_ho '_cls/'], ...
                    './VOC/JPEGImages/', name{h}, false)
            end
        end                
    end

    cellfun(@mean, accuracies)    
end
