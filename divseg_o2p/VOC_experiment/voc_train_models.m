function voc_train_models(feat_collection_learning, lc)
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

    DefaultVal('*lc', '3');
    DefaultVal('*feat_collection_learning', '''all_feats_pca_noncent_5000''');
    
    mask_type = 'CPMC_segms_150_sp_approx';
    gt_mask_type = 'ground_truth_sp_approx';

    svr_eps = 0.25; % the epsilon-band of SVR
    classes = 1:20;
    exp_dir = './VOC/'; % directory where data will be stored

    if 1 % train on all annotated data but val11       
        imgset_train = 'all_gt_segm_minus_val11';
    elseif 0
        imgset_train = 'train11';
    elseif 0
        imgset_train = 'all_gt_segm';        
    end
    
    if 1
        % this works on a pc with 32gb of ram plus 32gb of swap space
        
        % number of examples in each chunk of data (test if 
        % zeros(12500, MAX_CHUNK, 'single') fits comfortably in memory)
        MAX_CHUNK = 450000; % ~ 22 gb each chunk                
        
        % how many classes to train at once (we will store in 
        % memory the support vectors for all classes, when switching to the
        % next chunk)
        classes_chunk = 10;
    else
        % this will be slower to train, but should fit in pcs with 16 gb of RAM (haven't tested, though)        
        MAX_CHUNK = 200000; % 10 gb each chunk
        classes_chunk = 5; % it will load data from disk 4x (1 for each chunk of classes)
        
        % if this does not work on your pc, reduce classes_chunk further or
        % get more swap space or more ram
    end
    
    % Cache feature files - if you plan to train multiple times with
    % different parameters (or multiple class chunks) it is a bit faster 
    % if you store files of aggregated
    % features (one for each chunk). Doubles the amount of disk space
    % required. I haven't tested it turned off, it's likely it doesn't
    % work/requires a little coding effort.
    CACHE_AGG_FEATURES = true;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    fprintf('Training 20 models on image set %s.\n', imgset_train);
    t_learning = tic();

    range_classes = chunkify(classes,numel(classes)/classes_chunk);
        
    for i=1:numel(range_classes)
        o2p_train(exp_dir, imgset_train, mask_type, gt_mask_type, feat_collection_learning, ...
            range_classes{i}, lc, svr_eps, MAX_CHUNK, CACHE_AGG_FEATURES, [imgset_train '_' feat_collection_learning]);
    end
    
    t_learning = toc(t_learning);
    fprintf('Learning took: %d seconds\n', t_learning);
end