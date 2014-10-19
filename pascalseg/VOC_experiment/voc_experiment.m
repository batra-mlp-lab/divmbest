function voc_experiment()  
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
                
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%  Configuration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    exp_dir = './VOC/'; % directory where data will be stored
    
    % collection of features, defined in feat_config()
    feat_collection = 'all_feats';        
    PCA_DIMS = [5000 5000 2500]; % for sift_mask, sift and lbp, respectively         
    
    % default segments used
    gt_mask_type = 'ground_truth_sp_approx';
    mask_type = 'CPMC_segms_150_sp_approx';
    
    gt_imgsets = {'all_gt_segm_mirror', 'all_gt_segm'};

    % If set as true, only PCA versions of segment features will be stored in disk,
    % saving 250 gigabytes. However, if you plan to use more
    % pca dimensions later (or no pca at all), you'll have to recompute features again
    BUDGET_HARD_DRIVE = false;        
    
    % create multiple threads (set how many you have)
    N_THREADS = 6;
    if(matlabpool('size')~=N_THREADS)
        matlabpool('open', N_THREADS);
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    t_experiment = tic();    
    
    % 1. Get the data and the CPMC segments
    if(~exist('VOC/MySegmentsMat/ground_truth_sp_approx/', 'dir'))
        % do this just once
        generate_canonical_dataset(exp_dir);
    end
    
    
    % 2. Feature extraction on ground truth masks
    % (track progress of any feature extraction by looking in MySegmentsMat folder)
    disp('Extracting features on all ground truth masks.');
    t = tic();
    for i=1:numel(gt_imgsets)
        img_names = textread([exp_dir 'ImageSets/Segmentation/' gt_imgsets{i} '.txt'], '%s');
        o2p_extract_feats(exp_dir, gt_mask_type, feat_collection, img_names);
    end
    t_gt = toc(t);
    fprintf('Feature extraction on GT masks took: %d seconds\n', t_gt);
    
    % 3. PCA + projection of GT masks
    t_pca = tic();
    disp('Performing principal component analysis.');
    o2p_pca_noncent(exp_dir, gt_mask_type, feat_collection);
    
    for i=1:numel(gt_imgsets)
        img_names = textread([exp_dir 'ImageSets/Segmentation/' gt_imgsets{i} '.txt'], '%s');
        o2p_project_pca_noncent(exp_dir, img_names, gt_mask_type, gt_mask_type, feat_collection, PCA_DIMS)
        if BUDGET_HARD_DRIVE
            remove_original_features(exp_dir, img_names, gt_mask_type, feat_collection);
        end
    end
    t_pca = toc(t_pca); % should take an hour
    fprintf('PCA + projection of GT mask features took: %d seconds\n', t_pca);
    
    % 4. Feature extraction + PCA projection on CPMC segments
    disp('Extracting features on all segments (~16 hours).')
    %imgset = 'all_gt_segm';    
    imgset = 'all_gt_segm';    
    img_names = textread([exp_dir 'ImageSets/Segmentation/' imgset '.txt'], '%s');
    t_f_segm = tic();
    if BUDGET_HARD_DRIVE
        n_chunks = 10;
        img_names_chunks = chunkify(img_names', n_chunks);
        for i=1:n_chunks
            o2p_extract_feats(exp_dir, mask_type, feat_collection, img_names_chunks{i});
            o2p_project_pca_noncent(exp_dir, img_names_chunks{i}, gt_mask_type, mask_type, feat_collection, PCA_DIMS)
            remove_original_features(exp_dir, img_names_chunks{i}, mask_type, feat_collection);
        end
    else
        o2p_extract_feats(exp_dir, mask_type, feat_collection, img_names);
        o2p_project_pca_noncent(exp_dir, img_names, gt_mask_type, mask_type, feat_collection, PCA_DIMS);
    end
    t_f_segm = toc(t_f_segm);
    fprintf('Feature extraction on segments took: %d seconds\n', t_f_segm);
    
    toc(t_experiment)
end

function remove_original_features(exp_dir, img_names, mask_type, feat_collection)
    feats = feat_config(feat_collection);
    
    for h=1:numel(feats)
        for i=1:numel(img_names)
            file_path = [exp_dir 'MyMeasurements/' mask_type '_' feats{h} '/' img_names{i} '.mat'];
            if(exist(file_path, 'file'))
                system(['rm ' file_path]);
            end
        end
    end
end


