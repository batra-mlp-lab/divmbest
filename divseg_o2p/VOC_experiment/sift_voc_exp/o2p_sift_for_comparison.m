function o2p_sift_for_comparison()
    addpath('../../src/');
    addpath('../../src/SegmBrowser/');
    addpath('../../external_src/');
    addpath('../../external_src/vlfeats/toolbox/mex/mexa64/');
    addpath('../../external_src/vlfeats/toolbox/sift/');
    addpath('../../external_src/vlfeats/toolbox/misc/');
    addpath('../../external_src/vlfeats/toolbox/mex/mexa64/');
    addpath('../../external_src/ndSparse/');
    addpath('../../external_src/VOCcode/');
    addpath('../../src/liblinear-custom/matlab/');
    
    s = RandStream('mt19937ar','Seed',1234);
    RandStream.setGlobalStream(s);
    
    
    % these are the basic types, the power normalization will be set at
    % learning time
    exp_dir = '../VOC/';
    mask_type = {'ground_truth', 'ground_truth_sp_approx'};
    imgset = 'trainval11';
    
    img_names = textread([exp_dir 'ImageSets/Segmentation/' imgset '.txt'], '%s');
    
    % extract our feature combination on 'ground_truth' masks
    o2p_extract_feats(exp_dir, 'ground_truth', 'all_feats', img_names);
    
    % extract HOG features (8x8 cells) on 'ground_truth' masks (will require the TTI-C
    % part-based model package to compute HOG
    if 0
        SvmSegm_extract_measurements_on_masks('ground_truth', 'hog', [], img_names, exp_dir);
        SvmSegm_extract_measurements_on_masks('ground_truth_sp_approx', 'hog', [], img_names, exp_dir);
    end
    
    % now the features for table 1
    feat_types = {'SIFT_1MAXP', 'SIFT_2MAXP', 'SIFT_1AVGP', 'SIFT_2AVGP', 'SIFT_2AVGP_LOG'};
    enriched = [false true];    
                
    for i=1:numel(feat_types)  
        for j=1:numel(enriched)
            for k=1:numel(mask_type)
                pars = [];

                if(strcmp(feat_types{i}, 'SIFT_1MAXP'))            
                    pars.pooling_type = 'max';                   
                elseif(strcmp(feat_types{i}, 'SIFT_2MAXP'))
                    pars.pooling_type = 'max2p';
                elseif(strcmp(feat_types{i}, 'SIFT_1AVGP'))                
                    pars.pooling_type = 'avg';
                elseif(strcmp(feat_types{i}, 'SIFT_2AVGP'))                
                    pars.pooling_type = 'avg2p';                
                elseif(strcmp(feat_types{i}, 'SIFT_2AVGP_LOG'))
                    pars.pooling_type = 'log_avg';
                end

                pars.color_type = 'gray';
                pars.main_feat = {'really_dense_sift'};
                pars.fg = {'figure'};        
                pars.STEP = 4;
                pars.spatial_pyr =  false;
                pars.mode = 'single_octave';
                pars.color_type = 'gray';
                pars.weighting_type = 'uniform';
                pars.base_scales = [2 4 6 8];
                pars.sum_approx = -1;

                if enriched(j)
                    pars.enrichments{1} = {'rgb', 'hsv', 'lab', 'xy', 'scale'};
                    pars.name = ['ECCV12_TABLE1_ENRICHED_' feat_types{i}];
                else
                    pars.enrichments{1} = {};
                    pars.name = ['ECCV12_TABLE1_' feat_types{i}];
                end

                SvmSegm_extract_measurements_on_masks(mask_type{k}, 'pooling_v3', pars, img_names, exp_dir);
            end
        end
    end
end