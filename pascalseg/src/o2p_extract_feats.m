function o2p_extract_feats(exp_dir, mask_type, feat_collection, img_names)
    s = RandStream('mt19937ar','Seed',1234);
    RandStream.setGlobalStream(s);

    feat_types = feat_config(feat_collection);
    for i=1:numel(feat_types)
        if(strcmp(feat_types{i}, 'SIFT_GRAY_mask'))
            pars = [];
            pars.name = 'SIFT_GRAY_mask';
            pars.mode = 'single_octave';
            pars.color_type = 'gray';
            pars.pooling_type = 'log_avg';
            pars.weighting_type = 'uniform';
            pars.base_scales = [1 3 5 7];
            pars.STEP = 2;
            pars.color_type = 'gray';
            pars.fg = {'figure'};
            pars.do_img_masking = true;
            pars.spatial_pyr =  false;
            pars.main_feat = {'really_dense_sift'};
            pars.enrichments{1} = {'rgb', 'hsv', 'lab', 'xy', 'scale'}; %
            SvmSegm_extract_measurements_on_masks(mask_type, 'pooling_masked', pars, img_names, exp_dir);
        elseif(strcmp(feat_types{i}, 'SIFT_GRAY_f_g'))
            pars = [];
            pars.name = 'SIFT_GRAY_f_g';
            pars.mode = 'single_octave';
            pars.color_type = 'gray';
            pars.pooling_type = 'log_avg';
            pars.weighting_type = 'uniform';
            pars.base_scales = [2 4 6 8];
            pars.STEP = 4;
            pars.color_type = 'gray';
            pars.spatial_pyr =  false;
            pars.main_feat = {'really_dense_sift'};
            pars.fg = {'figure', 'ground'};
            pars.enrichments{1} = {'rgb', 'hsv', 'lab', 'xy', 'scale'};
            pars.enrichments{2} = {'rgb', 'hsv', 'lab', 'xy_fullimg', 'scale_fullimg'};
            pars.sum_type = 'cache_sp_v2';
            pars.sum_sp_type = 'ucm';
            pars.sum_approx = -1;
            SvmSegm_extract_measurements_on_masks(mask_type, 'pooling_v3', pars, img_names, exp_dir);
        elseif(strcmp(feat_types{i}, 'LBP_f'))
            pars = [];
            pars.name = 'LBP_f';
            pars.mode = 'single_octave';
            pars.color_type = 'gray';
            pars.pooling_type = 'log_avg';
            pars.weighting_type = 'uniform';
            pars.fg = {'figure'};
            pars.STEP = 4; % not used
            pars.base_scales = [2 4 6 8 10];
            pars.spatial_pyr =  false;
            pars.main_feat = {'really_dense_lbp'};
            pars.enrichments{1} = {'rgb', 'hsv', 'lab', 'xy', 'scale'};
            pars.sum_type = 'cache_sp_v2';
            pars.sum_approx = -1;
            pars.sum_sp_type = {'ucm'};
            SvmSegm_extract_measurements_on_masks(mask_type, 'pooling_v3', pars, img_names, exp_dir);
        else
            error('No such feature available.');
        end
    end
end
