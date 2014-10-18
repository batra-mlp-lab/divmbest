function segs = get_div_semantic_seg(img, masks, lambda, num_solutions)

  if nargin<2
	masks=[];
  end
  if nargin<3 || isempty(lambda)
	lambda=0.08;
  end
  if nargin<4 || isempty(num_solutions)
	num_solutions=30;
  end
  addpaths();


  %%%%%%%%%%% Preparing data and necessary variables %%%%%%%%
  overwrite=false;
  gt_mask_type='ground_truth_sp_approx';
  mask_type = 'CPMC_segms_150_sp_approx';
  feat_collection = 'all_feats';
  PCA_DIMS = [5000 5000 2500];
  exp_dir='./temp_divmbest/';
  img_set='temp_set';
  img_names=prepare_data(exp_dir, img, masks,mask_type,img_set);

  classes = 1:20;
  categories = {};
  for i=classes,
    categories{i} = VOC09_id_to_classname(i);
  end



  %%%%%%%% Extracting O2P Features using the CPMC Masks %%%%%%%%
  o2p_extract_feats(exp_dir, mask_type, feat_collection, img_names);
  o2p_project_pca_noncent(exp_dir, img_names, gt_mask_type, mask_type, feat_collection, PCA_DIMS);



  %%%%%%%%%%%% Generating diverse segmentations %%%%%%%%%%%%55
  mask_type_ho = 'CPMC_segms_150_sp_approx';
  feat_collection = 'all_feats_pca_noncent_5000';
  %div_type = 'divmbest';
  div_type = 'perturb';
  cache_dir = [exp_dir 'Cache/'];
  segs=[];

  [feats, power_scaling, input_scaling_type, feat_weights] = feat_config(feat_collection);
  if(all(feat_weights==1))
      feat_weights = [];
  end

  % Creating temporary set
  imgset_ho='temp_set';
  if(~strcmp(mask_type_ho, 'CPMC_150_segms'))
      ho_cache_file = [cache_dir  imgset_ho '_' feat_collection '_mask_' mask_type_ho '_ps_' int2str(power_scaling) '_scaling_' input_scaling_type];
  else
      ho_cache_file = [cache_dir imgset_ho '_' feat_collection '_sqrt_' int2str(power_scaling) '_scaling_' input_scaling_type];
  end

  browser_ho = SegmBrowser(exp_dir, mask_type_ho, imgset_ho);
  browser_ho.VOCopts.testset = imgset_ho;
  whole_ho_ids = 1:numel(browser_ho.whole_2_img_ids);

  SvmSegm_compute_overlaps(exp_dir, browser_ho.img_names, mask_type_ho);
  chunked_whole_ho_ids = chunkify(whole_ho_ids, ceil(numel(whole_ho_ids)/5000));

  feat = browser_ho.get_whole_feats(1, feats, input_scaling_type, feat_weights);

  beta = zeros(20, size(feat,1));
  for i=1:numel(classes)
      var = load([exp_dir 'MODELS/' browser_ho.categories{classes(i)} '.mat']);
       beta(i,:) = var.model.w;
  end

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

        THRESH = 0.20;%used to be 0.28  % any sufficiently small value to start search with 
        y_pred(21,:) = THRESH; % Background score

        for i=1:num_solutions,

          CURRTHRESH = THRESH;
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

              NMS_MAX_OVER = 1; % spatial non-maximum supression (1 means nms is not performed)
              NMS_MAX_SEGMS = 3; % max number of segments per image            
              SIMP_BIAS_STEP = 0.02; % background threshold increase for each additional segment above 1

              % max number of segments per image on average (used to set the background threshold)
              % we set this value to the average number of objects in the
              % training set. Of course, that is just a coincidence. ;-)
              MAX_AVG_N_SEGM = 2.2;

	      tic;
              while(n_segms_per_img > MAX_AVG_N_SEGM)
                  y_pred(setdiff(1:20, classes),:) = -10000; % remove non-selected classes

                  [local_ids, labels, scores, global_ids] = nms_inference_simplicity_bias(browser_ho, y_pred, nms_type, whole_ho_ids, NMS_MAX_OVER, NMS_MAX_SEGMS, SIMP_BIAS_STEP);
                  n_segms_per_img = numel(cell2mat(labels')) / numel(labels);

                  CURRTHRESH = CURRTHRESH+0.01;
                  y_pred(21,:) = y_pred(21,:)+0.01;
              end
              t=toc;
              fprintf('Time to run inference %f sec\n',t);
              CURRTHRESH = CURRTHRESH-0.01;
              all_THRESH(1,i) = CURRTHRESH;
              all_n_segms(1,i) = n_segms_per_img;

	      browser_ho.VOCopts.testset = imgset_ho;
              this_browser_ho = browser_ho;

              % output png's will be in ./<exp_dir>/results/*
              predsegs =  this_browser_ho.voc_segm_outputs(1:numel(browser_ho.img_names), global_ids, labels, scores, imgset_ho, true, segms2pixel_algo, i, true, y_pred, CURRTHRESH,lambda);
	      segs=[segs; predsegs];

              if(strcmp(div_type, 'perturb'))
                if(i==1)
                        orig_scores = y_pred;
                end
                y_pred = orig_scores;
              end

              % subtract lambda from scores of class labels of segments
              y_pred = subtract_lambda (browser_ho, lambda, y_pred, global_ids, labels, whole_ho_ids, THRESH, div_type);

	  end
	end




function img_names=prepare_data(exp_dir, img, masks, mask_type,img_set)

	if(~exist(exp_dir,'dir'))
		!wget https://filebox.ece.vt.edu/~senthil/divseg_o2p_env.tar.gz
		!tar xfz divseg_o2p_env.tar.gz
	end

	if(isstr(img))
		system(['cp ' img ' ' exp_dir 'JPEGImages/temp_img.jpg']);
		img_names={'temp_img'};
	elseif(iscellstr(img))
		img_names={};
		for i=1:length(img)
			system(['cp ' img{i} ' ' exp_dir 'JPEGImages/temp_img' num2str(i) '.jpg']);
			filename=['temp_img' num2str(i)];
			img_names=[img_names; filename];
		end
	else
		imwrite(img,[exp_dir 'JPEGImages/temp_img.jpg']);
		img_names={'temp_img'};
	end


	%%%%% CPMC Segments for must be in ./temp_divmbest/MySegmentsMat/<mask_type>/
	if nargin<2 || isempty(masks)
		%%% Extract CPMC
		for i=1:length(img_names)
			[masks, scores] = cpmc(exp_dir, img_names{i});
		end
	elseif(isstr(masks))
		system(['cp ' masks ' ' exp_dir 'MySegmentsMat/' mask_type '/temp_img.mat']);
	elseif(iscellstr(masks))
		for i=1:length(masks)
			system(['cp ' masks{i} ' ' exp_dir 'MySegmentsMat/' mask_type '/temp_img' num2str(i) '.mat']);
		end
	else
		save([exp_dir 'MySegmentsMat/' mask_type '/temp_img.mat'],'masks');
	end  


	%%%%% The list of images should be in <exp_dir>/ImageSets/Segmentation/<img_set>.txt
	f=fopen([exp_dir 'ImageSets/Segmentation/' img_set '.txt'],'w');
	for i=1:length(img_names)
	fprintf(f,'%s\n', img_names{i});
	end
	fclose(f);

	remove_cpmc_paths();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function addpaths()
    addpath('../src/');
    addpath('../src/SegmBrowser/');
    addpath('../VOC_experiment/');
    %addpath('../VOC_experiment/sift_voc_exp');
    addpath('../external_src/');
    addpath('../external_src/vlfeats/toolbox/mex/mexa64/');    
    addpath('../external_src/vlfeats/toolbox/sift/');
    addpath('../external_src/vlfeats/toolbox/misc/');
    addpath('../external_src/vlfeats/toolbox/mex/mexa64/');
    addpath('../external_src/ndSparse/');
    addpath('../external_src/VOCcode/');
    addpath('../src/liblinear-custom/matlab/');
    addpath('../external_src/immerge/');

%  addpath('../external_src/');
%  addpath('../external_src/vlfeats/toolbox/mex/mexa64/');
%  addpath(genpath('../external_src/vlfeats/toolbox/'));
%  addpath('../external_src/vlfeats/toolbox/mex/mexa64/');
%  addpath('../external_src/vlfeats/toolbox/sift/');
%  addpath('../external_src/vlfeats/toolbox/misc/');
%  addpath('../external_src/vlfeats/toolbox/imop/');
%  addpath('../external_src/ndSparse/');
%  addpath('../external_src/VOCcode/');
  addpath('../external_src/cpmc_release1/');
  addpath('../external_src/cpmc_release1/code/');
  addpath('../external_src/cpmc_release1/external_code/');
  addpath('../external_src/cpmc_release1/external_code/paraFmex/');
  addpath('../external_src/cpmc_release1/external_code/imrender/vgg/');
  addpath('../external_src/cpmc_release1/external_code/immerge/');
  addpath('../external_src/cpmc_release1/external_code/color_sift/');
  addpath('../external_src/cpmc_release1/external_code/vlfeats/toolbox/kmeans/');
  addpath('../external_src/cpmc_release1/external_code/vlfeats/toolbox/kmeans/');
  addpath('../external_src/cpmc_release1/external_code/vlfeats/toolbox/mex/mexa64/');
  addpath('../external_src/cpmc_release1/external_code/vlfeats/toolbox/mex/mexglx/');
  addpath('../external_src/cpmc_release1/external_code/globalPb/lib/');
  addpath('../external_src/cpmc_release1/external_code/mpi-chi2-v1_5/');
  %addpath('../src/');
  %addpath('../src/SegmBrowser/');




function remove_cpmc_paths()
  rmpath('../external_src/cpmc_release1/');
  rmpath('../external_src/cpmc_release1/code/');
  rmpath('../external_src/cpmc_release1/external_code/');
  rmpath('../external_src/cpmc_release1/external_code/paraFmex/');
  rmpath('../external_src/cpmc_release1/external_code/imrender/vgg/');
  rmpath('../external_src/cpmc_release1/external_code/immerge/');
  rmpath('../external_src/cpmc_release1/external_code/color_sift/');
  rmpath('../external_src/cpmc_release1/external_code/vlfeats/toolbox/kmeans/');
  rmpath('../external_src/cpmc_release1/external_code/vlfeats/toolbox/mex/mexa64/');
  rmpath('../external_src/cpmc_release1/external_code/vlfeats/toolbox/mex/mexglx/');
  rmpath('../external_src/cpmc_release1/external_code/globalPb/lib/');
  rmpath('../external_src/cpmc_release1/external_code/mpi-chi2-v1_5/');




function [Feat, dims, scaling] = load_features(exp_dir, mask_type, img_names, feat_types, scaling_type, weights, power_scaling)
	
	if nargin<8
		power_scaling=false;
	end

	the_folder = 'MyMeasurements/';
	d=0;
	dims=[];

	for i=1:numel(feat_types)
            D = myload([exp_dir the_folder mask_type '_' feat_types{i} '/' img_names{1} '.mat'], 'D');
            d = d + size(D,1);
            dims(i) = size(D,1);
        end


	Feats = zeros(d, numel(img_names), 'single');
	feat_ranges = [0 cumsum(dims)];


	for j=1:numel(feat_types)
            counter = 1;
            range{j} = (feat_ranges(j)+1):feat_ranges(j+1);

	    for i=1:numel(un_img_ids)
              vgg_progressbar('feature loading', i/numel(img_names), 5);
	      D = myload([exp_dir the_folder mask_type '_' feat_types{j} '/' img_names{i} '.mat'], 'D');
	      Feats(range{j}, counter) = D(:,local_ids);
	      counter=counter+1;
	    end
	    
	    if(~isempty(scaling_type))
                if(strcmp(scaling_type, 'norm_2') || strcmp(scaling_type, 'norm_1'))
                    chunks = chunkify(1:size(Feats,2), 10);
                    for k=1:numel(chunks) % necessary if data fills up most of memory
                        [Feats(range{j}, chunks{k}), scaling{j}] = scale_data(Feats(range{j},chunks{k}), scaling_type);
                    end
                else
                    if(~strcmp(scaling_type, 'none'))
                        error('not ready for this');
                    end
                end
            else
              scaling{j} = [];
            end

            if(~isempty(weights) && weights(j)~=1)
                Feats(range{j},:) = Feats(range{j},:)*weights(j);
            end
	end

	if((numel(feat_types)>1) && (strcmp(scaling_type, 'norm_2') || strcmp(scaling_type, 'norm_1')))
            chunks = chunkify(1:size(Feats,2), 10);
            for k=1:numel(chunks) % necessary if data fills up most of memory
                Feats(:,chunks{k}) = scale_data(Feats(:,chunks{k}), scaling_type);
            end
	end

	if(power_scaling)
        	Feats = squash_features(Feats, 'power');
    	end
