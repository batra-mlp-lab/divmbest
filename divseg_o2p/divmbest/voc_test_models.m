function [accuracies,avacc,conf,rawcounts,name]=voc_test_models(lambda, num_solutions, store, imgset_ho, THRESH, name)

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


    DefaultVal('*lambda', '0');
    DefaultVal('*num_solutions', '1');
    DefaultVal('*store','false');

    DefaultVal('*imgset_ho','''val12''');
    DefaultVal('*name','{''all_gt_segm_minus_val12_all_feats_pca_noncent_5000_3.000000''}');

    exp_dir = './VOC12/';
    mask_type_ho = 'CPMC_segms_150_sp_approx';
    feat_collection = 'all_feats_pca_noncent_5000';
    
%    type = 'divmbest';
    type = 'perturb';
 
    if 0
      imgset_ho = 'val11';    
      name = {'all_gt_segm_minus_val11_all_feats_pca_noncent_5000_3.000000'};
    elseif 0
      imgset_ho = 'test11';
      name = {'all_gt_segm_all_feats_pca_noncent_5000_3.000000'};
    elseif 0
      imgset_ho = 'val12';
      name = {'all_gt_segm_minus_val12_all_feats_pca_noncent_5000_3.000000'};
    elseif 0
      imgset_ho = 'test12';
      name = {'all_gt_segm_all_feats_pca_noncent_5000_3.000000'};
    else
	name={['all_gt_segm_minus_' imgset_ho '_all_feats_pca_noncent_5000_3.000000']};
    end
    
    cache_dir = [exp_dir '/Cache/'];

    classes = 1:20;

    [feats, power_scaling, input_scaling_type, feat_weights] = feat_config(feat_collection);
    if(all(feat_weights==1))
        feat_weights = [];
    end
    
    if(~strcmp(mask_type_ho, 'CPMC_150_segms'))
        ho_cache_file = [cache_dir  imgset_ho '_' feat_collection '_mask_' mask_type_ho '_ps_' int2str(power_scaling) '_scaling_' input_scaling_type];
    else
        ho_cache_file = [cache_dir imgset_ho '_' feat_collection '_sqrt_' int2str(power_scaling) '_scaling_' input_scaling_type];
    end

    MAX_INPUT_CHUNK = 450000;
        
    browser_ho = SegmBrowser(exp_dir, mask_type_ho, imgset_ho);
    browser_ho.VOCopts.testset = imgset_ho;

    whole_ho_ids = 1:numel(browser_ho.whole_2_img_ids);
    
	disp('Matlabpool...');
	
    % create multiple threads (set how many you have)
    N_THREADS = 12;
    %if(matlabpool('size')~=N_THREADS)
    %    matlabpool('open', N_THREADS);
    %end    
    
    % computes pairwise overlaps, which can be helpful in inference if
    % you want to experiment with any form of non-maximum supression (it will be cached)
    % the default is to have no non-maximum supression.
    SvmSegm_compute_overlaps(exp_dir, browser_ho.img_names, mask_type_ho); 
    
    chunked_whole_ho_ids = chunkify(whole_ho_ids, ceil(numel(whole_ho_ids)/MAX_INPUT_CHUNK));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%% Load ground truth %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(~strcmp(imgset_ho(1:4), 'test'))
      gtsegs = cell (1,numel (browser_ho.img_names));
      for k = 1:numel (browser_ho.img_names)
        gtfile = sprintf (browser_ho.VOCopts.seg.clsimgpath, browser_ho.img_names{k});
        gtsegs{k} = imread (gtfile);    
      end

      oraclesegs = cell (1,numel (browser_ho.img_names));
      oraclesegaccs = -inf (1,numel (browser_ho.img_names));
      oraclesegids = zeros (1,numel (browser_ho.img_names));
      oraclesegids_vsM = cell(1,num_solutions);
      oracleavacc_vsM = cell(1,num_solutions); 
    end
	
	
	disp('Doing work...');

	
	secret_dir = sprintf(browser_ho.VOCopts.seg.origcmpcscorerespath,name{1},browser_ho.VOCopts.testset,sprintf('lambda_%s',num2str(lambda)),browser_ho.img_names{1},browser_ho.img_names{1})
	

    for h=1:numel(name)
        % load models
        feat = browser_ho.get_whole_feats(1, feats, input_scaling_type, feat_weights);
        
        beta = zeros(20, size(feat,1));
        for i=1:numel(classes)            
            var = load([exp_dir 'MODELS/' name{h} '/' browser_ho.categories{classes(i)} '.mat']);
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

        save_original_cpmc_seg_scores(y_pred, browser_ho, name{h}, lambda);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%% Generate desired outputs %%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %THRESH = 0.20;%used to be 0.28  % any sufficiently small value to start search with 
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
                  n_segms_per_img = numel(cell2mat(labels')) / numel(labels)

                  CURRTHRESH = CURRTHRESH+0.01;
                  y_pred(21,:) = y_pred(21,:)+0.01;
              end
              t=toc;
              fprintf('Time to run inference %f sec\n',t);
              CURRTHRESH = CURRTHRESH-0.01;
              all_THRESH(h,i) = CURRTHRESH;
              all_n_segms(h,i) = n_segms_per_img

              %name{h}
              if 0 && (~strcmp(imgset_ho(1:4), 'test'))
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
              
              % output png's will be in ./VOC/results/*
	      predsegs =  this_browser_ho.voc_segm_outputs(1:numel(browser_ho.img_names), global_ids, labels, scores, name{h}, ~store, segms2pixel_algo, i, true, y_pred, CURRTHRESH,lambda);
              
              if(~strcmp(imgset_ho(1:4), 'test'))
                %[accuracies{h},avacc,conf,rawcounts] = VOCevalseg(this_browser_ho.VOCopts, name{h});
                for k = 1:numel(this_browser_ho.img_names),
                  [accs] = VOCevalseg_frommem (this_browser_ho.VOCopts, name{h}, {predsegs{k}-1}, {gtsegs{k}}, false);
                  avg_seg_accs(k,i) = mean(accs(~isnan(accs)));

                  if avg_seg_accs(k,i) > oraclesegaccs(k),
                    oraclesegs{k} = predsegs{k}-1;
                    oraclesegaccs(k) = avg_seg_accs(k,i);
                    oraclesegids(k) = i;
                  end
                end
              end

	      if(strcmp(type, 'perturb'))
		if(i==1)
			orig_scores = y_pred;
		end
		y_pred = orig_scores;
	      end

              % subtract lambda from scores of class labels of segments
              y_pred = subtract_lambda (browser_ho, lambda, y_pred, global_ids, labels, whole_ho_ids, THRESH, type);

              if(~strcmp(imgset_ho(1:4), 'test'))
                [~,oracleavacc_vsM{i}] = VOCevalseg_frommem (this_browser_ho.VOCopts, name{h}, oraclesegs, gtsegs);   
                oraclesegids_vsM{i} = oraclesegids;
              end

              if 0
                  % this saves images with label transparencies in folder
                  % "visuals"
                  this_browser_ho.voc_segm_visuals([exp_dir '/results/VOC2012/Segmentation/' name{h} '_' imgset_ho '_cls/'], ...
                      [exp_dir '/JPEGImages/'], name{h}, true, i);
              end
          end 
        end % num_solutions         
        this_browser_ho = browser_ho;

        if(~strcmp(imgset_ho(1:4), 'test'))
          [accuracies{h},avacc{h},conf{h},rawcounts{h}] = VOCevalseg_frommem (this_browser_ho.VOCopts, name{h}, oraclesegs, gtsegs);   
        end

        if store & ~strcmp(imgset_ho(1:4),'test'), 
          save_avg_seg_acc(this_browser_ho, name{h}, avg_seg_accs, 1:numel(browser_ho.img_names),num_solutions,lambda);
          oracledir = sprintf([this_browser_ho.VOCopts.seg.divsolresdir '/oracle/'],name{h},this_browser_ho.VOCopts.testset,sprintf('lambda_%s',num2str(lambda))); 
          copy_oraclesegs_to_oracle_folder(this_browser_ho, name{h}, 1:numel(browser_ho.img_names), oraclesegids, oracledir,lambda);
          save_oracle_info(this_browser_ho, 1:numel(browser_ho.img_names),oraclesegids, accuracies,avacc,oracleavacc_vsM, oraclesegids_vsM, oracledir);
        end

        if 0
          % this saves images with label transparencies in folder
          % "visuals"
          oracledir = sprintf([this_browser_ho.VOCopts.seg.divsolresdir '/oracle/'],name{h},this_browser_ho.VOCopts.testset,sprintf('lambda_%s',num2str(lambda))); 
          this_browser_ho.voc_segm_visuals(oracledir,[exp_dir '/JPEGImages/'], name{h}, true, [], lambda);
        end
      end

    %cellfun(@mean, accuracies)    
end

function scores = subtract_lambda (nPBM, lambda, scores, global_ids, labels, whole_ids, THRESH, type)
  % subtract lambda from segment labels that were chosen in previous solution, set the background label to
  % THRESH and reduce THRESH by lambda for those segments that were assigned to background 
  n = hist (nPBM.whole_2_img_ids (whole_ids), numel(nPBM.img_names));
  whole_ids_cell = mat2cell (whole_ids, 1, n);
  
  scz = size (scores);

  for i = 1:length (n),
    foresegids = global_ids{i};
    backsegids = setdiff (whole_ids_cell{i}, foresegids); 
    foreindx = sub2ind (scz, labels{i}, foresegids);
   
    scores(21,:) = THRESH;
    if(strcmp(type, 'divmbest'))
	scores(foreindx) = scores(foreindx) - lambda;
	if ~isempty(backsegids),
	      scores(21,backsegids) = scores(21,backsegids) - lambda;
	end
    elseif(strcmp(type, 'perturb'))
	U = rand(scz);
	gumbel = log(-log(U));
	gumbel = lambda.*gumbel;
	scores = scores - gumbel; % Domain agnostic perturbation
	% scores(1:20,:) = scores(1:20,:) - gumbel(1:20,:); % Domain aware perturbation
    end
  end
end

function save_avg_seg_acc(nPBM, name, avg_seg_accs, img_ids, M, lambda)

 for k=1:length(img_ids),
    divsolaccdir = sprintf([nPBM.VOCopts.seg.divsolresdir '/%s/'], name, nPBM.VOCopts.testset, sprintf('lambda_%s',num2str(lambda)), nPBM.img_names{img_ids(k)});
    if(~exist(divsolaccdir,'dir'))
      mkdir(divsolaccdir);
    end
    seg_avg_acc_path = sprintf(nPBM.VOCopts.seg.solaccrespath,name,nPBM.VOCopts.testset,sprintf('lambda_%s',num2str(lambda)), nPBM.img_names{img_ids(k)},nPBM.img_names{img_ids(k)},M); 
    avg_seg_acc = avg_seg_accs(k,:);
    save(seg_avg_acc_path, 'avg_seg_acc');
  end
end

function copy_oraclesegs_to_oracle_folder(nPBM, name, img_ids, oraclesegids, oracledir,lambda)
  if (~exist(oracledir,'dir'))
    mkdir(oracledir);
  end
  for k=1:length(img_ids),
    dir_name = sprintf([nPBM.VOCopts.seg.divsolresdir '/%s'],name,nPBM.VOCopts.testset,sprintf('lambda_%s',num2str(lambda)),nPBM.img_names{img_ids(k)});
    system(sprintf('cp %s/%s_S%04d.png %s/%s.png',dir_name,nPBM.img_names{img_ids(k)},oraclesegids(k),oracledir,nPBM.img_names{img_ids(k)}));
  end
end

function save_oracle_info(nPBM, img_ids, oraclesegids, oracleaccs, oracleavacc, oracleavacc_vsM, oraclesegids_vsM, oracledir)
  if (~exist(oracledir,'dir'))
    mkdir(oracledir);
  end
  oraclefile = sprintf('%s/oracleinfo.mat',oracledir);
  img_names = nPBM.img_names;
  save(oraclefile, 'img_names','img_ids','oraclesegids','oracleaccs','oracleavacc','oracleavacc_vsM','oraclesegids_vsM'); 
end

function save_original_cpmc_seg_scores(y_pred, nPBM, name, lambda)
make_divsol_dir(y_pred, nPBM, name, lambda);

  for k=1:length(nPBM.img_names)
make_sol_dir(y_pred, nPBM, name, lambda,k);
    cpmcscorefile = sprintf(nPBM.VOCopts.seg.origcmpcscorerespath,name,nPBM.VOCopts.testset,sprintf('lambda_%s',num2str(lambda)),nPBM.img_names{k},nPBM.img_names{k})
    scores = y_pred(:,find(nPBM.whole_2_img_ids==k));
    save(cpmcscorefile,'scores');
  end
end


function make_divsol_dir(y_pred, nPBM, name, lambda)
	mkdir(fullfile('./VOC12/results/VOC2012_plus_berkeley/Segmentation',[name '_' nPBM.VOCopts.testset '_cls']));
	mkdir(fullfile('./VOC12/results/VOC2012_plus_berkeley/Segmentation',[name '_' nPBM.VOCopts.testset '_cls'],['divsol_lambda_' num2str(lambda)]));
end
	
function make_sol_dir(y_pred, nPBM, name, lambda,k)
	mkdir(fullfile('./VOC12/results/VOC2012_plus_berkeley/Segmentation',[name '_' nPBM.VOCopts.testset '_cls'],['divsol_lambda_' num2str(lambda)],nPBM.img_names{k}));
end
