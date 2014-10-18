function o2p_train(exp_dir, imgset_train, mask_type, gt_mask_type, feat_collection, range_classes, lc, svr_par, MAX_CHUNK, CACHE_AGG_FEATS, name)
  DefaultVal('*lc', '0.3');
  DefaultVal('*range_classes', '1:20');
  DefaultVal('*svr_par', '0.25');
  DefaultVal('*MAX_CHUNK', '450000');
  DefaultVal('*CACHE_AGG_FEATS', 'true');
  DefaultVal('*name', '[]');
  
  cache_dir = [exp_dir '/Cache/'];
  if(~exist(cache_dir, 'dir'))
      mkdir(cache_dir);
  end
  
  % parameters
  linear_model_type = 'svr';
  svr_prec = 0.01;  
  reduce_by_inference = true;
  WARM_START = true;
  concat_SVS = true;
  N_PASSES = 1; % if you want to iterate multiple times over the training data (only correctly functioning with 1 pass right now)

  imgset_train_GT = imgset_train;
  imgset_train_BS = imgset_train;
  
  if(isempty(name))
    name = [feat_collection '_' linear_model_type '_mining_2012_' imgset_train '_' mask_type];
  end 
  
  [feats, power_scaling, input_scaling_type, feat_weights] = feat_config(feat_collection); 
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%% training 1: get segment qualities %%%%%%%%%%%%%%%%%%%%%%%  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

  browser_train = SegmBrowser(exp_dir, mask_type, imgset_train);

  whole_train_ids = 1:numel(browser_train.whole_2_img_ids);      
  [y_train_all] = browser_train.get_overlaps_wholes(whole_train_ids);
  y_train_all = single(y_train_all);  
  
  browser_train_GT = SegmBrowser(exp_dir, gt_mask_type, imgset_train_GT);
  whole_train_ids_GT = 1:numel(browser_train_GT.whole_2_img_ids);
  y_train_GT = browser_train_GT.get_overlaps_wholes(whole_train_ids_GT);
  [~, labels_y_train_GT] = max(y_train_GT,[], 2);

  browser_train_GT_mirror = SegmBrowser(exp_dir, gt_mask_type, [imgset_train_GT '_mirror']);
  whole_train_ids_GT_mirror = 1:numel(browser_train_GT_mirror.whole_2_img_ids);
  y_train_GT_mirror = browser_train_GT_mirror.get_overlaps_wholes(whole_train_ids_GT_mirror);
  [~, labels_y_train_GT_mirror] = max(y_train_GT,[], 2);  

  N_CHUNKS = ceil(numel(browser_train.whole_2_img_ids)/MAX_CHUNK);
  chunks = chunkify(1:numel(browser_train.whole_2_img_ids), N_CHUNKS);  
  
  % select best object segments above a certain threshold
  browser_train_BS = SegmBrowser(exp_dir, mask_type, imgset_train_BS);
  BEST_SEGMENT_THRESH = 0.5; % 0.5
  best_segms = browser_train_BS.get_best_wholes(1:numel(browser_train_BS.global_2_local_obj_ids));
  y_best_segms = browser_train_BS.get_overlaps_wholes(best_segms);
  m = max(y_best_segms,[],2);
  best_good = (m>BEST_SEGMENT_THRESH);  
  y_best_segms = y_best_segms(best_good,:);  
  best_segms = best_segms(best_good);
  % remove repeated
  [best_segms, ids] = unique(best_segms);
  y_best_segms = y_best_segms(ids, :);  
  % keep those belonging to the right class
  [~, labels_best_segms] = max(y_best_segms, [], 2);
  
  % set mean of outputs to zero, for each class
  m = mean([y_train_GT; y_train_GT_mirror; y_train_all]);
  y_train_GT = bsxfun(@minus, y_train_GT, m);
  y_train_GT_mirror = bsxfun(@minus, y_train_GT_mirror, m);
  y_train_all = bsxfun(@minus, y_train_all, m);
  
  for g=1:numel(lc)      
      % clean up old tmp data
      train_cache_file_SVs = [cache_dir imgset_train '_' feat_collection '_SVs_' mask_type];
      if(exist([train_cache_file_SVs '.bin'], 'file'))
          system(['rm ' train_cache_file_SVs '.bin']);
          system(['rm ' train_cache_file_SVs '_dims.mat']);
      end
      
      % create folder to save models in
      folder_models = [exp_dir 'MODELS/' name '_' sprintf('%f', lc(g)) '/'];
      if(~exist(folder_models, 'dir'))
          mkdir(folder_models);
      end

      sv_whole_ids = [];
      new_alphas = [];
      zero_importance_svs = cell(20,1);
      non_zero_importance = zero_importance_svs;
      
      for h=1:N_PASSES          
          for i=1:N_CHUNKS               
              Feats = [];
              all_SV_ids = cell(numel(range_classes),1);
              
              if CACHE_AGG_FEATS
                train_cache_file_GT = [cache_dir imgset_train_GT '_' feat_collection '_' gt_mask_type];                        
                train_cache_file_best_segms = [cache_dir imgset_train_BS '_' feat_collection '_BS_' int2str(BEST_SEGMENT_THRESH) '_' mask_type];
                train_cache_file_segms = [cache_dir imgset_train '_' feat_collection '_chunk_' int2str(i) '_of_' int2str(N_CHUNKS) '_' mask_type];
              else                  
                train_cache_file_GT = [];
                train_cache_file_best_segms = [];
                train_cache_file_segms = [];
              end
              
              % save data into file, if not done before
              feat_loading_wrapper_altered(browser_train_GT, whole_train_ids_GT, feats, input_scaling_type, power_scaling, train_cache_file_GT, 'GT', feat_weights);
              feat_loading_wrapper_altered(browser_train_GT_mirror, whole_train_ids_GT_mirror, feats, input_scaling_type, power_scaling, train_cache_file_GT, 'GT_mirror', feat_weights);
              n_GT_examples = numel(whole_train_ids_GT) + numel(whole_train_ids_GT_mirror);
              
              feat_loading_wrapper_altered(browser_train_BS, best_segms, feats, input_scaling_type, power_scaling, train_cache_file_best_segms, 'BestSegms', feat_weights);
              n_GT_examples = n_GT_examples + numel(best_segms);
              
              feat_loading_wrapper_altered(browser_train, chunks{i}, feats, input_scaling_type, power_scaling, train_cache_file_segms, 'Segms', feat_weights);

              y_train = [y_train_all(chunks{i},:); y_train_GT; y_train_GT_mirror; y_best_segms];
              if(concat_SVS && exist([train_cache_file_SVs '.bin'], 'file'))                  
                all_caches = {train_cache_file_segms, train_cache_file_GT, train_cache_file_best_segms, train_cache_file_SVs};
                y_train = [y_train; y_train_SVs];
              else
                all_caches = {train_cache_file_segms, train_cache_file_GT, train_cache_file_best_segms};
              end          
              t_load = tic();
              [Feats, dims] = feat_loading_wrapper_altered([], [], [], [], [], all_caches);
              t_load = toc(t_load)
              gt_ids = (numel(chunks{i})+1):(numel(chunks{i})+n_GT_examples);

              for class_id=range_classes
                  non_zero_importance{class_id} = 1:size(Feats,2);
              end
              
              if(reduce_by_inference && (h~=1 || i~=1))
                  % load models and use them to find examples violating the margin
                  if(i==1)
                    [models, w] = load_models(folder_models, browser_train.categories(range_classes), h-1, N_CHUNKS);  
                  else                      
                    [models, w] = load_models(folder_models, browser_train.categories(range_classes), h, i-1);
                  end
                  
                  y_pred_chunk = w'*Feats;    
                  if(strcmp(linear_model_type, 'svm'))
                      lbl = single(y_train(:,range_classes)>0);
                      lbl(lbl==0) = -1;
                      err = (lbl.*y_pred_chunk')<1;
                      viol = err;
                  elseif(strcmp(linear_model_type, 'svr'))
                      err = abs((y_train(:,range_classes)) - y_pred_chunk');
                      viol = err>svr_par;
                  else
                      error('No such linear model type');
                  end
                       
                  counter = 1;
                  for class_id=range_classes
                      % add to the list any new example that is incorrectly
                      % predicted 
                      n_new = numel(chunks{i});
                      n_new_plus_gt = n_new + numel(gt_ids);
                      non_zero_importance{class_id} = find(viol(1:n_new,counter)'); % & ...
                         %(y_train(1:n_new,class_id)<=0)');
                      
                      % add also all gt examples
                      non_zero_importance{class_id} = [non_zero_importance{class_id} gt_ids];
                      
                      % add also the SVs from the previous chunk
                      non_zero_importance{class_id} = [non_zero_importance{class_id} ...
                         setdiff((n_new_plus_gt+1):size(Feats,2), zero_importance_svs{class_id})];
                     
                      non_zero_importance{class_id} = sort(non_zero_importance{class_id}, 'ascend');
                      
                      counter = counter + 1;
                  end                  
              end
              
              for class_id=range_classes
                  t_batch_learn = tic();
                  if (h==1 && i==1)
                      alphas = zeros(size(Feats,2),1, 'single');
                      w = zeros(size(Feats, 1),1, 'single');
                  else
                     if(i==1)
                         this_model = load_models(folder_models, browser_train.categories(class_id), h-1, N_CHUNKS);                      
                     else
                         this_model = load_models(folder_models, browser_train.categories(class_id), h, i-1);
                     end
                     w = (this_model{1}.w)';
                     alphas = new_alphas{class_id};
                  end
                  
                  instance_weights = zeros(size(Feats,2),1, 'single');
                  instance_weights(non_zero_importance{class_id}) = 1;

                  % zero weight to segments overlapping the object but
                  % are not the best if svm
                  if(strcmp(linear_model_type, 'svm') || strcmp(linear_model_type, 'sgd'))
                      overlapping_obj = find(y_train(1:numel(chunks{i}),class_id) > 0);
                      instance_weights(overlapping_obj) = 0;
                  end
                  instance_weights(zero_importance_svs{class_id}) = 0;
                  
                  if 1
                    % train linear model
                    if(WARM_START)
                        model = train_liblinear(Feats, y_train(:,class_id), single(lc(g)), alphas, w, instance_weights, linear_model_type, single(svr_par), single(svr_prec));
                    else
                        model = train_liblinear(Feats, y_train(:,class_id), single(lc(g)), zeros(size(alphas), 'single'), zeros(size(w), 'single'), instance_weights, linear_model_type, single(svr_par), single(svr_prec));
                    end

                    n_SVs = numel(model.SVids)
                    if 0 
                        % disabled because it is expensive wrt memory
                        cost_svr(h,i,class_id) = svr_cost((model.w*Feats)', y_train(:,class_id), instance_weights*lc(g), model.w, svr_par);
                        last_cost_svr = cost_svr(:,i,class_id)
                        cost_fit(h,i,class_id) = mean(abs((y_train(:, class_id))' - model.w*Feats));
                    end

                      all_SV_ids{class_id} = model.SVids;  
                  end
                      
                  
                  % save model
                  category = browser_train.categories{class_id};

                  if(~((h==N_PASSES) && (i==N_CHUNKS)))
                    file_to_save = [folder_models category '_iter_' int2str(h) '_chunk_' int2str(i) '.mat'];
                  else
                    file_to_save = [folder_models category '.mat'];
                  end

                  mysave(file_to_save, 'model', model);
                  t_batch_learn = toc(t_batch_learn)
              end
                            
              if(~((h==N_PASSES) && (i==N_CHUNKS)))
                  % keep unique ids that do not correspond to GT segments (those we
                  % add always)
                  all_SV_ids_un = unique(cell2mat(all_SV_ids'));
                  all_SV_ids_un = setdiff(all_SV_ids_un, numel(chunks{i})+1:numel(chunks{i})+n_GT_examples);
                  
                  SvFeats = Feats(:,all_SV_ids_un);
                  y_train_SVs = y_train(all_SV_ids_un,:);

                  if(i==N_CHUNKS)
                      next_chunk = 1;
                  else
                      next_chunk = i+1;
                  end
                  % set zero importance for SVs from other classes
                  for class_id=range_classes
                      [~, zero_importance_svs{class_id}] = setdiff(all_SV_ids_un, all_SV_ids{class_id});
                      zero_importance_svs{class_id} = zero_importance_svs{class_id} + (numel(chunks{next_chunk}) + n_GT_examples);
                  end
                  
                % save svs
                feat_loading_wrapper_altered(browser_train, [], feats, input_scaling_type, power_scaling, train_cache_file_SVs, 'SV_Segms', feat_weights, SvFeats, dims);                        
                SvFeats = [];
                                
                new_alphas = cell(numel(range_classes),1);
                
                for class_id=range_classes
                    category = browser_train.categories(class_id);
                    this_model = load_models(folder_models, category, h, i);
                    this_model = this_model{1};
                    
                    [inters_GT, gt_sv_ids_a, gt_sv_ids_b] = intersect(gt_ids, this_model.SVids);
                    
                    new_alphas{class_id} = zeros(numel(chunks{next_chunk}) + n_GT_examples + numel(all_SV_ids_un), 1, 'single');
                    new_alphas{class_id}(numel(chunks{next_chunk}) + gt_sv_ids_a) = this_model.alphas(gt_sv_ids_b);
                    
                    if(concat_SVS)
                        [setd, non_gt_svs] = setdiff(this_model.SVids, inters_GT);
                        %assert(all(union(gt_sv_ids_b, non_gt_svs) == 1:numel(this_model.SVids)));
                        
                        [~, sv_ids_a] = intersect(all_SV_ids_un, setd);
                        alpha_ids = numel(chunks{next_chunk})+n_GT_examples+sv_ids_a;
                        assert(max(alpha_ids)<=numel(new_alphas{class_id}));
                        
                        new_alphas{class_id}(alpha_ids) = this_model.alphas(non_gt_svs);
                    else
                        new_alphas{class_id}((numel(chunks{next_chunk}) + n_GT_examples +1):end,:) = [];
                    end
                end
              end
          end
      end
  end
end
