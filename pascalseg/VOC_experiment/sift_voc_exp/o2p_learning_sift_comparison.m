function o2p_learning_sift_comparison()
    addpath('../../src/');
    addpath('../../src/SegmBrowser/');
    addpath('../../external_src/');
    addpath('../../external_src/vlfeats/toolbox/mex/mexa64/');
    addpath('../../external_src/vlfeats/toolbox/sift/');
    addpath('../../external_src/vlfeats/toolbox/misc/');
    addpath('../../external_src/vlfeats/toolbox/mex/mexa64/');
    addpath('../../external_src/ndSparse/');
    addpath('../../external_src/VOCcode/');
    addpath('../../external_src/libsvm-3.11/matlab/');

    s = RandStream('mt19937ar','Seed',1234);
    RandStream.setGlobalStream(s);

    imgset_train = 'train11';
    imgset_ho = 'val11';

    exp_dir = '../VOC/';

    all_feats = {
        'ECCV12_TABLE1_ENRICHED_SIFT_2AVGP_LOG', ...
        'ECCV12_TABLE1_ENRICHED_SIFT_2AVGP', ...
        'ECCV12_TABLE1_ENRICHED_SIFT_1AVGP', ...
        'ECCV12_TABLE1_ENRICHED_SIFT_2MAXP', ...
        'ECCV12_TABLE1_ENRICHED_SIFT_1MAXP', ...
        'ECCV12_TABLE1_SIFT_2AVGP_LOG', ...
        'ECCV12_TABLE1_SIFT_2AVGP', ...
        'ECCV12_TABLE1_SIFT_1AVGP', ...
        'ECCV12_TABLE1_SIFT_2MAXP', ...
        'ECCV12_TABLE1_SIFT_1MAXP'};
    
    mask_types = {'ground_truth', 'ground_truth_sp_approx'};        
    
    PowerNorm = true;
    
    for iter1=1:numel(all_feats)
        for iter2=1:numel(mask_types)        
            feat_sel = iter1;
            mask_type = mask_types{iter2};
            
            % Already searched over approximately best values for each
            % feature on ground_truth
            lc = [30 30 10000 30 10000 ...
                    30 30 10000 30 1000];
            
            feats = all_feats(feat_sel);
            lc = lc(feat_sel);
            scaling_type = ['norm_2']; % this is not required for 2AVGP_LOG, but some of the others have extreme ranges that make libsvm fail.. setting it for all
            weights = 1;
            
            MAX_ELEMS_TRAINING = 10000000000;
            MAX_ELEMS_TESTING = MAX_ELEMS_TRAINING;
            labels = 1:20;
            trainer = 'SVM';
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%% training %%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            browser_train_GT = SegmBrowser(exp_dir, mask_type, imgset_train);
            whole_train_ids = 1:numel(browser_train_GT.whole_2_img_ids);
            
            browser_ho_GT = SegmBrowser(exp_dir, mask_type, imgset_ho);
            whole_ho_ids = 1:numel(browser_ho_GT.whole_2_img_ids);
            
            [y_train] = browser_train_GT.get_overlaps_wholes(whole_train_ids);
            
            t_all = tic();
            if strcmp(trainer, 'SVM')
                % get dimensionality
                libsvm = cell(numel(labels),1);
                
                K = compute_K_train(browser_train_GT, MAX_ELEMS_TRAINING, feats, weights, scaling_type, PowerNorm, whole_train_ids);
                
                for j=1:numel(labels)
                    t_class = tic();
                    trainlabel = logical(y_train(:,j));
                    these_labels = -ones(numel(trainlabel),1);
                    these_labels(trainlabel) = 1;
                    
                    if 0
                        error('add linear trainer here');
                    elseif 1
                        if(j==1)
                            K = [(1:size(K,1))' double(K)];
                        end
                        libsvm{j} = svmtrain(these_labels, ...
                            K, sprintf(' -t 4 -c %f -q -p 0.0001', lc));
                        %K, sprintf(' -t 4 -c %f -q -p 0.00001', lc));
                        
                        % compute w vector
                        [SVids, srt] = sort(libsvm{j}.SVs, 'ascend');
                        SV_alpha = libsvm{j}.sv_coef(srt);
                        
                        [Feats_one_sample, dims] = browser_train_GT.get_whole_feats(whole_train_ids(1), feats, scaling_type, weights);
                        if(PowerNorm)
                            Feats_one_sample = squash_features(Feats_one_sample, 'power');
                        end
                        
                        total_dims = size(Feats_one_sample,1);
                        total_elems = numel(whole_train_ids(SVids))*total_dims;
                        n_chunks = ceil(total_elems/MAX_ELEMS_TRAINING);
                        whole_SV_ids_chunks = chunkify(whole_train_ids(SVids), n_chunks);
                        SV_alpha_chunks = chunkify(SV_alpha', n_chunks);
                        w = zeros(total_dims,1);
                        for k=1:numel(whole_SV_ids_chunks)
                            SVs = browser_train_GT.get_whole_feats(whole_SV_ids_chunks{k}, feats, scaling_type, weights);
                            if(PowerNorm)
                                SVs = squash_features(SVs, 'power');
                            end
                            
                            w = w + SVs * SV_alpha_chunks{k}';
                        end
                        
                        clear SVs;
                        b = - libsvm{j}.rho;
                        if  libsvm{j}.Label(1) == -1
                            w = -w;
                            b = -b;
                        end
                        libsvm{j}.w = w;
                        libsvm{j}.b = b;
                    else % liblinear
                        
                    end
                    t_class = toc(t_class)
                end
            else
                error('no such trainer');
            end
            time_training = toc(t_all)
            clear K;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%% validation on hold out data %%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [y_ho] = browser_ho_GT.get_overlaps_wholes(whole_ho_ids);
            if(strcmp(trainer, 'SVM'))
                t_testing = tic();
                total_dims = numel(libsvm{1}.w);
                total_elems = numel(whole_ho_ids)*total_dims;
                n_chunks = ceil(total_elems/MAX_ELEMS_TESTING);
                
                whole_ids_chunks = chunkify(whole_ho_ids, n_chunks);
                scores = zeros(numel(libsvm),numel(whole_ho_ids));
                for j=1:n_chunks
                    Feats_ho = browser_ho_GT.get_whole_feats(whole_ids_chunks{j}, feats, scaling_type, weights);
                    
                    if(PowerNorm)
                        Feats_ho = squash_features(Feats_ho, 'power');
                    end
                    
                    for k=1:numel(labels)
                        scores(k, whole_ids_chunks{j}) = libsvm{k}.w'*Feats_ho + libsvm{k}.b;
                    end
                end
                t_testing = toc(t_testing)
            end
            
            [val, labels_ho] = max(y_ho, [], 2);
            [~, pred_labels_ho] = max(scores);
            acc = sum(labels_ho' == pred_labels_ho) / numel(labels_ho)
            avg_ac = avg_acc(labels_ho, pred_labels_ho', 20)
            mean_avg_ac(iter1,iter2) = mean(avg_ac);
        end
    end
    mean_avg_ac
    PowerNorm
end



function K = compute_K(browser1, browser2, MAX_ELEMS, feats, scaling_type)
    train_mode = false;
    whole_ids_1 = 1:numel(browser1.whole_2_img_ids);
    if(isempty(browser2))
        train_mode = true;
        browser2 = browser1;
    end
    whole_ids_2 = 1:numel(browser2.whole_2_img_ids);

    K = zeros(numel(whole_ids_1), numel(whole_ids_2), 'single');

    [Feats1, dims] = browser1.get_whole_feats(whole_ids_1(1), feats, scaling_type, weights);

    total_dims = sum(dims);
    total_elems = numel([whole_ids_1 whole_ids_2])*total_dims;
    n_chunks = ceil(total_elems/MAX_ELEMS);

    whole_ids_1_chunks = chunkify(whole_ids_1, n_chunks);
    whole_ids_2_chunks = chunkify(whole_ids_2, n_chunks);

    for i=1:n_chunks
        [Feats1, dims] = browser1.get_whole_feats(whole_ids_1_chunks{i}, feats, scaling_type, weights);

        index2 = 1;
        if(train_mode)
            index2 = i;
        end

        for j=index2:n_chunks
            [Feats2, dims] = browser2.get_whole_feats(whole_ids_2_chunks{j}, feats, scaling_type, weights);

            K(whole_ids_1_chunks{i}, whole_ids_2_chunks{j}) = Feats1'*Feats2;

            if(train_mode && (i~=j))
                K(whole_ids_1_chunks{j},whole_ids_2_chunks{i}) = K(whole_ids_1_chunks{i}, whole_ids_2_chunks{j})';
            end
        end
    end
end