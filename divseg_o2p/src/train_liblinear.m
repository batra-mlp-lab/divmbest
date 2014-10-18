function model = train_liblinear(Feats, y_train, lc, alphas, w, inst_w, svm_type, svr_par, svr_prec)
    DefaultVal('*svr_prec', '0.001');
    DefaultVal('*svm_type', '''svr''');
    DefaultVal('*svr_par', '0.2');    
    DefaultVal('*inst_w', '[]');        
    
    if(nargin==3)
        w = zeros(size(Feats,1),1,'single');
        alphas = zeros(size(Feats,2),1,'single');
    end
    
    if(isempty(inst_w))
        inst_w = ones(size(Feats,2),1, 'single');
    end
    
    alphas=single(alphas);
    these_labels = int8(y_train>0);
    these_labels(these_labels==0) = -1;
    
    % -s is the solver (1 is dual, 2 is primal)
    if strcmp(svm_type, 'svm')        
        model = svmlin_train_weights(single(these_labels), Feats, sprintf('-s 1 -c %f ', lc), 'col', alphas, w, inst_w);
        model.SVids = model.alphas>0;
    elseif strcmp(svm_type, 'svr')
        % train using regression
        t = tic();                                          
        model = svmlin_train_weights(single(y_train), Feats, sprintf('-s 12 -e %f -c %f -p %f', svr_prec, lc, svr_par), 'col', alphas, w, inst_w);
        toc(t)
        model.SVids = model.alphas~=0;
    elseif(strcmp(svm_type, 'sgd'))
        %lambda = 0.00001; % worse
        lambda = 0.000001; 
        %lambda = 0.0000001; % worse
        
        N_ITER = 5000000; 
        %N_ITER = 10000000; 
        %N_ITER = 20000000; % worse
        if 0
            the_perm = [];
            if(any(inst_w==0))
                new_perm = find(inst_w ~= 0);
            else
                new_perm = 1:numel(inst_w);
            end
            n_passes = 1;
            new_perm = repmat(new_perm, n_passes, 1);
            r = randperm(numel(new_perm));
            the_perm = new_perm(r);
            the_perm = uint32(the_perm);

            t = tic();
            model.w = vl_pegasos(Feats, these_labels,lambda, 'BiasMultiplier', 0, 'Permutation', the_perm);
            toc(t)
        else
            Feats(:,inst_w==0) = [];
            these_labels(inst_w==0) = [];
            t = tic();
            model.w = vl_pegasos(Feats, these_labels,lambda, 'BiasMultiplier', 0, 'NumIterations', N_ITER);
            toc(t)
        end
        
        model.w = model.w';
        model.alphas = [];
        model.SVids = [];
        model.Label = [];
        %plot(model.w'*Feats)
    end
        
    if(~isempty(model.Label))
        if(model.Label(1)~=1)
            model.w = -model.w;            
        end                
    end
    
    model.SVids = find(model.SVids);    
    model.alphas = model.alphas(model.SVids);    
end
