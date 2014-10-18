% Caltech 101 experiment from ECCV 2012 paper:
% "Semantic Segmentation with Second-Order Pooling", by
% Joao Carreira, Rui Caseiro, Jorge Batista, Cristian Sminchisescu
%
% Code by Joao Carreira, July 2012
% 
% This script will download the data automatically (should work in linux).


% get repeatable results;
s = RandStream('mt19937ar','Seed',1234);
RandStream.setGlobalStream(s);

% add paths
addpath('../src/');
addpath('../external_src/');
addpath('../src/SegmBrowser/');
addpath('../external_src/libsvm-3.11/matlab/');
addpath('../external_src/vlfeats/toolbox/sift/');
addpath('../external_src/vlfeats/toolbox/misc/');
addpath('../external_src/vlfeats/toolbox/mex/mexa64/');
addpath('../external_src/VOCcode/');

total_time = tic();

EXTENDED_SIFT = false; % set to true to use extended sift instead of plain sift.
exp_dir = './Caltech101/'; % path where data will be stored

if 0
    % if you want to parallelize change 0 to 1
    matlabpool open 6
end
    
% only need to do this once
if(~exist(exp_dir, 'dir'))
    generate_canonical_dataset();
end

imgset = 'all';
disp('Preparing the dataset browser...');
browser = SegmBrowser(exp_dir, 'ground_truth', 'all');
y_data = browser.get_overlaps_wholes(1:numel(browser.whole_2_img_ids));

% feature extraction
pars.mode = 'single_octave';
pars.color_type = 'gray'; % gray, opponent, hsv
pars.main_feat = {'really_dense_sift'}; %, };
pars.STEP = 4;
pars.base_scales = [2 4 6 8];

if 1 % Plain SIFT features (gets 79.2)
    pars.enrichments = {}; %{'rgb', 'hsv', 'lab', 'xy_fullimg', 'scale_fullimg'};
else % Enriched SIFT features (15 additional dimensions, gets 80.8)
    pars.enrichments = {'rgb', 'hsv', 'lab', 'xy_fullimg', 'scale_fullimg'};
end

% image resizing parameters
min_imsize = 45; 
max_imsize = 100;

% Spatial pyramid code adapted from a Liefeng Bo's caltech101 script, who adapted it
% from Lazebnik's code.
pyramid = [1 2 4];
pgrid = pyramid.^2; 

% get feature dimensionality
I = browser.get_Imgs(1);        
[D,F] = compute_shape_invariant_feats(I{1}, pars.main_feat, {pars.enrichments}, pars.mode, pars.color_type, [], pars.STEP, pars.base_scales);
single_desc_dim = (size(D,1)*(size(D,1)+1)/2);
feat_dim = sum(pgrid) * single_desc_dim;

Feats = zeros(feat_dim, numel(browser.img_names), 'single');
offset = 0.001*eye(size(D,1), size(D,1));
true_mat = true(size(D,1),size(D,1));
in_triu = triu(true_mat);

disp('Starting feature extraction.');
for i=1:numel(browser.img_names) 
    vgg_progressbar('Feature extraction', double(i/numel(browser.img_names)), 3);
    I = browser.get_Imgs(i);
    I = I{1};
    
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end;
    
    % resize image
    [im_h, im_w, ~] = size(I);
    
    if max(im_h, im_w) > max_imsize,
        I = imresize(I, max_imsize/max(im_h, im_w), 'bicubic');
        [im_h, im_w] = size(I);
    end;
    if min(im_h, im_w) < min_imsize,
        I = imresize(I, min_imsize/min(im_h, im_w), 'bicubic');
    end;
    
    % compute local sift descriptors
    masks = true(size(I,1), size(I,2));
    [D,F] = compute_shape_invariant_feats(I, pars.main_feat, {pars.enrichments}, pars.mode, pars.color_type, [], pars.STEP, pars.base_scales);
    
    y = F(1,:);
    x = F(2,:);
    y = y - min(y) + 1 ;
    x = x - min(x) + 1;
    
    % pool local sift descriptors over a spatial pyramid
    sgrid = sum(pgrid);
    weights = (1./pgrid); % divided by the number of grids at the corresponding level
    weights = weights/sum(weights);
    
    counter = 1;
    for s = 1:length(pyramid)
        width = (size(I,2)-1)/pyramid(s);
        height = (size(I,1)-1)/pyramid(s);
        xgrid = ceil(x/height);
        ygrid = ceil(y/width);
        allgrid = (ygrid -1 )*pyramid(s) + xgrid;
        for t = 1:pgrid(s)
            range = counter:counter+single_desc_dim-1;
            ind = find(allgrid == t);
            if(numel(ind)>1)
                theD = D(:,ind);
                
                % second-order pooling
                regionD = real(logm(((1/size(theD,2)).*(theD *theD')) + offset)); 
                regionD = regionD(in_triu);                
                
                Feats(range,i) = weights(s) * regionD(:);
            end
            counter = counter + single_desc_dim;
        end
    end
end
Feats = squash_features(Feats, 'power');

% train and test multiple times
N_RUNS = 5;
n_train_exemplars = 30;
lc = 10;
acc = [];
disp('Training and Testing.');
for i = 1:N_RUNS
    % generate training and test partitions
    indextrain = [];
    indextest = [];
    labelnum = size(y_data,2);
    [~, caltech101label] = find(y_data);
    for j = 1:labelnum
        index = find(caltech101label == j);
        perm = randperm(length(index));
        indextrain = [indextrain index(perm(1:n_train_exemplars))'];
        % training images per category is less than 50
        n_test_exemplars = min(n_train_exemplars+50,length(index));
        indextest = [indextest index(perm(n_train_exemplars+1:n_test_exemplars))'];
    end
    
    % generate training and test labels
    trainlabel = caltech101label(indextrain);
    testlabel = caltech101label(indextest);    
   
    trainsift = Feats(:,indextrain);
    testsift = Feats(:,indextest);
    
    % libsvm dual
    K = double(trainsift'*trainsift);
    K_test = double(trainsift'*testsift)';
    
    labels = unique(trainlabel);
    
    libsvm = cell(numel(labels),1);
    for j=1:numel(labels)
        these_labels = -ones(numel(trainlabel),1);
        these_labels(trainlabel == labels(j)) = 1;
        libsvm{j} = svmtrain(these_labels, ...
            [(1:size(K,1))' K], ...
            sprintf(' -t 4 -c %f -q -p 0.00001', lc));
    end
    
    scores = cell(numel(libsvm),1);
    for j=1:numel(labels)
        [predictlabel, duh, scores{j}] = ...
            svmpredict(zeros(size(K_test,1),1), [[size(K,1)+1:size(K,1)+size(K_test,1)]' K_test], ...
            libsvm{j});
        if(libsvm{j}.Label(1)==-1)
            scores{j} = -scores{j};
        end
    end
    
    scores2 = scores';
    scores2 = cell2mat(scores2);
    [value, predictlabel] = max(scores2, [], 2);
    
    acc(i) = mean(predictlabel == testlabel);

    % print and save classification accuracy
    fprintf(['Average accuracy over %d runs: %f\n'], i, mean(acc));
end

acc

t = toc(total_time);
fprintf('This experiment took: %f seconds\n', t);
