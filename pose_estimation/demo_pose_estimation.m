%% This is a demo script that estimates multiple diverse human-body poses.

%% Create a structure called 'params' that stores all the arguments, which will be pass to the DivMBest function.

params.name = 'PARSE';
params.K = [6 6 6 6 6 6 6 6 6 6 6 6 6 6 ...
    6 6 6 6 6 6 6 6 6 6 6 6];
params.pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
params.sbin = 4;
params.type = 'divmbest';
params.nummodes = 5;
params.one_scale = 0;
[pos, neg, test] = PARSE_data(params.name);
params.test = test;

% Comment following line if you want to run DivMBest on the entire database.
imnum = 53;
params.test = test(imnum);

%% Uncomment the following line if the model is not cached and comment the subsequent two lines
%model = trainmodel(name,pos,neg,K,pa,sbin);
% load existing model for now
load('PARSE_model.mat');
params.model = model;
params.suffix = num2str(params.K')';

params.lambda = -0.5;

%% ================================
%% CallDivMBest

output = DivMBest_pose_estimation(params);

%% ================================
% Visualization

if(length(params.test) == 1)
im = imread(params.test.im);
colorset = {'g','g','y','r','r','r','r','y','y','y','m','m','m','m','y','b','b','b','b','y','y','y','c','c','c','c'};
boxes = output.boxes_mmodes;
boxes = boxes{:};

figure,
subplot(2,5,3), imshow(im);
title('Original Image');

for ii = 1:5
    subplot(2,5,5+ii),
    showskeletons(im,boxes(ii,:),colorset,model.pa);
    title(['Divsol #' num2str(ii)]);
end
end
