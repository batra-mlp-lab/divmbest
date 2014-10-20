function output = DivMBest_pose_estimation(params)

if(exist('params', 'var'))
name = params.name;
K = params.K;
pa = params.pa;
sbin = params.sbin;

type = params.type;
nummodes = params.nummodes;

one_scale = params.one_scale;

test = params.test;

model = params.model;
suffix = params.suffix;

lambda = params.lambda;

else
	name = 'PARSE';
	K = [6 6 6 6 6 6 6 6 6 6 6 6 6 6 ...
    6 6 6 6 6 6 6 6 6 6 6 6];
	pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
	sbin = 4;
	type = 'divmbest';
	nummodes = 50;
	one_scale = 0;
	[pos, neg, test] = PARSE_data(name);
	
	%% Uncomment the following line if the model is not cached and comment the subsequent two lines
	%model = trainmodel(name,pos,neg,K,pa,sbin);
	% load existing model for now
	load('PARSE_model.mat');
	suffix = num2str(K')';
	lambda = -0.05;
end

% Download PARSE dataset if it does not exist
if(~exist('./PARSE/','dir'))
	try
		!wget https://filebox.ece.vt.edu/~vittal/embr/parse_dataset.tar
		!tar xfz parse_dataset.tar
	catch
		error('Unable to download/untar PARSE dataset. Please download/untar manually.');
	end
end

%% DivMBest
boxes_mmodes = testmodel_mmodes(name,model,test,suffix,nummodes,one_scale,lambda,type);
det_mmodes = PARSE_transback(boxes_mmodes);

output.boxes_mmodes = boxes_mmodes;
output.det_mmodes = det_mmodes;

end
