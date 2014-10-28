% Set up global paths used throughout the code
addpath ./third_party_code/learning;
addpath ./third_party_code/detection;
addpath detection;

if isunix()
  addpath ./third_party_code/mex_unix;
elseif ispc()
  addpath ./third_party_code/mex_pc;
end

% directory for caching models, intermediate data, and results
if(exist('./DivMBest_pose_estimation_PATH.mat', 'file'))
    load('./DivMBest_pose_estimation_PATH.mat');
	cachedir = [DivMBest_pose_estimation_PATH 'cache/'];
	parsedir = [DivMBest_pose_estimation_PATH 'PARSE/'];
	inriadir = [DivMBest_pose_estimation_PATH 'INRIA/'];
else
    cachedir = 'cache/';
    parsedir = 'PARSE/';
    inriadir = './INRIA/';
end
if ~exist(cachedir,'dir')
  mkdir(cachedir);
end

if ~exist([cachedir 'imrotate/'],'dir')
  mkdir([cachedir 'imrotate/']);
end

if ~exist([cachedir 'imflip/'],'dir')
  mkdir([cachedir 'imflip/']);
end

% buffydir = './BUFFY/';

% addpath(buffydir);
