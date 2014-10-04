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
cachedir = 'cache/';
if ~exist(cachedir,'dir')
  mkdir(cachedir);
end

if ~exist([cachedir 'imrotate/'],'dir')
  mkdir([cachedir 'imrotate/']);
end

if ~exist([cachedir 'imflip/'],'dir')
  mkdir([cachedir 'imflip/']);
end

buffydir = './BUFFY/';
parsedir = './PARSE/';
inriadir = './INRIA/';

addpath(buffydir);
