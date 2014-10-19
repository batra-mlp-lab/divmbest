function [F D]=Segm_collect_measurements_on_sol(type, pars, img_names, exp_dir,overwrite)
  DefaultVal('*overwrite', 'false');
  
  if(strcmp(type, 'pairwise_category'))
    type_func = @collect_database_on_pairwise_category;
    dir_name = [type '/' pars.name '/'];
  elseif(strcmp(type, 'classification'))
    type_func = @collect_database_on_classification;
    dir_name = [type '/' pars.name '/'];
  elseif(strcmp(type, 'globalpb'))
    type_func = @collect_database_on_globalpb;
    dir_name = [type '/' pars.name '/'];
  else
    error('no such type defined');
  end

  the_dir = [exp_dir 'MyMeasurements/Collated/' type];
  if ~exist(the_dir,'dir')
    mkdir(the_dir);
  end

  [F,D] = type_func(pars.name, pars, img_names, exp_dir, dir_name);

  if overwrite || ~exist([the_dir '/' pars.name '.mat'])
    %save([the_dir '/' pars.name '.mat'],'D','F');
    feat = permute(D,[2 3 1]);
    save([the_dir '/' pars.name '.mat'],'feat');
  end
end

function [F,D]=collect_database_on_pairwise_category(type, the_pars, img_names, exp_dir, dir_name)
  if strcmp(type,'pairwise_category_centroid_diff')
    %dim = 840;
    dim = 420;
    normalize = false;
  elseif strcmp(type, 'pairwise_category_sum_of_pixels')
    dim = 210; 
    normalize = true;
  elseif strcmp(type, 'pairwise_category_sum_of_pixels_chi2_hom_kern')
    dim = 1050; 
    normalize = false;
  elseif strcmp(type, 'pairwise_category_cooccurrence')
    dim = 210;
    normalize = true;
  elseif strcmp(type, 'pairwise_category_cooccurrence_chi2_hom_kern')
    dim = 1050;
    normalize = false;
  else
    error('No such classification feature available');
  end
 
  D = zeros(dim,the_pars.numsol_2collect,length(img_names));

  parfor i=1:length(img_names),
    the_dir = [exp_dir 'MyMeasurements/' dir_name img_names{i} '/'];
    sol_feat_files = dir([the_dir '/*_S*.mat']);
    sol_feat_files = {sol_feat_files(:).name};

    if length(sol_feat_files)<the_pars.numsol_2collect,
      error('Not enough pairwise_category solution measurements'); 
    end

    imgsols_D = zeros(dim,the_pars.numsol_2collect); 
    for k=1:the_pars.numsol_2collect,
      feat=load([the_dir '/' img_names{i} sprintf('_S%04d.mat',k)],'D');
      if normalize
        C=sum(feat.D);
        if C~=0,
          feat.D = feat.D /C;
        end
      end
      imgsols_D(:,k) = feat.D;   
    end
    D(:,:,i) = imgsols_D;
  end
  F=[];
end

function [F,D]=collect_database_on_classification(type, the_pars, img_names, exp_dir, dir_name) 
  if strcmp(type,'uva_segment_classification_score')
    dim = 20;
  else
    error('No such classification feature available');
  end

  D = zeros(dim,the_pars.numsol_2collect,length(img_names));

  parfor i=1:length(img_names),
    the_dir = [exp_dir 'MyMeasurements/' dir_name img_names{i} '/'];
    sol_feat_files = dir([the_dir '/*_S*.mat']);
    sol_feat_files = {sol_feat_files(:).name};

    if length(sol_feat_files)<the_pars.numsol_2collect,
      error('Not enough pairwise_category solution measurements'); 
    end

    imgsols_D = zeros(dim,the_pars.numsol_2collect);
    for k=1:the_pars.numsol_2collect,
      feat=load([the_dir '/' img_names{i} sprintf('_S%04d.mat',k)],'D');
      imgsols_D(:,k) = feat.D;
    end
    D(:,:,i) = imgsols_D;
  end
  F=[];
end

function [F,D]=collect_database_on_globalpb(type, the_pars, img_names, exp_dir, dir_name)
  if strcmp(type(1:end-5),'globalpb_chamfer_distance'),
    %dim = 840;
    dim = 120;
    normalize = true;
  else
    error('No such globalpb feature available');
  end
  D = zeros(dim,the_pars.numsol_2collect,length(img_names));

  parfor i=1:length(img_names),
    the_dir = [exp_dir 'MyMeasurements/' dir_name img_names{i} '/'];
    sol_feat_files = dir([the_dir '/*_S*.mat']);
    sol_feat_files = {sol_feat_files(:).name};

    if length(sol_feat_files)<the_pars.numsol_2collect,
      error('Not enough globalpb solution measurements'); 
    end

    imgsols_D = zeros(dim,the_pars.numsol_2collect);
    for k=1:the_pars.numsol_2collect,
      feat=load([the_dir '/' img_names{i} sprintf('_S%04d.mat',k)],'D');
      
      if normalize,
        sumfeat = sum(feat.D); 
        if sumfeat~=0,
          feat.D = feat.D /sumfeat;
        end
      end
      imgsols_D(:,k) = feat.D;
    end
    D(:,:,i) = imgsols_D;
  end
  F=[];
end

