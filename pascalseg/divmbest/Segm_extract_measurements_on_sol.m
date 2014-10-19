function Segm_extract_measurements_on_sol(type, pars, img_names, exp_dir, overwrite)
  DefaultVal('*overwrite', 'false');

  if(strcmp(type, 'pairwise_category'))
    type_func = @run_exp_database_do_pairwise_category;
    dir_name = [type '/' pars.name '/'];
  elseif(strcmp(type, 'classification'))
    type_func = @run_exp_database_do_classification;
    dir_name = [type '/' pars.name '/'];
    if strcmp(pars.name, 'uva_segment_classification_score')
      pars = uva_classification_data(pars);
    end
  elseif(strcmp(type,'globalpb'))
    type_func =@run_exp_database_do_globalpb;
    dir_name = [type '/' pars.name '/'];
  else
    error('no such type defined');
  end


  parfor i=1:numel(img_names)
  %for i = 840% 1:numel(img_names)
    %fprintf(1,'%d: %s\n',i,img_names{i});

    fprintf(1,'.');
    img_name = img_names{i};
    img_sol_names = dir([exp_dir '/' img_name '/*_S*.png']);
    img_sol_names = {img_sol_names(:).name};
    img_sol_names = cellfun(@(x) x(1:regexp(x,'^*[.]')-1),img_sol_names,'UniformOutput',false);

    the_pars = add_j_to_pars(pars, i, img_name);

    for k=1:length(img_sol_names),
      filename_to_save = [exp_dir 'MyMeasurements/' dir_name img_name '/'...
                        img_sol_names{k} '.mat'];
      if ~overwrite
        if(exist(filename_to_save, 'file'))
          continue;
        end
      end

      sol_file = [exp_dir '/' img_name '/' img_sol_names{k} '.png'];
      sol = imread(sol_file);

      my_all(exp_dir, pars.name, sol, the_pars, dir_name, img_name, img_sol_names{k}, type_func);
    end
  end
  
end

function [F, D] = my_all(exp_dir, type, sol, the_pars, dir_name, img_name, img_sol_name, type_func)
  [F,D] = type_func(type, sol, the_pars); 
  the_dir = [exp_dir 'MyMeasurements/' dir_name img_name '/'];

  if ~(exist(the_dir, 'dir'))
    mkdir(the_dir);
  end
  save([the_dir img_sol_name '.mat'],'D','-V6');
end


function pars = add_j_to_pars(pars,j, img_name)
    pars.j = j;
    pars.img_name = img_name;	
end

function [F,D] = run_exp_database_do_pairwise_category(type, sol, pars)

  categories = pars.categories;
  numcategories = length(categories);
  soltemp = zeros(size(sol));  
  sol_labels_orig = unique(sol);
  sol_labels = sol_labels_orig;
  sol_labels(sol_labels_orig==0)=numcategories+1;
  label_pairs = combnk(sol_labels,2);


 
  if strcmp(type,'pairwise_category_centroid_diff')
    sol_sz=size(sol);
    height = sol_sz(1);
    width = sol_sz(2);
    M = zeros(numcategories+1, numcategories+1,2);
    for i = 1:length(sol_labels_orig),
      soltemp(sol==sol_labels_orig(i)) = i;
    end

    label_pairs = [label_pairs; flipdim(label_pairs,2)];
    centers = regionprops(soltemp,'centroid');
    for i = 1:size(label_pairs,1),
      row = find(sol_labels==label_pairs(i,1));
      col = find(sol_labels==label_pairs(i,2));
      M(label_pairs(i,1), label_pairs(i,2),:) =  (centers(row).Centroid - centers(col).Centroid)./[width height]; 
    end
    F = [];
    D = M([triu(true(size(M(:,:,1))),1) triu(true(size(M(:,:,1))),1)]);

  elseif strcmp(type, 'pairwise_category_sum_of_pixels')|strcmp(type, 'pairwise_category_sum_of_pixels_chi2_hom_kern')
    M = zeros(numcategories+1, numcategories+1);
    soltemp = sol;
    soltemp(soltemp==0) = numcategories+1;
    label_pairs = sort(label_pairs,2);
    for i = 1:size(label_pairs,1)  
      M(label_pairs(i,1),label_pairs(i,2)) = nnz(soltemp==label_pairs(i,1)) +...
                                              nnz(soltemp==label_pairs(i,2));
    end
    F = [];
    D = M(triu(true(size(M)),1)) / numel(sol);
    
    if pars.chi2_hom_kernel, %homogenous chi2 kernel map
      D = vl_homkermap(D,2);
    end

  elseif strcmp(type, 'pairwise_category_cooccurrence')|strcmp(type, 'pairwise_category_cooccurrence_chi2_hom_kern')
    M = zeros(numcategories+1, numcategories+1);
    label_pairs = sort(label_pairs,2);
    for i=1:size(label_pairs,1)
      M(label_pairs(i,1),label_pairs(i,2)) = 1;
    end
    F = [];
    D = M(triu(true(size(M)),1)) / (((numcategories+1)^2 - numcategories+1)/2);

    if pars.chi2_hom_kernel, %homogenous chi2 kernel map
      D = vl_homkermap(D,2);
    end
  else
    error('No such pairwise category feature available');
  end
end

function [F,D] = run_exp_database_do_classification(type, sol, pars)
  img_name   = pars.img_name;
  categories = pars.categories;
  numcategories = length(categories);
  sol_labels = unique(sol);
  sol_labels(sol_labels==0)=[];

  if strcmp(type,'uva_segment_classification_score')
   loc = find(strcmp(pars.testIms_all,img_name)); 
    if ~isempty(loc),
      D = zeros(numcategories,1);
      D(sol_labels) = pars.clfs_all(loc,sol_labels);
    else
      D = zeros(numcategories,1);
    end
    F = [];
  else
    error('No such classification feature available');
  end
end

function [F,D] = run_exp_database_do_globalpb(type, sol, pars)

  ndistbins = 6;
  ngpbvalbins = 6;
  nthreshs = 10;
  %dim = (nthreshs*ndistbins + nthreshs*ndistbins*ngpbvalbins)*2;
  dim = nthreshs*ndistbins*2;

  F=[];
  D=zeros(dim,1);

  if nnz(sol)==0,  
    return;
  end

  img_name = pars.img_name;
  disp([pars.globalPB_resdir '/' img_name '.thin.pb']);
  gpb_thin=loadPb([pars.globalPB_resdir '/' img_name '.thin.pb']);
  max_pb = max(gpb_thin(:));
  rbins=[-inf floor(logspace(1,log10(norm(size(gpb_thin))/2),ndistbins-1)) inf];

  threshs=logspace(0,1.5,nthreshs)/(10^1.5)*max_pb;
  threshs=[0 threshs(1:end-1)];

  img_sz = size(gpb_thin);

  if pars.foreground
    sol_mask = sol>0;
    if nnz(sol_mask)==numel(sol), %the whole image is covered with segments so we can't handle this and return zero vector
      fprintf('\nSolution covers entire image: %s\n',img_name); %example: image 687, 2009_000771 in VOC2012 
      return;
    end
    sf=SegmFeatures([],sol_mask);
    bndry = sf.compute_boundaries(sol_mask,1); 
    bndry=unique(cell2mat(bndry{:}),'rows','stable');
    sol_bndry_mask = zeros(size(sol_mask));
    sol_bndry_mask(sub2ind(size(sol_bndry_mask),bndry(:,1),bndry(:,2)))=1;
  else
    lbls=unique(sol);
    lbls(lbls==0)=[];
    sol_bndry_mask = zeros(size(sol));
    for l=1:numel(lbls),
      sol_mask = (sol==lbls(l));
      if nnz(sol_mask)==numel(sol), %the whole image is covered with segments so we can't handle this and return zero vector
        fprintf('\nSolution covers entire image: %s\n',img_name); %example: image 687, 2009_000771 in VOC2012 
        return;
      end
      sf=SegmFeatures([],sol_mask);
      bndry = sf.compute_boundaries(sol_mask,1); 
      bndry=unique(cell2mat(bndry{:}),'rows','stable');
      sol_bndry_mask(sub2ind(size(sol_bndry_mask),bndry(:,1),bndry(:,2)))=1;
    end  
  end

  for i =1:nthreshs
    if threshs(i)>0,
      gpbthreshbins=logspace(log10(threshs(i)),log10(max_pb),ngpbvalbins); %maximum should be max_pb, example error in image if 1 set: image 613 VOC2012 
      gpbthreshbins=gpbthreshbins(1:end-1);
    else
      gpbthreshbins=[0 logspace(0,1.5,ngpbvalbins)/(10^1.5)]*max_pb;
      gpbthreshbins=gpbthreshbins(1:end-1);
    end
    gpb_thresh=double(gpb_thin>=threshs(i));


    gpbthreshtemp = gpb_thresh;
    gpbthreshtemp(gpbthreshtemp==0)=inf;
    gpbthreshtemp(gpbthreshtemp==1)=0;
    [dtform, nbrs]=vl_imdisttf(single(gpbthreshtemp));
    dtform = sqrt(dtform);
    dtform_binnum=vl_binsearch(rbins,double(dtform(:)));
    r_map=reshape(dtform_binnum,img_sz(1),img_sz(2));
   
    sol_bndry_binned_dist_mask=sol_bndry_mask.*r_map;
    sol_bndry_binned_dist=sol_bndry_binned_dist_mask(sol_bndry_binned_dist_mask>0);
    
    closest_gpb_nbrs  = nbrs.*sol_bndry_mask;
    gPB_val_sol_bndry =gpb_thin(closest_gpb_nbrs(closest_gpb_nbrs>0));
  
    try
    %dist_hist_sol_bndry_to_gPB{i}=vl_binsum(zeros(ndistbins,1),1,sol_bndry_binned_dist) ./ nnz(sol_bndry_mask);
    gPB_hist_per_dist_from_sol_bndry_to_gPB{i}=vl_binsum(zeros(ndistbins,1),gPB_val_sol_bndry,sol_bndry_binned_dist) ./ nnz(sol_bndry_mask);
    catch
      fprintf(1,'%s\n',img_name);
      error('bad image');
    end

%    gpbvalhistsolbndrytogPB={};
%    for k=1:ndistbins,
%      closest_gpb_nbrs=nbrs.*(sol_bndry_binned_dist_mask==k);
%      if nnz(closest_gpb_nbrs) >0,
%        nbr_gpbvals=gpb_thin(closest_gpb_nbrs(closest_gpb_nbrs>0));
%        try
%        gpbvalhistsolbndrytogPB{k}=vl_binsum(zeros(ngpbvalbins,1),1,vl_binsearch(gpbthreshbins,nbr_gpbvals))./nnz(closest_gpb_nbrs>0);
%        catch
%          fprintf(1,'%s\n',img_name);
%          error('bad image');
%        end
%      else
%        gpbvalhistsolbndrytogPB{k}=zeros(ngpbvalbins,1);
%      end
%    end
%    gpbval_hist_sol_bndry_to_gPB{i}=cat(1,gpbvalhistsolbndrytogPB{:});
    %cat(1,gpbvalhistsolbndrytogPB{:})

    solbndrymasktemp = sol_bndry_mask;
    solbndrymasktemp(solbndrymasktemp==0)=inf;
    solbndrymasktemp(solbndrymasktemp==1)=0;
    [dtform, nbrs]=vl_imdisttf(single(solbndrymasktemp));
    dtform = sqrt(dtform);
    dtform_binnum=vl_binsearch(rbins,double(dtform(:)));
    r_map=reshape(dtform_binnum,img_sz(1),img_sz(2));
   
    gpb_binned_dist_mask = gpb_thresh .* r_map;
    gpb_binned_dist = gpb_binned_dist_mask(gpb_binned_dist_mask>0);

    gPB_val_gPB_bndry = gpb_thin(find(gpb_thresh>0));
    try
      %dist_hist_gPB_to_sol_bndry{i} = vl_binsum(zeros(ndistbins,1),1,gpb_binned_dist) ./nnz(gpb_thresh); 
      gPB_hist_per_dist_from_gPB_to_sol_bndry{i} = vl_binsum(zeros(ndistbins,1),gPB_val_gPB_bndry,gpb_binned_dist) ./nnz(gpb_thresh);
    catch
      fprintf(1,'%s\n',img_name);
      error('bad image');
    end
%    gpbvalhistgPBtosolbndry={};
%    for k=1:ndistbins,
%      closest_gpb_nbrs = (gpb_binned_dist_mask==k);
%      if nnz(closest_gpb_nbrs)>0
%        nbr_gpbvals=gpb_thin(find(closest_gpb_nbrs>0));
%        gpbvalhistgPBtosolbndry{k}=vl_binsum(zeros(ngpbvalbins,1),1,vl_binsearch(gpbthreshbins,nbr_gpbvals))./nnz(closest_gpb_nbrs>0);
%      else
%        gpbvalhistgPBtosolbndry{k}=zeros(ngpbvalbins,1);
%      end
%    end
%    gpbval_hist_gPB_to_sol_bndry{i} =cat(1,gpbvalhistgPBtosolbndry{:});
    %cat(1,gpbvalhistgPBtosolbndry{:})

  end %for i
  D=[cat(1,gPB_hist_per_dist_from_sol_bndry_to_gPB{:});... 
     cat(1,gPB_hist_per_dist_from_gPB_to_sol_bndry{:})];
%  D=[cat(1,dist_hist_sol_bndry_to_gPB{:});...
%     cat(1,dist_hist_gPB_to_sol_bndry{:})];...
%     cat(1,gpbval_hist_sol_bndry_to_gPB{:});...
%     cat(1,gpbval_hist_gPB_to_sol_bndry{:})];
  F=[];
end

function pars=uva_classification_data(pars)
    for i =1:length(pars.classification_data_names)
      classif{i} = load([pars.classification_resdir '/' pars.classification_data_names{i}]);
    end
  
    testIms_all = [];
    clfs_all = [];
    for i =1:length(pars.classification_data_names)
      if strcmp(pars.classification_data_names{i},'Pascal2007And2012Clfs')
        testIms_all = [testIms_all; cellfun(@(x) ['2007_' x],classif{i}.testIms2007,'UniformOutput',false);...
                   classif{i}.testIms2012(:)];                 
        clfs_all=[clfs_all; classif{i}.clfs2007; classif{i}.clfs2012];
      elseif strcmp(pars.classification_data_names{i}, 'Pascal2012Val')
        testIms_all = [testIms_all; classif{i}.testIms2012val(:)];
        clfs_all=[clfs_all; classif{i}.clfs2012val];
      else
        error('UVA classification data name invalid');
      end
    end
    [pars.testIms_all order]=sortrows(testIms_all);
    pars.clfs_all=clfs_all(order,:);

end
