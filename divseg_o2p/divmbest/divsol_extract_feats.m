function divsol_extract_feats(exp_dir, feat_collection, img_names, categories, overwrite)

  feat_types = sol_feat_config(feat_collection);
  for i =1:numel(feat_types)
    if(strcmp(feat_types{i}, 'pairwise_category_centroid_diff'))
      pars =[];
      pars.categories = categories;   
      pars.name = 'pairwise_category_centroid_diff';
      Segm_extract_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i}, 'pairwise_category_sum_of_pixels'))
      pars =[];
      pars.categories = categories;
      pars.name = 'pairwise_category_sum_of_pixels';
      pars.chi2_hom_kernel = false;
      Segm_extract_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i}, 'pairwise_category_sum_of_pixels_chi2_hom_kern'))
      pars =[];
      pars.categories = categories;
      pars.name = 'pairwise_category_sum_of_pixels_chi2_hom_kern';
      pars.chi2_hom_kernel=true;
      Segm_extract_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i}, 'pairwise_category_cooccurrence'))
      pars =[];
      pars.categories = categories;
      pars.chi2_hom_kernel=false;
      pars.name = 'pairwise_category_cooccurrence';
      Segm_extract_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i},'pairwise_category_cooccurrence_chi2_hom_kern'))
      pars =[];
      pars.categories = categories;
      pars.name = 'pairwise_category_cooccurrence_chi2_hom_kern';
      pars.chi2_hom_kernel=true;
      Segm_extract_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
     elseif(strcmp(feat_types{i},'uva_segment_classification_score'))
      pars =[];
      pars.categories = categories;
      pars.name = 'uva_segment_classification_score';
      pars.classification_resdir = '/share/data/vision-greg/Pascal/uijlings_classification';
      pars.classification_data_names = {'Pascal2007And2012Clfs', 'Pascal2012Val'}; 
      Segm_extract_measurements_on_sol('classification', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i},'globalpb_chamfer_distance_fore'))
      pars = [];
      pars.categories = categories;
      pars.foreground = true;
      pars.name = 'globalpb_chamfer_distance_fore';
      pars.globalPB_resdir ='/share/data/vision-greg/Pascal/VOCdevkit/VOC2012/gPb';%JPEGImages';
      Segm_extract_measurements_on_sol('globalpb', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i},'globalpb_chamfer_distance_segm'))
      pars = [];
      pars.categories = categories;
      pars.foreground = false;
      pars.name = 'globalpb_chamfer_distance_segm';
      pars.globalPB_resdir ='/share/data/vision-greg/Pascal/VOCdevkit/VOC2012/gPb';%JPEGImages';
      Segm_extract_measurements_on_sol('globalpb', pars, img_names, exp_dir,overwrite);
    else
      error('No such feature available.');
    end
  end
end


