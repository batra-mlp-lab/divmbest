function divsol_collect_feats(exp_dir, feat_collection, img_names, numsol_2collect,overwrite)
  feat_types = sol_feat_config(feat_collection);
  for i =1:numel(feat_types)
    if(strcmp(feat_types{i}, 'pairwise_category_centroid_diff'))
      pars =[];
      pars.name = 'pairwise_category_centroid_diff';
      pars.numsol_2collect = numsol_2collect;
      Segm_collect_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i}, 'pairwise_category_sum_of_pixels'))
      pars =[];
      pars.name = 'pairwise_category_sum_of_pixels';
      pars.numsol_2collect = numsol_2collect;
      pars.chi2_hom_kernel=false;
      Segm_collect_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i},'pairwise_category_sum_of_pixels_chi2_hom_kern'))
      pars =[];
      pars.name = 'pairwise_category_sum_of_pixels_chi2_hom_kern';
      pars.numsol_2collect = numsol_2collect;
      pars.chi2_hom_kernel=true;
      Segm_collect_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i}, 'pairwise_category_cooccurrence'))
      pars =[];
      pars.name = 'pairwise_category_cooccurrence';
      pars.numsol_2collect = numsol_2collect;
      pars.chi2_hom_kernel=false;
      Segm_collect_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i}, 'pairwise_category_cooccurrence_chi2_hom_kern'))
      pars =[];
      pars.name = 'pairwise_category_cooccurrence_chi2_hom_kern';
      pars.numsol_2collect = numsol_2collect;
      pars.chi2_hom_kernel=true;
      Segm_collect_measurements_on_sol('pairwise_category', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i},'uva_segment_classification_score'))
      pars =[];
      pars.name = 'uva_segment_classification_score';
      pars.numsol_2collect = numsol_2collect;
      Segm_collect_measurements_on_sol('classification', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i},'globalpb_chamfer_distance_fore'))
      pars =[];
      pars.name = 'globalpb_chamfer_distance_fore';
      pars.numsol_2collect = numsol_2collect;
      Segm_collect_measurements_on_sol('globalpb', pars, img_names, exp_dir,overwrite);
    elseif(strcmp(feat_types{i},'globalpb_chamfer_distance_segm'))
      pars =[];
      pars.name = 'globalpb_chamfer_distance_segm';
      pars.numsol_2collect = numsol_2collect;
      Segm_collect_measurements_on_sol('globalpb', pars, img_names, exp_dir,overwrite);
    else
      error('No such feature available.');
    end
  end
 
end

