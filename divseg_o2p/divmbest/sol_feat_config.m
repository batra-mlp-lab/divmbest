function [feats] = sol_feat_config(feat_collection)

  feats = {
    'pairwise_category_centroid_diff',...
    'pairwise_category_sum_of_pixels',...
    'pairwise_category_cooccurrence',...
    'uva_segment_classification_score',...
    'globalpb_chamfer_distance_fore',...
    'globalpb_chamfer_distance_segm',...      
    'pairwise_category_sum_of_pixels_chi2_hom_kern',...
    'pairwise_category_cooccurrence_chi2_hom_kern'
  };

  if(strcmp(feat_collection, 'all_feats'))
    feats = feats([1 2 3 4 6]);
  elseif(strcmp(feat_collection, 'pairwise_category'))
    feats = feats([1 2 3]);
  elseif(strcmp(feat_collection, 'classification'))
    feats = feats([4]);
  elseif(strcmp(feat_collection, 'globalpb'))
    feats = feats([6]);
  elseif(strcmp(feat_collection, 'pairwise_category_chi2_hom_kern'))
    feats = feats([7 8]);
  elseif(strcmp(feat_collection, 'o2p_best'))
    feats = feats([1 2 3 6]);
  else
    error('no such type');
  end
