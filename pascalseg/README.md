##Diverse Semantic Segmentations

This directory contains the source code to generate multiple diverse semantic segmentations for the 20 Classes in PASCAL VOC. This code uses the O2P source code, the readme for which is located in README_O2P.

The Diverse Segmentation code is majorly located in the pascalseg/divmbest folder. The Demo can be run by calling get_div_semantic_seg(<image_path>);. The code also requires an experiment directory to be set up which will contain the model files for O2P, necessary files for CPMC and other data files. This environment is automatically set up in the get_div_semantic_seg function by downloading the necessary files from https://filebox.ece.vt.edu/~senthil/divseg_o2p_env.tar.gz. The experiment directory is set up at pascalseg/divmbest/temp_divmbest.

Alternatively, the pascalseg/VOC_experiment directory contains the o2p code which is used to train and test models on all PASCAL VOC images. The voc_test_models.m has been modified from the original o2p code to produce diverse semantic segmentations.

The structure of the experiment directory is as follows:
<exp_dir>
|-- Annotations
|-- Browser
|   `-- CPMC_segms_150_sp_approx
|-- Cache
|-- delete
|-- DIM_REDUC
|   |-- pca_basis_ground_truth_sp_approx_LBP_f_noncent.mat
|   |-- pca_basis_ground_truth_sp_approx_SIFT_GRAY_f_g_noncent.mat
|   `-- pca_basis_ground_truth_sp_approx_SIFT_GRAY_mask_noncent.mat
|-- ImageSets
|   `-- Segmentation
|-- JPEGImages
|-- MODELS
|   |-- aeroplane.mat
|   |-- bicycle.mat
|   |-- bird.mat
|   |-- boat.mat
|   |-- bottle.mat
|   |-- bus.mat
|   |-- car.mat
|   |-- cat.mat
|   |-- chair.mat
|   |-- cow.mat
|   |-- diningtable.mat
|   |-- dog.mat
|   |-- horse.mat
|   |-- motorbike.mat
|   |-- person.mat
|   |-- pottedplant.mat
|   |-- sheep.mat
|   |-- sofa.mat
|   |-- train.mat
|   `-- tvmonitor.mat
|-- MyCodebooks
|   |-- kmeans_dense_color_sift_3_scales_300_words.mat
|   `-- kmeans_dense_sift_4_scales_300_words.mat
|-- MyMeasurements
|   |-- CPMC_segms_150_sp_approx_back_mask_phog_nopb_20_orientations_3_levels
|   |-- CPMC_segms_150_sp_approx_bow_dense_color_sift_3_scales_figure_300
|   |-- CPMC_segms_150_sp_approx_bow_dense_sift_4_scales_figure_300
|   |-- CPMC_segms_150_sp_approx_LBP_f
|   |-- CPMC_segms_150_sp_approx_LBP_f_pca_2500_noncent
|   |-- CPMC_segms_150_sp_approx_mask_phog_scale_inv_20_orientations_2_levels
|   |-- CPMC_segms_150_sp_approx_SIFT_GRAY_f_g
|   |-- CPMC_segms_150_sp_approx_SIFT_GRAY_f_g_pca_5000_noncent
|   |-- CPMC_segms_150_sp_approx_SIFT_GRAY_mask
|   |-- CPMC_segms_150_sp_approx_SIFT_GRAY_mask_pca_5000_noncent
|   |-- dense_color_sift_3_scales
|   |-- dense_sift_4_scales
|   |-- dummy_masks_back_mask_phog_nopb_20_orientations_3_levels
|   |-- dummy_masks_bow_dense_color_sift_3_scales_figure_300
|   |-- dummy_masks_bow_dense_sift_4_scales_figure_300
|   `-- dummy_masks_mask_phog_scale_inv_20_orientations_2_levels
|-- MyOverlaps
|   `-- CPMC_segms_150_sp_approx
|-- MySegmentRankers
|   |-- attention_model_fewfeats_lambda_10.00_train.mat
|   `-- attention_model_fewfeats_lambda_10.00_trainval.mat
|-- MySegmentsMat
|   |-- CPMC_segms_150_sp_approx
|   `-- dummy_masks
|-- PB
|-- results
|-- SegmentationClass
|-- SegmentationObject
|-- SegmentEval
|   `-- CPMC_segms_150_sp_approx
|       `-- overlap
`-- WindowsOfInterest
    `-- grid_sampler

4
