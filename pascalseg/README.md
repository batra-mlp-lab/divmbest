##Diverse Semantic Segmentations

This directory contains the source code to generate multiple diverse semantic segmentations for the 20 Classes in PASCAL VOC. This code uses the O2P source code, the readme for which is located in README_O2P.

The Diverse Segmentation code is majorly located in the pascalseg/divmbest folder. The Demo can be run by calling get_div_semantic_seg(<image_path>);. The code also requires an experiment directory to be set up which will contain the model files for O2P, necessary files for CPMC and other data files. This environment is automatically set up in the get_div_semantic_seg function by downloading the necessary files from https://filebox.ece.vt.edu/~senthil/divseg_o2p_env.tar.gz. The experiment directory is set up at pascalseg/divmbest/temp_divmbest. The structure of the experiment directory can be found in the exp_dir_structure. 

Alternatively, the pascalseg/VOC_experiment directory contains the o2p code which is used to train and test models on all PASCAL VOC images. The voc_test_models.m has been modified from the original o2p code to produce diverse semantic segmentations.

