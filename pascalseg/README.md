%%%%%Diverse Semantic Segmentations

This directory contains the source code to generate multiple diverse semantic segmentations for the 20 Classes in PASCAL VOC. This code uses the O2P source code, the readme for which is located in README_O2P.


##Instructions
The Diverse Segmentation code is majorly located in the pascalseg/divmbest folder. 
The Demo can be run in matlab as follows:
cd divmbest/pascalseg/divmbest
segs=get_div_semantic_seg(<image_path>);

The code also automatically sets up an experiment directory which will contain the model files for O2P, necessary files for CPMC and other data files. This environment is set up by downloading the necessary files from https://filebox.ece.vt.edu/~senthil/divseg_o2p_env.tar.gz. The experiment directory is set up at pascalseg/divmbest/temp_divmbest.
If you wish to mannually set up the experiment directory with your models and PCA files, the structure of the experiment directory should be as shown in the file exp_dir_structure.


##Running on complete datasets
Alternatively, the pascalseg/VOC_experiment directory contains the o2p code which is used to train and test models on all PASCAL VOC images. The voc_test_models.m has been modified from the original o2p code to produce diverse semantic segmentations.

