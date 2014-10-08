This directory contains the source code to generate DivMBest solutions for the human pose estimation experiments.

Please follow the steps to run the demo.

1. Download the PARSE dataset from the following link ( https://filebox.ece.vt.edu/~vittal/embr/parse_dataset.tar ) and place the PARSE directory within the current directory (./divmbest/pose_estimation/)

2. Run the following to compile the various mex files.

> cd third_party_code
> compile;

3. Run DivMBest_pose_estimation.m to generate the diverse solutions for the images in the ./PARSE directory.
