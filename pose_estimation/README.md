This directory contains the source code to generate DivMBest solutions for the human pose estimation experiments.

Please follow the steps to run the demo.

i. Download the PARSE dataset from the following link ( https://filebox.ece.vt.edu/~vittal/embr/parse_dataset.tar ) and place the PARSE directory within the current directory (./divmbest/pose_estimation/)

ii. Run the following to compile the various mex files.

> cd third_party_code
<br/>
> compile;

iii. Run DivMBest_pose_estimation.m to generate the diverse solutions for the images in the ./PARSE directory.
