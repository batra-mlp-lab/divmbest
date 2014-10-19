## Instructions

This directory contains the source code to generate DivMBest solutions for the human pose estimation experiments.

Please follow the steps to run the demo.

i. Download the PARSE dataset from the following link ( https://filebox.ece.vt.edu/~vittal/embr/parse_dataset.tar ) and place the PARSE directory within the current directory (./divmbest/pose_estimation/)

ii. Run the following to compile the various mex files.

> cd third_party_code
<br/>
> compile;
<br/>
> cd ..

iii. Run DivMBest_pose_estimation.m to generate the diverse solutions for the images in the ./PARSE directory.

## Acknowledgements

We thank Yang and Ramanan for releasing the code accompanying the following publication.


        @article{yang2013articulated,
          title={Articulated human detection with flexible mixtures of parts},
          author={Yang, Yi and Ramanan, Deva},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          volume={35},
          number={12},
          pages={2878-2890},
          year={2013},
          publisher={IEEE}
        }
