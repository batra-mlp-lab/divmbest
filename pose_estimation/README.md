## Instructions

This directory contains the source code to generate DivMBest solutions for the human pose estimation experiments.

Run the following to compile the various mex files.

> cd third_party_code
<br/>
> compile;
<br/>
> cd ..

To run the demo, execute the following script in matlab:

> demo_pose_estimation;

To get divsols on all images in the PARSE dataset,  execute the following script in matlab:

> DivMBest_pose_estimation;


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
