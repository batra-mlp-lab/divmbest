This directory contains the source code to generate multiple diverse solutions for the interactive binary segmentation experiment.
The demo requires TRW-S to be installed to run correctly. Please first download and install it from the following source ( http://www.robots.ox.ac.uk/%7Eojw/files/imrender_v2.4.zip ).

The repository contains data corresponding to an example image within the 'voctest50data' directory. The entire database (corresponding to PASCAL VOC2007 val set) can be downloaded from ( https://filebox.ece.vt.edu/~vittal/embr/voctest50data.tar ). Please replace the downloaded directory with the 'voctest50data' directory. You would need to write your own wrapper to loop through the entire dataset.

The demo can be seen by running the DivMBest_intseg.m script.
