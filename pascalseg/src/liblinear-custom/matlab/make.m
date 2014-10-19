% This make.m is used under Windows

mex -O -D_DENSE_REP -largeArrayDims -c ../blas/saxpy.c ../blas/sdot.c ../blas/snrm2.c ../blas/sscal.c -outdir ../blas 
mex -O -D_DENSE_REP -largeArrayDims -c ../linear.cpp 
mex -O -D_DENSE_REP -largeArrayDims -c ../tron.cpp 
mex -O -D_DENSE_REP -largeArrayDims -c linear_model_matlab.c -I../ 
mex -O  -D_DENSE_REP -largeArrayDims train.c -I../ tron.o linear.o linear_model_matlab.o ../blas/saxpy.o ../blas/sdot.o ../blas/snrm2.o ../blas/sscal.o 
%

% mex -g -D_DENSE_REP -largeArrayDims -c ../blas/saxpy.c ../blas/sdot.c ../blas/snrm2.c ../blas/sscal.c -outdir ../blas
% mex -g -D_DENSE_REP -largeArrayDims -c ../linear.cpp
% mex -g -D_DENSE_REP -largeArrayDims -c ../tron.cpp
% mex -g -D_DENSE_REP -largeArrayDims -c linear_model_matlab.c -I../
% mex -g  -D_DENSE_REP -largeArrayDims train.c -I../ tron.o linear.o linear_model_matlab.o ../blas/saxpy.o ../blas/sdot.o ../blas/snrm2.o ../blas/sscal.o

!cp train.mexa64 svmlin_train_weights.mexa64
%!cp train.mexglx svmlin_train.mexglx
