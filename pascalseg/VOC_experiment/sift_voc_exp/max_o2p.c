/*---
function [mat_out] = max_o2p(mat_in)
Input:
    mat_in - nxm set of m local features of dimensionality n, single precision    

Output:
    mat_out = nxn symmetric matrix 

Joao Carreira, February 2012
--*/

# include "mex.h"
# include "math.h"

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
    
void mexFunction(
  int nargout,
  mxArray *out[],
  int nargin,
  const mxArray *in[]) {

  /* declare variables */
  int nr, nc, np;
  unsigned int i, j, l;
  float *feats;
  float *X;

  /* check argument */
  if (nargin<1) {
      mexErrMsgTxt("One argument required");
  }
  if (nargout>1) {
      mexErrMsgTxt("Too many output arguments");
  }

  nr = mxGetM(in[0]);
  nc = mxGetN(in[0]);
  
  /* get the ij-index pair */
  if (!mxIsSingle(in[0])) {
      mexErrMsgTxt("Input should have single precision.");
  }

  feats = mxGetData(in[0]);

  /* create output */  
  out[0] = mxCreateNumericMatrix(nr, nr,mxSINGLE_CLASS,mxREAL);
  if (out[0]==NULL) {
    mexErrMsgTxt("Not enough memory for the output matrix");
  }
  X = mxGetPr(out[0]);
  
  /* reset matrix */
  np = nr*nr;  
  for (i=0; i<np; i++) {
      X[i] = 0;
  }  
  
  for (i=0; i<nc; i++) { /* iterate over local features */
      for (j=0; j<nr; j++) { /* iterate over dimensions */
          for (l=0; l<nr; l++) { /* iterate over dimensions */              
            X[j*nr+l] = max(X[j*nr+l], feats[i*nr + j]*feats[i*nr + l]);            
      }
    }
  }  
}    
