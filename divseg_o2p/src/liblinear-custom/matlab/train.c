#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "linear.h"

#include "mex.h"
#include "linear_model_matlab.h"


#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#define  _DENSE_REP 1

void print_null(const char *s){}

void (*liblinear_default_print_string) (const char *);

void exit_with_help()
{
	mexPrintf(
	"Usage: model = train(y, Feats, 'liblinear_options', 'col', alphas, w, inst_w);\n"
    "Alphas and w are useful to warm start the algorithm. Inst_w allows you to give a preferential treatment to some points.\n" 
    "Among -s I only implemented/tested option -s 1 and 12 \n"            
#ifdef _DENSE_REP
	" ( warning : training_instance_matrix must be dense )\n"
#endif
	"liblinear_options:\n"
	"-s type : set type of solver (default 1, only trust [1 2 11 12])\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- multi-class support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"	11 -- L2-regularized L2-loss epsilon support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss epsilon support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss epsilon support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
    "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" 
	"	-s 1, 3, 4 and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'([zeros])|,\n"
	"		where f is the dual function (default 0.001)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"col:\n"
	"	if 'col' is setted, training_instance_matrix is parsed in column format, otherwise is in row format\n"
	);
}

/* liblinear arguments */
struct parameter param;		/* set by parse_command_line */
struct problem prob;		/* set by read_problem */
struct model *model_;

#ifdef _DENSE_REP
float *x_space;
#else
struct feature_node *x_space;
#endif

int cross_validation_flag;
int col_format_flag;
int nr_fold;
float bias;

float do_cross_validation()
{
	int i;
	int total_correct = 0;
	int *target = Malloc(int,prob.l);
	float retval = 0.0;

	cross_validation(&prob,&param,nr_fold,target);

	for(i=0;i<prob.l;i++)
		if(target[i] == prob.y[i])
			++total_correct;
	mexPrintf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	retval = 100.0*total_correct/prob.l;

	free(target);
	return retval;
}

int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
	int i, argc = 1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];

	/* default values */
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; /* see setting below */
	param.nr_weight = 0;
	param.weight_label = NULL;
	cross_validation_flag = 0;
	col_format_flag = 0;
	bias = -1;

	/* train loaded only once under matlab */
	if(liblinear_default_print_string == NULL)
		liblinear_default_print_string = liblinear_print_string;
	else
		liblinear_print_string = liblinear_default_print_string;

	if(nrhs <= 1)
		return 1;

	if(nrhs > 3)
	{
		mxGetString(prhs[3], cmd, mxGetN(prhs[3])+1);
		if(strcmp(cmd, "col") == 0)
			col_format_flag = 1;
	}
    
	/* put options in argv[] */
	if(nrhs > 2)
	{
		mxGetString(prhs[2], cmd,  mxGetN(prhs[2]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q')
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;                
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'B':
				bias = atof(argv[i]);
				break;
			case 'v':
				cross_validation_flag = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					mexPrintf("n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (float*) realloc(param.weight,sizeof(float)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			case 'q':
				liblinear_print_string = &print_null;
				i--;
				break;
			default:
				mexPrintf("unknown option\n");
				return 1;
		}
	}

	if(param.eps == INF) {
		switch(param.solver_type)
		{
			case L2R_LR: 
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL: 
			case L2R_L1LOSS_SVC_DUAL: 
			case MCSVM_CS: 
			case L2R_LR_DUAL: 
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC: 
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.001;
				break;
		}
	}
	return 0;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

int read_problem_sparse(const mxArray *label_vec, const mxArray *instance_mat, const mxArray *alphas_vec, const mxArray *w_vec, const mxArray *w_inst_vec)
{
	int i, j;
#ifdef _DENSE_REP
#else
	int  k, low, high;
	mwIndex *ir, *jc;
#endif
	int elements, max_index, label_vector_row_num, alphas_in_vector_row_num, nfeats, w_in_vector_row_num, w_inst_row_num;
	float *samples, *labels, *alphas, *w, *w_inst;
	mxArray *instance_mat_col; 

	prob.x = NULL;
	prob.y = NULL;
    prob.alphas_in = NULL;
    prob.w_in = NULL;
    prob.W = NULL;
	x_space = NULL;

	if(col_format_flag)
		instance_mat_col = (mxArray *)instance_mat;
	else
	{
	
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_col = plhs[0];
		mxDestroyArray(prhs[0]);
	}

    
    nfeats = (int) mxGetM(instance_mat_col);
    prob.l = (int) mxGetN(instance_mat_col);
    label_vector_row_num = (int) mxGetM(label_vec);
    alphas_in_vector_row_num = (int) mxGetM(alphas_vec);
    w_in_vector_row_num = (int) mxGetM(w_vec);
    w_inst_row_num = (int) mxGetM(w_inst_vec);
            
	if(label_vector_row_num!=prob.l)
	{
		mexPrintf("Length of label vector does not match # of instances.\n");
		return -1;
	}

  
	if(alphas_in_vector_row_num!=prob.l)
	{
		mexPrintf("Length of alphas vector does not match # of instances.\n");
		return -1;
	}
  
  
	if(w_in_vector_row_num!=nfeats)
	{
		mexPrintf("Length of w vector does not match # of features.\n");
		return -1;
	}

      
	if(w_inst_row_num!=prob.l)
	{
		mexPrintf("Length of w_ist vector does not match # of instances.\n");
		return -1;
	}

    w_inst = (float *) mxGetPr(w_inst_vec);    
    w = (float *) mxGetPr(w_vec);
	alphas = (float *) mxGetPr(alphas_vec);
	labels = (float *) mxGetPr(label_vec);
	samples = (float *) mxGetPr(instance_mat_col);

#ifdef _DENSE_REP
	max_index = (int) mxGetM(instance_mat_col);

	if(bias >= 0)	  
	  elements = (max_index + 1)*label_vector_row_num; 
	else
	  elements = max_index*label_vector_row_num;
	
	prob.y = Malloc(float, prob.l);
	prob.x = Malloc(float *, prob.l);
    prob.alphas_in = Malloc(float, prob.l);
    prob.w_in = Malloc(float, nfeats);
    prob.W = Malloc(float, prob.l);
    
	/*x_space = Malloc(float, elements);*/
	prob.bias = bias;

    x_space = samples; /* Joao */

	long int x_space_idx = 0, sample_idx  = 0;

  for(i=0;i<nfeats;i++)
    prob.w_in[i] =  w[i];
  
	for(i=0;i<prob.l;i++) {
        prob.x[i] = &x_space[x_space_idx];
        prob.y[i] = labels[i];
        prob.alphas_in[i] = alphas[i];
        prob.W[i] = w_inst[i];
                
        x_space_idx = x_space_idx+max_index; /* joao */
        /* Joao 
        for(j=0;j<max_index;j++)
        {
                   x_space[x_space_idx++] = samples[sample_idx++];
        }*/
        if(prob.bias >= 0)
        {
            mexErrMsgTxt("joao: not ready for this");
        }
	}

#else

#endif
	if(prob.bias>=0)
		prob.n = max_index+1;
	else
		prob.n = max_index;

	return 0;
}

void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;
	srand(1);

	if(nrhs == 7) /* force alphas_in and w_in to be initialized */
	{
		int err=0;

		if(!mxIsClass(prhs[0], "single") || !mxIsClass(prhs[1], "single") || !mxIsClass(prhs[4], "single") || !mxIsClass(prhs[4], "single") || !mxIsClass(prhs[6], "single")) {
			mexPrintf("Error: label vector, instance matrix and alphas_in must be float\n");
			fake_answer(plhs);
			return;
		}

		if(parse_command_line(nrhs, prhs, NULL))
		{
			exit_with_help();
			destroy_param(&param);
			fake_answer(plhs);
			return;
		}
#ifdef _DENSE_REP
		if(!mxIsSparse(prhs[1]))
			err = read_problem_sparse(prhs[0], prhs[1], prhs[4], prhs[5], prhs[6]);
		else
		{
			mexPrintf("Training_instance_matrix must be dense\n");
			destroy_param(&param);
			fake_answer(plhs);
			return;
		}
#else
		if(mxIsSparse(prhs[1]))
			err = read_problem_sparse(prhs[0], prhs[1], prhs[4]);
		else
		{
			mexPrintf("Training_instance_matrix must be sparse\n");
			destroy_param(&param);
			fake_answer(plhs);
			return;
		}
#endif

		error_msg = check_parameter(&prob, &param);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				mexPrintf("Error: %s\n", error_msg);
			destroy_param(&param);
			free(prob.y);
			free(prob.x);
            free(prob.alphas_in);
            free(prob.w_in);
			/*free(x_space);*/
			fake_answer(plhs);
			return;
		}
        
		if(cross_validation_flag)
		{
			float *ptr;
			plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
			ptr = (float*) mxGetPr(plhs[0]);
			ptr[0] = do_cross_validation();
		}
		else
		{            
			const char *error_msg;
			model_ = train(&prob, &param);            
            
			error_msg = model_to_matlab_structure(plhs, model_);            
            
			if(error_msg)
				mexPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
            
			destroy_model(model_);
		}
		destroy_param(&param);
		free(prob.y);
		free(prob.x);
        free(prob.alphas_in);
        free(prob.w_in);
		/*free(x_space);*/
	}
	else
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}
