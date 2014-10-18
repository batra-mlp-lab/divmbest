#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "tron.h"

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern float snrm2_(int *, float *, int *);
extern float sdot_(int *, float *, int *, float *, int *);
extern int saxpy_(int *, float *, float *, int *, float *, int *);
extern int sscal_(int *, float *, float *, int *);

#ifdef __cplusplus
}
#endif

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void TRON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*tron_print_string)(buf);
}

TRON::TRON(const function *fun_obj, float eps, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
	tron_print_string = default_print;
}

TRON::~TRON()
{
}

void TRON::tron(float *w, float *initial_w)
{
	// Parameters for updating the iterates.
	float eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	float sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	float delta, snorm, one=1.0;
	float alpha, f, fnew, prered, actred, gs;
	int search = 1, iter = 1, inc = 1;
	float *s = new float[n];
	float *r = new float[n];
	float *w_new = new float[n];
	float *g = new float[n];

	for (i=0; i<n; i++)
		w[i] = 0;

    f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	delta = snrm2_(&n, g, &inc);

    /* repeat with warm w */
    for (i=0; i<n; i++)
        w[i] = initial_w[i];

    f = fun_obj->fun(w);
	fun_obj->grad(w, g);
    /* */
    
    
	float gnorm1 = delta;
	float gnorm = gnorm1;

	if (gnorm <= eps*gnorm1)
		search = 0;

	iter = 1;

	while (iter <= max_iter && search)
	{
		cg_iter = trcg(delta, g, s, r);

		memcpy(w_new, w, sizeof(float)*n);
		saxpy_(&n, &one, s, &inc, w_new, &inc);

		gs = sdot_(&n, g, &inc, s, &inc);
		prered = -0.5*(gs-sdot_(&n, s, &inc, r, &inc));
                fnew = fun_obj->fun(w_new);

		// Compute the actual reduction.
	        actred = f - fnew;

		// On the first iteration, adjust the initial step bound.
		snorm = snrm2_(&n, s, &inc);
		if (iter == 1)
			delta = min(delta, snorm);

		// Compute prediction alpha*snorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = max(sigma1, (float)-0.5*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
		else
			delta = max(delta, min(alpha*snorm, sigma3*delta));

		/*info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);*/

		if (actred > eta0*prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(float)*n);
			f = fnew;
		        fun_obj->grad(w, g);

			gnorm = snrm2_(&n, g, &inc);
			if (gnorm <= eps*gnorm1)
				break;
		}
		if (f < -1.0e+32)
		{
			info("warning: f < -1.0e+32\n");
			break;
		}
		if (fabs(actred) <= 0 && prered <= 0)
		{
			info("warning: actred and prered <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f) &&
		    fabs(prered) <= 1.0e-12*fabs(f))
		{
			info("warning: actred and prered too small\n");
			break;
		}
	}

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
}

int TRON::trcg(float delta, float *g, float *s, float *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	float one = 1;
	float *d = new float[n];
	float *Hd = new float[n];
	float rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = 0.1*snrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = sdot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (snrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/sdot_(&n, d, &inc, Hd, &inc);
		saxpy_(&n, &alpha, d, &inc, s, &inc);
		if (snrm2_(&n, s, &inc) > delta)
		{
			/*info("cg reaches trust region boundary\n");*/
			alpha = -alpha;
			saxpy_(&n, &alpha, d, &inc, s, &inc);

			float std = sdot_(&n, s, &inc, d, &inc);
			float sts = sdot_(&n, s, &inc, s, &inc);
			float dtd = sdot_(&n, d, &inc, d, &inc);
			float dsq = delta*delta;
			float rad = sqrt(std*std + dtd*(dsq-sts));
			if (std >= 0)
				alpha = (dsq - sts)/(std + rad);
			else
				alpha = (rad - std)/dtd;
			saxpy_(&n, &alpha, d, &inc, s, &inc);
			alpha = -alpha;
			saxpy_(&n, &alpha, Hd, &inc, r, &inc);
			break;
		}
		alpha = -alpha;
		saxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = sdot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		sscal_(&n, &beta, d, &inc);
		saxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

float TRON::norm_inf(int n, float *x)
{
	float dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf))
{
	tron_print_string = print_string;
}
