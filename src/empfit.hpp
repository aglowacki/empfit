/***
Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.

Copyright 2024. UChicago Argonne, LLC. This software was produced
under U.S. Government contract DE-AC02-06CH11357 for Argonne National
Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
modified to produce derivative works, such modified software should
be clearly marked, so as not to confuse it with the version available
from ANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***/

#ifndef EMPFIT_HPP
#define EMPFIT_HPP


#include <Eigen/Core>
#include <unordered_map>
#include "fit_param.hpp"

namespace optimize
{

//-----------------------------------------------------------------------------------------------------------

template <typename T>
constexpr T MP_MachEp0()
{
  if constexpr (std::is_same_v<T, float>) 
  {
      return 1.19209e-07;
  }
  else if constexpr (std::is_same_v<T, double>) 
  {
      return 2.2204460e-16;
  }
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
constexpr T MP_Dwarf()
{
  if constexpr (std::is_same_v<T, float>) 
  {
      return 1.17549e-38;
  }
  else if constexpr (std::is_same_v<T, double>) 
  {
      return 2.2250739e-308;
  }
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
constexpr T MP_Giant()
{
  if constexpr (std::is_same_v<T, float>) 
  {
      return 3.40282e+38;
  }
  else if constexpr (std::is_same_v<T, double>) 
  {
      return 1.7976931e+308;
  }
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
constexpr T MP_RDwarf()
{
  return (sqrt(MP_Dwarf<T>()*1.5)*10);
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
constexpr T MP_RGiant()
{
  return (sqrt(MP_Giant<T>())*0.1);
}

//-----------------------------------------------------------------------------------------------------------

enum class Errors { INPUT=0, NotANum=-16, FUNC=-17, NPOINTS=-18, NFREE=-19, MEMORY=-20, INITBOUNDS=-21, BOUNDS=-22, PARAM=-23, DOF=-24, USER_QUIT=-25, OK_CHI=1, OK_PAR=2, OK_BOTH=3, OK_DIR=4, MAXITER=5, FTOL=6, XTOL=7, GTOL=8};

enum class FUNC_RET {OK=0, USER_QUIT=1};

//-----------------------------------------------------------------------------------------------------------

template <typename T>
using Callback_fuc = std::function<FUNC_RET(const data_struct::Fit_Parameters<T> * const, data_struct::ArrayTr<T>& out_resid, void* user_data)>;

//-----------------------------------------------------------------------------------------------------------

std::string ErrorToString(Errors err)
{
  switch(err)
  {
    case Errors::INPUT:
      return "General input error";
    case Errors::NotANum:
      return "User function produced non-finite values";
    case Errors::FUNC:
      return "No user function was supplied";
    case Errors::NPOINTS:
      return "No user data points were supplied";
    case Errors::NFREE:
      return "No free parameters";
    case Errors::MEMORY:
      return "Memory allocation error";
    case Errors::INITBOUNDS:
      return "Initial values inconsistent w constraints";
    case Errors::BOUNDS:
      return "Initial constraints inconsistent";
    case Errors::PARAM:
      return "General parameter error";
    case Errors::DOF:
      return "Not enough degrees of freedom";
    case Errors::USER_QUIT:
      return "User quit in function";
    case Errors::OK_CHI:
      return "Convergence in chi-square value";
    case Errors::OK_PAR:
      return "Convergence in parameter value";
    case Errors::OK_BOTH:
      return "Both MP_OK_PAR and MP_OK_CHI hold";
    case Errors::OK_DIR:
      return "Convergence in orthogonality";
    case Errors::MAXITER:
      return "Maximum number of iterations reached";
    case Errors::FTOL:
      return " ftol is too small; no further improvement";
    case Errors::XTOL:
      return " xtol is too small; no further improvement";
    case Errors::GTOL:
      return " gtol is too small; no further improvement";
  }

  return "Unknown error code";
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
struct Options 
{
  
  Options()
  {
    ftol = 1.0e10;
    xtol = 1.0e10;
    gtol = 1.0e10;
    stepfactor = 100.0;
    maxiter = 200;
    douserscale = 0;
    maxfev = 0;
    covtol = 1e-14;
    nofinitecheck = 0;
    epsfcn = MP_MachEp0<T>();    
  }

//-----------------------------------------------------------------------------------------------------------

  Options(const Options<T>& o)
  {
    if (o->ftol > 0) ftol = o->ftol;
    if (o->xtol > 0) xtol = o->xtol;
    if (o->gtol > 0) gtol = o->gtol;
    if (o->stepfactor > 0) stepfactor = o->stepfactor;
    //if (o->nprint >= 0) nprint = o->nprint;
    if (o->epsfcn > 0) epsfcn = o->epsfcn;
    if (o->maxiter > 0) maxiter = o->maxiter;
    if (o->maxiter == -1) maxiter = 0;
    if (o->douserscale != 0) douserscale = o->douserscale;
    if (o->covtol > 0) covtol = o->covtol;
    if (o->nofinitecheck > 0) nofinitecheck = o->nofinitecheck;
    maxfev = o->maxfev;
  }

  T ftol;    // Relative chi-square convergence criterium Default: 1e-10 
  T xtol;    // Relative parameter convergence criterium  Default: 1e-10 
  T gtol;    // Orthogonality convergence criterium       Default: 1e-10 
  T epsfcn;  // Finite derivative step size               Default: MP_MACHEP0 
  T stepfactor; // Initial step bound                     Default: 100.0 
  T covtol;  // Range tolerance for covariance calculation Default: 1e-14 
  int maxiter;    /* Maximum number of iterations.  If maxiter == MP_NO_ITER,
                     then basic error checking is done, and parameter
                     errors/covariances are estimated based on input
                     parameter values, but no fitting iterations are done. 
		     Default: 200
		  */
  int maxfev;     // Maximum number of function evaluations, or 0 for no limit Default: 0 (no limit) 
  //int nprint;     // Default: 1 
  int douserscale;/* Scale variables by user values?
		     1 = yes, user scale values in diag;
		     0 = no, variables scaled internally (Default) */
  int nofinitecheck; /* Disable check for infinite quantities from user?
			0 = do not perform check (Default)
			1 = perform check 
		     */
  
};

//-----------------------------------------------------------------------------------------------------------

template <typename T>
struct Result 
{
  Result(size_t size)
  {
    resid.resize(size);
    resid.setZero(size);
    xerror.resize(size);
    xerror.setZero(size);
    covar.resize(size);
    covar.setZero(size);
    nfev = 0;
    niter = 0;
    status = Errors::INPUT;
  }

  Result(Errors err)
  {
    status = err;
  }

  T bestnorm;     // Final chi^2 
  T orignorm;     // Starting value of chi^2 
  int niter;           // Number of iterations 
  int nfev;            // Number of function evaluations 
  Errors status;          // Fitting status code 
  
  int npar;            // Total number of parameters 
  int nfree;           // Number of free parameters 
  int npegged;         // Number of pegged parameters 
  int nfunc;           // Number of residuals (= num. of data points) 

  data_struct::ArrayTr<T> resid;       // Final residuals nfunc-vector, or 0 if not desired 
  data_struct::ArrayTr<T> xerror;      // Final parameter uncertainties (1-sigma) npar-vector, or 0 if not desired 
  data_struct::ArrayTr<T> covar;       // Final parameter covariance matrix npar x npar array, or 0 if not desired 

};


//-----------------------------------------------------------------------------------------------------------
/*
   *     **********
   *
   *     function enorm
   *
   *     given an n-vector x, this function calculates the
   *     euclidean norm of x.
   *
   *     the euclidean norm is computed by accumulating the sum of
   *     squares in three different sums. the sums of squares for the
   *     small and large components are scaled so that no overflows
   *     occur. non-destructive underflows are permitted. underflows
   *     and overflows do not occur in the computation of the unscaled
   *     sum of squares for the intermediate components.
   *     the definitions of small, intermediate and large components
   *     depend on two constants, rdwarf and rgiant. the main
   *     restrictions on these constants are that rdwarf**2 not
   *     underflow and rgiant**2 not overflow. the constants
   *     given here are suitable for every known computer.
   *
   *     the function statement is
   *
   *	double precision function enorm(n,x)
   *
   *     where
   *
   *	n is a positive integer input variable.
   *
   *	x is an input array of length n.
   *
   *     subprograms called
   *
   *	fortran-supplied ... dabs,dsqrt
   *
   *     argonne national laboratory. minpack project. march 1980.
   *     burton s. garbow, kenneth e. hillstrom, jorge j. more
   *
   *     **********
   */
template <typename T>
T mp_enorm(data_struct::ArrayTr<T> &out_resid) 
{
  T rdwarf = MP_RDwarf<T>();
  T rgiant = MP_RGiant<T>();
  T floatn = static_cast<T>(out_resid.size());
  T agiant = rgiant/floatn;

  T s1 = (T)0.0;
  T s2 = (T)0.0;
  T s3 = (T)0.0;
  T xabs = (T)0.0;
  T x1max = (T)0.0;
  T x3max = (T)0.0;
  T ans = (T)0.0;
  T temp = (T)0.0;
  
  data_struct::ArrayTr<T> abs_resid = out_resid.abs();

  for (int i=0; i<abs_resid.size(); i++) 
  {
    xabs = abs_resid[i];
    if ((xabs > rdwarf) && (xabs < agiant))
    {
	    // sum for intermediate components.
      s2 += xabs*xabs;
      continue;
    }
      
    if (xabs > rdwarf)
    {
	    // sum for large components.
      if (xabs > x1max)
      {
        temp = x1max/xabs;
        s1 = (T)1.0 + s1*temp*temp;
        x1max = xabs;
      }
      else
      {
        temp = xabs/x1max;
        s1 += temp*temp;
      }
      continue;
    }
    // sum for small components.
    if (xabs > x3max)
    {
      temp = x3max/xabs;
      s3 = (T)1.0 + s3*temp*temp;
      x3max = xabs;
    }
    else	
    {
      if (xabs != (T)0.0)
        {
          temp = xabs/x3max;
          s3 += temp*temp;
        }
    }
  }
  // calculation of norm. 
  if (s1 != (T)0.0) 
  {
    temp = s1 + (s2/x1max)/x1max;
    ans = x1max*sqrt(temp);
    return(ans);
  }
  if (s2 != (T)0.0) 
  {
    if (s2 >= x3max)
    {
      temp = s2*( (T)1.0 + (x3max/s2)*(x3max*s3));
    }
    else
    {
      temp = x3max*((s2/x3max)+(x3max*s3));
    }
    ans = sqrt(temp);
  }
  else
  {
      ans = x3max*sqrt(s3);
  }
  return(ans);
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
FUNC_RET mp_fdjac2(Callback_fuc<T> funct,
	      int m, int n, int *ifree, int npar, double *x, data_struct::ArrayTr<T>& fvec,
	      data_struct::ArrayTr<T>& fjac, double epsfcn,
	      double *wa, void *priv, Result<T>& result,
	      double *step, double *dstep, int *dside,
	      int *qulimited, double *ulimit,
	      int *ddebug, double *ddrtol, double *ddatol,
	      double *wa2, double **dvec)
{
  int i,j;
  double h;
  int has_analytical_deriv = 0;
  int has_numerical_deriv = 0;
  int has_debug_deriv = 0;
  
  T temp = std::max(epsfcn, MP_MachEp0());
  T eps = sqrt(temp);
  int ij = 0;

  for (j=0; j<npar; j++) dvec[j] = 0;

  /* Initialize the Jacobian derivative matrix */
  for (j=0; j<(n*m); j++) fjac[j] = 0;

  /* Check for which parameters need analytical derivatives and which
     need numerical ones */
  for (j=0; j<n; j++) 
  {
    // Loop through free parameters only 
    if (dside && dside[ifree[j]] == 3 && ddebug[ifree[j]] == 0) 
    {
      /* Purely analytical derivatives */
      dvec[ifree[j]] = fjac + j*m;
      has_analytical_deriv = 1;
    }
    else if (dside && ddebug[ifree[j]] == 1) 
    {
      /* Numerical and analytical derivatives as a debug cross-check */
      dvec[ifree[j]] = fjac + j*m;
      has_analytical_deriv = 1;
      has_numerical_deriv = 1;
      has_debug_deriv = 1;
    }
    else 
    {
      has_numerical_deriv = 1;
    }
  }

  // If there are any parameters requiring analytical derivatives, then compute them first. 
  if (has_analytical_deriv) 
  {
    FUNC_RET ret = funct(&params, out_resid, user_data);
    iflag = mp_call(funct, m, npar, x, wa, dvec, priv);
    result.nfev += 1;
    if (ret == FUNC_RET::USER_QUIT )
    {
      return ret;
    }
  }

  if (has_debug_deriv) 
  {
    printf("FJAC DEBUG BEGIN\n");
    printf("#  %10s %10s %10s %10s %10s %10s\n", "IPNT", "FUNC", "DERIV_U", "DERIV_N", "DIFF_ABS", "DIFF_REL");
  }

  // Any parameters requiring numerical derivatives 
  if (has_numerical_deriv) 
  {
    for (j=0; j<n; j++) 
    {
      // Loop thru free parms 
      int dsidei = (dside)?(dside[ifree[j]]):(0);
      int debug  = ddebug[ifree[j]];
      double dr = ddrtol[ifree[j]], da = ddatol[ifree[j]];
      
      // Check for debugging 
      if (debug) 
      {
        printf("FJAC PARM %d\n", ifree[j]);
      }

      // Skip parameters already done by user-computed partials 
      if (dside && dsidei == 3) 
      {
        ij += m; // still need to advance fjac pointer 
        continue;
      }

      temp = x[ifree[j]];
      h = eps * fabs(temp);
      if (step  &&  step[ifree[j]] > 0)
      {
        h = step[ifree[j]];
      } 
      if (dstep && dstep[ifree[j]] > 0)
      {
        h = fabs(dstep[ifree[j]]*temp);
      }
      if (h == (T)0.0)
      {
        h = eps;
      }

      // If negative step requested, or we are against the upper limit 
      if ((dside && dsidei == -1) ||  (dside && dsidei == 0 && qulimited && ulimit && qulimited[j] && (temp > (ulimit[j]-h)))) 
      {
        h = -h;
      }

      x[ifree[j]] = temp + h;
      FUNC_RET ret = funct(&params, out_resid, user_data);
      iflag = mp_call(funct, m, npar, x, wa, 0, priv);
      result.nfev += 1;
      if (ret == FUNC_RET::USER_QUIT )
      {
        return ret;
      }
      x[ifree[j]] = temp;

      if (dsidei <= 1) 
      {
        // COMPUTE THE ONE-SIDED DERIVATIVE 
        if (! debug) 
        {
          // Non-debug path for speed 
          for (i=0; i<m; i++, ij++) 
          {
            fjac[ij] = (wa[i] - fvec[i])/h; // fjac[i+m*j] 
          }
        }
        else
        {
          // Debug path for correctness 
          for (i=0; i<m; i++, ij++) 
          {
            double fjold = fjac[ij];
            fjac[ij] = (wa[i] - fvec[i])/h; // fjac[i+m*j] 
            if ((da == 0 && dr == 0 && (fjold != 0 || fjac[ij] != 0)) || ((da != 0 || dr != 0) && (fabs(fjold-fjac[ij]) > da + fabs(fjold)*dr))) 
            {
              printf("   %10d %10.4g %10.4g %10.4g %10.4g %10.4g\n",  i, fvec[i], fjold, fjac[ij], fjold-fjac[ij], (fjold == 0)?(0):((fjold-fjac[ij])/fjold));
            }
          }
        } // end debugging

      } 
      else
      {
        // dside > 2 
        // COMPUTE THE TWO-SIDED DERIVATIVE 
        for (i=0; i<m; i++) 
        {
          wa2[i] = wa[i];
        }

        // Evaluate at x - h 
        x[ifree[j]] = temp - h;
        FUNC_RET ret = funct(&params, out_resid, user_data);
        iflag = mp_call(funct, m, npar, x, wa, 0, priv);
        result.nfev += 1;
        if (ret == FUNC_RET::USER_QUIT)
        {
          return ret;
        }
        x[ifree[j]] = temp;

        // Now compute derivative as (f(x+h) - f(x-h))/(2h) 
        if (! debug ) 
        {
          // Non-debug path for speed 
          for (i=0; i<m; i++, ij++) 
          {
            fjac[ij] = (wa2[ij] - wa[i])/(2*h); // fjac[i+m*j]
          }
              
        }
        else
        {
          // Debug path for correctness 
          for (i=0; i<m; i++, ij++) 
          {
            double fjold = fjac[ij];
            fjac[ij] = (wa2[i] - wa[i])/(2*h); // fjac[i+m*j] 
            if ((da == 0 && dr == 0 && (fjold != 0 || fjac[ij] != 0)) || ((da != 0 || dr != 0) && (fabs(fjold-fjac[ij]) > da + fabs(fjold)*dr))) 
            {
              printf("   %10d %10.4g %10.4g %10.4g %10.4g %10.4g\n", i, fvec[i], fjold, fjac[ij], fjold-fjac[ij], (fjold == 0)?(0):((fjold-fjac[ij])/fjold));
            }
          }
        } // end debugging 
        
      } // if (dside > 2) 
    }  // end for
  } // if (has_numerical_derivative) 
  if (has_debug_deriv) 
  {
    printf("FJAC DEBUG END\n");
  }

  return FUNC_RET::OK; 
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
Result<T> empfit(Callback_fuc<T> funct, size_t resid_size, const data_struct::Fit_Parameters<T>& params, Options<T>* my_options, void *user_data)
{
  if(funct == nullptr)
  {
    return Result<T> (Errors::FUNC);
  }
  if(resid_size <= 0)
  {
    return Result<T> (Errors::NPOINTS);
  }
  if(params.size() == 0)
  {
    return Result<T> (Errors::NFREE);
  }
  // check if we have any non fixed params
  size_t num_free_params = 0;
  for(const auto& itr : params)
  {
    if(itr.second.bound_type != data_struct::Fit_Bound::FIXED)
    {
      switch(itr.second.bound_type)
      {
        case data_struct::Fit_Bound::LIMITED_HI:
          if( itr.second.value > itr.second.max_val )
          {
            return Result<T> (Errors::INITBOUNDS);
          }
        break;
        case data_struct::Fit_Bound::LIMITED_LO:
          if( itr.second.value < itr.second.min_val )
          {
            return Result<T> (Errors::INITBOUNDS);
          }
        break;
        case data_struct::Fit_Bound::LIMITED_LO_HI:
          if( itr.second.value > itr.second.max_val )
          {
            return Result<T> (Errors::INITBOUNDS);
          }
          if( itr.second.value < itr.second.min_val )
          {
            return Result<T> (Errors::INITBOUNDS);
          }
          if(itr.second.min_val > itr.second.max_val)
          {
            return Result<T> (Errors::BOUNDS);
          }
        break;
      }
      num_free_params ++;
    }
  }
  if(num_free_params == 0)
  {
    return Result<T> (Errors::NFREE);
  }
  else if(resid_size < num_free_params)
  {
    return Result<T> (Errors::DOF);
  }

  // initialze result
  Result<T> result(params.size());
  data_struct::ArrayTr<T> out_resid;
  out_resid.resize(resid_size);
  out_resid.setZero(resid_size);

  Options<T> options;
  if(my_options != nullptr)
  {
      options = *my_options;
  }

  int nfree = 0;
  int npegged = 0;
  T fnorm = (T)-1.0;
  T fnorm1 = (T)-1.0;
  T xnorm = (T)-1.0;
  T delta = (T)0.0;

  data_struct::ArrayTr<T> fjac(resid_size * num_free_params);
  /*
  mp_malloc(fvec, double, m);
  mp_malloc(qtf, double, nfree);
  mp_malloc(x, double, nfree);
  mp_malloc(xnew, double, npar);
  mp_malloc(fjac, double, m*nfree);
  mp_malloc(diag, double, npar);
  mp_malloc(wa1, double, npar);
  mp_malloc(wa2, double, m); // Maximum usage is "m" in mpfit_fdjac2() 
  mp_malloc(wa3, double, npar);
  mp_malloc(wa4, double, m);
  mp_malloc(ipvt, int, npar);
  mp_malloc(DArrptr, double *, npar);
  */

  FUNC_RET ret = funct(&params, out_resid, user_data);
  result.nfev += 1;
  if(ret == FUNC_RET::USER_QUIT)
  {
    result.status = Errors::USER_QUIT;  
  }


  fnorm = mp_enorm(out_resid);
  result.orignorm = fnorm*fnorm;
/*
  // Make a new copy 
  for (i=0; i<npar; i++) {
    xnew[i] = xall[i];
  }

  // Transfer free parameters to 'x' 
  for (i=0; i<nfree; i++) {
    x[i] = xall[ifree[i]];
  }

  // Initialize Levelberg-Marquardt parameter and iteration counter 

  par = 0.0;
  iter = 1;
  for (i=0; i<nfree; i++) {
    qtf[i] = 0;
  }

  // Beginning of the outer loop 
 OUTER_LOOP:
  for (i=0; i<nfree; i++) {
    xnew[ifree[i]] = x[i];
  }

   iflag = mp_fdjac2(funct, m, nfree, ifree, npar, xnew, fvec, fjac, ldfjac,
		    conf.epsfcn, wa4, private_data, &nfev,
		    step, dstep, mpside, qulim, ulim,
		    ddebug, ddrtol, ddatol, wa2, dvecptr);
  if (iflag < 0) {
    goto CLEANUP;
  }
  // dvecptr is double ** 
  // dvecptr[npars] [?]
*/
  ret = mp_fdjac2(funct, m, nfree, ifree, npar, xnew, out_resid, fjac,
		    conf.epsfcn, wa4, private_data, &result,
		    step, dstep, mpside, qulim, ulim,
		    ddebug, ddrtol, ddatol, wa2, dvecptr);
  if(ret == FUNC_RET::USER_QUIT)
  {
    result.status = Errors::USER_QUIT;  
  }



  return result;
}


}
#endif