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
#include <Eigen/Dense>
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
  size_t maxiter;    /* Maximum number of iterations.  If maxiter == MP_NO_ITER,
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
  
  int nfree;           // Number of free parameters 
  int npegged;         // Number of pegged parameters 
  int nfunc;           // Number of residuals (= num. of data points) 

  data_struct::ArrayTr<T> resid;       // Final residuals nfunc-vector, or 0 if not desired 
  data_struct::ArrayTr<T> xerror;      // Final parameter uncertainties (1-sigma) npar-vector, or 0 if not desired 
  data_struct::ArrayTr<T> covar;       // Final parameter covariance matrix npar x npar array, or 0 if not desired 

};


//-----------------------------------------------------------------------------------------------------------

template <typename T>
T mp_enorm(data_struct::ArrayTr<T> &out_resid, int start_offset=0) 
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

  for (int i=start_offset; i<abs_resid.size(); i++) 
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
	      data_struct::Fit_Parameters<T>& params, data_struct::ArrayTr<T>& out_resid,
	      std::unordered_map<std::string, data_struct::ArrayTr<T> >& fjac,
        T epsfcn, data_struct::ArrayTr<T> &wa, void *user_data, Result<T>& result,
	      data_struct::ArrayTr<T> &wa2)
{
  T h = (T)0.0;
  int has_analytical_deriv = 0;
  int has_numerical_deriv = 0;
  int has_debug_deriv = 0;
  
  T temp = std::max(epsfcn, MP_MachEp0<T>());
  T eps = sqrt(temp);
  //size_t resid_size = out_resid.size();

  //for (j=0; j<npar; j++) dvec[j] = 0;

  /* Check for which parameters need analytical derivatives and which
     need numerical ones */
  //for (j=0; j<result.nfree; j++) 
  for(auto &itr : fjac)
  {
    data_struct::Fit_Param<T>& param = params.at(itr.first);
    // Loop through free parameters only 
    if (param.side == data_struct::Derivative::User && false == param.debug) 
    {
      /* Purely analytical derivatives */
      //dvec[ifree[j]] = fjac + j * resid_size;
      has_analytical_deriv = 1;
    }
    else if (param.debug) 
    {
      /* Numerical and analytical derivatives as a debug cross-check */
      //dvec[ifree[j]] = fjac + j * resid_size;
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
    FUNC_RET ret = funct(&params, wa, user_data);
    //iflag = mp_call(funct, resid_size, npar, x, wa, dvec, priv);
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
    //for (j=0; j<result.nfree; j++) 
    for(auto &itr : fjac)
    {
      data_struct::Fit_Param<T>& param = params.at(itr.first);
      // Loop thru free parms 
      
      // Check for debugging 
      /* TODO: fix print
      if (param.debug) 
      {
        printf("FJAC PARM %s\n", pitr.first);
      }
      */
      // Skip parameters already done by user-computed partials 
      if (param.side == data_struct::Derivative::User) 
      {
        continue;
      }

      temp = param.value;
      h = eps * fabs(temp);
      if (param.step_size > 0)
      {
        h = param.step_size;
      } 
      if (param.relstep > 0) 
      {
        h = fabs(param.relstep * temp);
      }
      if (h == (T)0.0)
      {
        h = eps;
      }

      // If negative step requested, or we are against the upper limit 
      if ((param.side == data_struct::Derivative::NegOneSide) 
      ||  (param.side == data_struct::Derivative::AutoOneSide 
      && (param.bound_type == data_struct::Fit_Bound::LIMITED_HI || param.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
      && (temp > (param.max_val - h)))) 
      {
        h = -h;
      }

      param.value = temp + h;
      FUNC_RET ret = funct(&params, wa, user_data);
      //iflag = mp_call(funct, resid_size, npar, x, wa, 0, priv);
      result.nfev += 1;
      if (ret == FUNC_RET::USER_QUIT )
      {
        return ret;
      }
      param.value = temp;

      if (param.side <= data_struct::Derivative::OneSide) 
      {
        // COMPUTE THE ONE-SIDED DERIVATIVE 
        if (false == param.debug) 
        {
          // Non-debug path for speed 
          itr.second = (wa - out_resid)/h; 
        }
        else
        {
          // Debug path for correctness 
          itr.second = (wa - out_resid)/h; 
          /* TODO: fix debug print
          for (i=0; i<resid_size; i++, ij++) 
          {
            T fjold = fjac[ij];
            
            if ((param.deriv_abstol == 0 && param.deriv_reltol == 0 && (fjold != 0 || fjac[ij] != 0)) || ((param.deriv_abstol != 0 || param.deriv_reltol != 0) && (fabs(fjold-fjac[ij]) > param.deriv_abstol + fabs(fjold) * param.deriv_reltol))) 
            {
              printf("   %10d %10.4g %10.4g %10.4g %10.4g %10.4g\n",  i, out_resid[i], fjold, fjac[ij], fjold-fjac[ij], (fjold == 0)?(0):((fjold-fjac[ij])/fjold));
            }
          }
          */
        } // end debugging

      } 
      else
      {
        // dside > 2 
        // COMPUTE THE TWO-SIDED DERIVATIVE 
        wa2 = wa;

        // Evaluate at x - h 
        param.value = temp - h;
        FUNC_RET ret = funct(&params, wa, user_data);
        //iflag = mp_call(funct, resid_size, npar, x, wa, 0, priv);
        result.nfev += 1;
        if (ret == FUNC_RET::USER_QUIT)
        {
          return ret;
        }
        param.value = temp;

        // Now compute derivative as (f(x+h) - f(x-h))/(2h) 
        if (false == param.debug ) 
        {
          // Non-debug path for speed 
          itr.second = (wa2 - wa)/(2*h);    
        }
        else
        {
          itr.second = (wa2 - wa)/(2*h);
          /* TODO: fix debug print
          // Debug path for correctness 
          for (i=0; i<resid_size; i++, ij++) 
          {
            T fjold = fjac[ij];
            
            if ((param.deriv_abstol == 0 && param.deriv_reltol == 0 && (fjold != 0 || fjac[ij] != 0)) || ((param.deriv_abstol != 0 || param.deriv_reltol != 0) && (fabs(fjold-fjac[ij]) > param.deriv_abstol + fabs(fjold) * param.deriv_reltol))) 
            {
              printf("   %10d %10.4g %10.4g %10.4g %10.4g %10.4g\n", i, out_resid[i], fjold, fjac[ij], fjold-fjac[ij], (fjold == 0)?(0):((fjold-fjac[ij])/fjold));
            }
          }
          */
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
void mp_qrfac(std::unordered_map<std::string, data_struct::ArrayTr<T> >& fjac, int pivot, data_struct::ArrayTr<int>& ipvt, data_struct::ArrayTr<T>&  rdiag,
             data_struct::ArrayTr<T>& acnorm, data_struct::ArrayTr<T>&  wa)
{
  T ajnorm,sum,temp;

  // compute the initial column norms and initialize several arrays.
  int j = 0;
  int m = 0;
  for(auto& itr: fjac)
  {
    m = itr.second.size();
    acnorm[j] = mp_enorm<T>(itr.second);
    rdiag[j] = acnorm[j];
    wa[j] = rdiag[j];
    if (pivot != 0)
    {
      ipvt[j] = j;
    }
    j++;
  }
  
  // reduce a to r with householder transformations.
  int n = fjac.size();
  j = 0;
  for(auto& itr: fjac) 
  {
    if (pivot == 1)
    {
      // bring the column of largest norm into the pivot position.
      int kmax = j;
      for (int k=j; k<n; k++)
      {
        if (rdiag[k] > rdiag[kmax])
        {
          kmax = k;
        }
      }
      if (kmax != j)
      {  
        int ij = j;
        int jj = kmax;
        for (int i=0; i<m; i++)
        {
          temp = itr.second[ij]; 
          itr.second[ij] = itr.second[jj];
          itr.second[jj] = temp;
          ij += 1;
          jj += 1;
        }
        rdiag[kmax] = rdiag[j];
        wa[kmax] = wa[j];
        int k = ipvt[j];
        ipvt[j] = ipvt[kmax];
        ipvt[kmax] = k;
      }
    }  
    // compute the householder transformation to reduce the j-th column of a to a multiple of the j-th unit vector.
    int jj = j;
    ajnorm = mp_enorm<T>(itr.second, j);
    if (ajnorm != 0.0)
    {
      if (itr.second[jj] < 0.0)
      {
        ajnorm = -ajnorm;
      }
      int ij = jj;
      for (int i=j; i<m; i++)
      {
        itr.second[ij] /= ajnorm;
        ij += 1; 
      }
      itr.second[jj] += 1.0;
      // apply the transformation to the remaining columns and update the norms.
      int jp1 = j + 1;
      if (jp1 < n)
      {
        for (int k=jp1; k<n; k++)
        {
          sum = 0.0;
          ij = j + k;
          jj = j + j;
          for (int i=j; i<m; i++)
          {
            sum += itr.second[jj] * itr.second[ij];
            ij += 1; 
            jj += 1; 
          }
          temp = sum / itr.second[j];
          ij = j + k;
          jj = j + j;
          for (int i=j; i<m; i++)
          {
            itr.second[ij] -= temp * itr.second[jj];
            ij += 1;
            jj += 1;
          }
          if ((pivot != 0) && (rdiag[k] != 0.0))
          {
            temp = itr.second[j+k] / rdiag[k];
            temp = std::max( 0.0, 1.0 - temp * temp );
            rdiag[k] *= sqrt(temp);
            temp = rdiag[k] / wa[k];
            if (((T)0.05 * temp * temp) <= MP_MachEp0<T>())
            {
              rdiag[k] = mp_enorm(itr.second, jp1+k);
              wa[k] = rdiag[k];
            }
          }
        }
      }
    }  
    rdiag[j] = -ajnorm;
    j++;
  }
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
void mp_qrsolv(std::unordered_map<std::string, data_struct::ArrayTr<T> >& fjac, data_struct::ArrayTr<int>& ipvt, data_struct::ArrayTr<T>& diag,
	       data_struct::ArrayTr<T>& qtb, data_struct::ArrayTr<T>& x, data_struct::ArrayTr<T>& sdiag, data_struct::ArrayTr<T>& wa)
{

  int n = fjac.size();
  int i,ij,ik,jp1,k,kp1,l,nsing;
  double cosx,cotan,qtbpj,sinx,sum,tanx,temp;
  
   //     copy r and (q transpose)*b to preserve input and initialize s.
   //     in particular, save the diagonal elements of r in x.
   
  int kk = 0;
  int j=0;
  /*
  for (int j=0; j<n; j++) 
  {
    ij = kk;
    ik = kk;
    for (int i=j; i<n; i++)
    {
      fjac[ij] = fjac[ik];
      ij += 1;   // [i+ldr*j] 
      ik += ldr; // [j+ldr*i]
    }
    x[j] = fjac[kk];
    wa[j] = qtb[j];
    kk += ldr+1; // j+ldr*j 
  }
  
   //     eliminate the diagonal matrix d using a givens rotation.
  for (j=0; j<n; j++) 
  {
    //	 prepare the row of d to be eliminated, locating the
    //	 diagonal element using p from the qr factorization.
    l = ipvt[j];
    if (diag[l] == (T)0.0)
    {
      goto L90;
    }
    for (k=j; k<n; k++)
    {
      sdiag[k] = (T)0.0;
    }
    sdiag[j] = diag[l];
    
    //	 the transformations to eliminate the row of d
    //	 modify only a single element of (q transpose)*b
    //	 beyond the first n, which is initially zero.
     
    qtbpj = (T)0.0;
    for (k=j; k<n; k++)
    {
    
      //	    determine a givens rotation which eliminates the
      //	    appropriate element in the current row of d.
      
      if (sdiag[k] == (T)0.0)
      {
        continue;
      }
      kk = k;
      if (fabs(fjac[kk]) < fabs(sdiag[k]))
      {
        cotan = fjac[kk] / sdiag[k];
        sinx = (T)0.5 / sqrt((T)0.25 + (T)0.25 * cotan * cotan);
        cosx = sinx * cotan;
      }
      else
      {
        tanx = sdiag[k]/fjac[kk];
        cosx = (T)0.5 / sqrt((T)0.25 + (T)0.25 * tanx * tanx);
        sinx = cosx*tanx;
      }
      
      //	    compute the modified diagonal element of r and
      //	    the modified element of ((q transpose)*b,0).
      
      fjac[kk] = cosx*fjac[kk] + sinx * sdiag[k];
      temp = cosx*wa[k] + sinx*qtbpj;
      qtbpj = -sinx*wa[k] + cosx*qtbpj;
      wa[k] = temp;
      
      //	    accumulate the tranformation in the row of s.
      kp1 = k + 1;
      if (n > kp1)
      {
        ik = kk + 1;
        for (i=kp1; i<n; i++)
        {
          temp = cosx*fjac[ik] + sinx*sdiag[i];
          sdiag[i] = -sinx*fjac[ik] + cosx*sdiag[i];
          fjac[ik] = temp;
          ik += 1; // [i+ldr*k] 
        }
      }
    }
    L90:
    
    //	 store the diagonal element of s and restore
    //	 the corresponding diagonal element of r.
     
    kk = j;
    sdiag[j] = fjac[kk];
    fjac[kk] = x[j];
  }
  */
  //     solve the triangular system for z. if the system is
  //     singular, then obtain a least squares solution.
   
  nsing = n;
  for (j=0; j<n; j++) 
  {
    if ((sdiag[j] == (T)0.0) && (nsing == n))
    {
      nsing = j;
    }
    if (nsing < n)
    {
      wa[j] = (T)0.0;
    }
  }
  if (nsing < 1)
  {
    goto L150;
  }
  /*
  for (k=0; k<nsing; k++) 
  {
    j = nsing - k - 1;
    sum = (T)0.0;
    jp1 = j + 1;
    if (nsing > jp1)
    {
      ij = jp1;
      for (i=jp1; i<nsing; i++)
      {
        sum += fjac[ij]*wa[i];
        ij += 1; // [i+ldr*j] 
      }
    }
    wa[j] = (wa[j] - sum)/sdiag[j];
  }
  */
 L150:
  
   //  permute the components of z back to components of x.
   
  for (j=0; j<n; j++) 
  {
    l = ipvt[j];
    x[l] = wa[j];
  }

}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
void mp_lmpar(std::unordered_map<std::string, data_struct::ArrayTr<T> >& fjac, data_struct::ArrayTr<int>& ipvt, data_struct::ArrayTr<T> &diag,
	      data_struct::ArrayTr<T>& qtb, T delta, T& par, data_struct::ArrayTr<T>& x,
	      data_struct::ArrayTr<T>& sdiag, data_struct::ArrayTr<T>& wa1, data_struct::ArrayTr<T>&  wa2) 
{

  int i,iter,ij,jj,j,jm1,jp1,k,l,nsing;
  double dxnorm,fp,gnorm,parc,parl,paru;
  double sum,temp;
  
   //     compute and store in x the gauss-newton direction. if the
   //     jacobian is rank-deficient, obtain a least squares solution.
  nsing = fjac.size();
  int n = fjac.size();
  jj = 0;
  /*
  for (j=0; j<n; j++) 
  {
    wa1[j] = qtb[j];
    if ((fjac[jj] == (T)0.0) && (nsing == n))
    {
      nsing = j;
    }
    if (nsing < n)
    {
      wa1[j] = (T)0.0;
    }
    jj += ldr+1; // [j+ldr*j] 
  }
  
  if (nsing >= 1) 
  {
    for (k=0; k<nsing; k++)
    {
      j = nsing - k - 1;
      wa1[j] = wa1[j] / fjac[j];
      temp = wa1[j];
      jm1 = j - 1;
      if (jm1 >= 0)
      {
        ij = ldr * j;
        for (i=0; i<=jm1; i++)
        {
          wa1[i] -= fjac[ij]*temp;
          ij += 1;
        }
      }
    }
  }
  */
  for (j=0; j<n; j++) 
  {
    l = ipvt[j];
    x[l] = wa1[j];
  }
  
   //     initialize the iteration counter.
   //     evaluate the function at the origin, and test
   //     for acceptance of the gauss-newton direction.
   
  iter = 0;
  for (j=0; j<n; j++)
  {
    wa2[j] = diag[j] * x[j];
  }
  dxnorm = mp_enorm(wa2); //mp_enorm(n,wa2);
  fp = dxnorm - delta;
  if (fp <= (T)0.1 * delta)
  {
    if (iter == 0)
    {
      par = (T)0.0;
    }
    return;
  }
  
   //     if the jacobian is not rank deficient, the newton
   //     step provides a lower bound, parl, for the zero of
   //     the function. otherwise set this bound to zero.
   
  parl = (T)0.0;
  if (nsing >= n) 
  {
    for (j=0; j<n; j++)
    {
	    l = ipvt[j];
	    wa1[j] = diag[l] * (wa2[l] / dxnorm);
    }
    jj = 0;
    j=0;
    for (auto& itr : fjac)
    {
    	sum = (T)0.0;
    	jm1 = j - 1;
      if (jm1 >= 0)
      {
        ij = jj;
        for (i=0; i<=jm1; i++)
        {
          sum += itr.second[ij]*wa1[i];
          ij += 1;
        }
      }
	    wa1[j] = (wa1[j] - sum) / itr.second[j];
	    jj ++; // [i+ldr*j] 
    }
    temp = mp_enorm(wa1); //mp_enorm(n,wa1);
    parl = ( (fp / delta) / temp) / temp;
  }
  
   //     calculate an upper bound, paru, for the zero of the function.
   
  jj = 0;
  j = 0;
  for (auto& itr : fjac)
  {
    sum = (T)0.0;
    ij = jj;
    for (i=0; i<=j; i++)
    {
	    sum += itr.second[ij]*qtb[i];
	    ij += 1;
    }
    l = ipvt[j];
    wa1[j] = sum /  diag[l];
    jj ++; // [i+ldr*j] 
    j++;
  }
  gnorm = mp_enorm(wa1); //mp_enorm(n,wa1);
  paru = gnorm/delta;
  if (paru == (T)0.0)
  {
    paru = MP_Dwarf<T>() / std::min(delta,(T)0.1);
  }
  
   //     if the input par lies outside of the interval (parl,paru),
   //     set par to the closer endpoint.
   
  par = std::max( par, parl);
  par = std::min( par, paru);
  if (par == (T)0.0)
  {
    par = gnorm/dxnorm;
  }
  
   //     beginning of an iteration.
   
  do
  {
    
    iter += 1;
    
    //	 evaluate the function at the current value of par.
    
    if (par == (T)0.0)
    {
      par = std::max(MP_Dwarf<T>(), (T)0.001 * paru);
    }
    temp = sqrt( par );
    for (j=0; j<n; j++)
    {
      wa1[j] = temp*diag[j];
    }

    mp_qrsolv(fjac,ipvt,wa1,qtb,x,sdiag,wa2);

    for (j=0; j<n; j++)
    {
      wa2[j] = diag[j] * x[j];
    }
    dxnorm = mp_enorm(wa2); //mp_enorm(n,wa2);
    temp = fp;
    fp = dxnorm - delta;
    
    //	 if the function is small enough, accept the current value
    //	 of par. also test for the exceptional cases where parl
    //	 is zero or the number of iterations has reached 10.
    
    if ((fabs(fp) <= (T)0.1 * delta)
        || ((parl == (T)0.0) && (fp <= temp) && (temp < (T)0.0))
        || (iter == 10))
    {
      break;
    }
    
    //	 compute the newton correction.
    
    for (j=0; j<n; j++) 
    {
      l = ipvt[j];
      wa1[j] = diag[l] * (wa2[l] / dxnorm);
    }
    jj = 0;
    j = 0;
    for (auto& itr : fjac)
    {
      wa1[j] = wa1[j] / sdiag[j];
      temp = wa1[j];
      jp1 = j + 1;
      if (jp1 < n)
      {
        ij = jp1 + jj;
        for (i=jp1; i<n; i++)
        {
          wa1[i] -= itr.second[ij] * temp;
          ij += 1; 
        }
    }
    jj ++;
    j++; 
    }
    temp = mp_enorm(wa1);
    parc = (( fp / delta ) / temp) / temp;
    
    //	 depending on the sign of the function, update parl or paru.
    
    if (fp > (T)0.0)
    {
      parl = std::max(parl, par);
    }
    if (fp < (T)0.0)
    {
      paru = std::min(paru, par);
    }
    
    //	 compute an improved estimate for par.
    par = std::max(parl, par + parc);
  
  }
  while((fabs(fp) <= (T)0.1 * delta)
      || ((parl == (T)0.0) && (fp <= temp) && (temp < (T)0.0))
      || (iter == 10));
  
}

//-----------------------------------------------------------------------------------------------------------

template <typename T>
Result<T> empfit(Callback_fuc<T> funct, size_t resid_size, data_struct::Fit_Parameters<T>& params, Options<T>* my_options, void *user_data)
{
  bool qanylim = false;
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
          qanylim = true;
        break;
        case data_struct::Fit_Bound::LIMITED_LO:
          if( itr.second.value < itr.second.min_val )
          {
            return Result<T> (Errors::INITBOUNDS);
          }
          qanylim = true;
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
          qanylim = true;
        break;
        default:
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
  result.nfree = num_free_params;
  data_struct::ArrayTr<T> out_resid;
  out_resid.resize(resid_size);
  out_resid.setZero(resid_size);

  Options<T> options;
  if(my_options != nullptr)
  {
      options = *my_options;
  }

  T fnorm = (T)-1.0;
  T fnorm1 = (T)-1.0;
  T xnorm = (T)-1.0;
  T delta = (T)0.0;
  T temp = (T)0.0;

  data_struct::ArrayTr<T> diag(num_free_params);
  data_struct::ArrayTr<T> qtf(num_free_params);
  qtf.setZero(num_free_params);
  // Initialize the Jacobian derivative matrix 
  // param_name, jac array
  std::unordered_map<std::string, data_struct::ArrayTr<T> > fjac;
  for(const auto& itr : params)
  {
    if(itr.second.bound_type != data_struct::Fit_Bound::FIXED)
    {
      fjac[itr.first].resize(resid_size);
      fjac[itr.first].setZero(resid_size);
    }
  }
  data_struct::MatrixTr<T> A = data_struct::MatrixTr<T>::Zero(resid_size, params.size()); // Used for QR decomp of fjac

  data_struct::ArrayTr<int> ipvt(num_free_params);
  data_struct::ArrayTr<T> acnorm(num_free_params);
  // alloc work array so we don't need to keep allocating in the functions
  data_struct::ArrayTr<T> wa1(num_free_params);
  data_struct::ArrayTr<T> wa2(resid_size);
  data_struct::ArrayTr<T> wa3(num_free_params);
  data_struct::ArrayTr<T> wa4(resid_size);
  data_struct::ArrayTr<T> wa5(num_free_params);
  
  FUNC_RET ret = funct(&params, out_resid, user_data);
  result.nfev += 1;
  if(ret == FUNC_RET::USER_QUIT)
  {
    result.status = Errors::USER_QUIT;  
  }

  // try and compare with 
  ////data_struct::VectorTr<T> fvec = out_resid;
  ////result.orignorm = fvec.squaredNorm();
  fnorm = mp_enorm<T>(out_resid);
  result.orignorm = fnorm*fnorm;

  // Make a new copy 
  data_struct::Fit_Parameters<T> saved_params = params;

  // Initialize Levelberg-Marquardt parameter and iteration counter 
  T par = 0.0;
  size_t iter = 1;
 
  // Beginning of the outer loop 
 ////OUTER_LOOP:
  

  ret = mp_fdjac2(funct, params, out_resid, fjac, options.epsfcn, wa4, user_data, result, wa2);
  if(ret == FUNC_RET::USER_QUIT)
  {
    result.status = Errors::USER_QUIT;  
    return result;
  }


  // Determine if any of the parameters are pegged at the limits 
  for(auto &pitr : params)
  {
    if(pitr.second.bound_type == data_struct::Fit_Bound::FIXED || pitr.second.bound_type == data_struct::Fit_Bound::FIT)
    {
      continue;
    }
    
    bool lpegged = false;
    bool upegged = false;
    if(pitr.second.bound_type == data_struct::Fit_Bound::LIMITED_LO || pitr.second.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
    {
      // yes check if the params is equal, later we check if <=
      lpegged = (pitr.second.value == pitr.second.min_val);
    }
    if(pitr.second.bound_type == data_struct::Fit_Bound::LIMITED_HI || pitr.second.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
    {
      // yes check if the params is equal, later we check if >=
      lpegged = (pitr.second.value == pitr.second.max_val);
    }
    
    T sum = 0;

    // If the parameter is pegged at a limit, compute the gradient direction 
    if (lpegged || upegged) 
    {
      data_struct::ArrayTr<T> prod = out_resid * fjac[pitr.first];
      sum = prod.sum();
    }
    // If pegged at lower limit and gradient is toward negative then reset gradient to zero 
    if (lpegged && (sum > 0)) 
    {
      fjac[pitr.first].setZero(resid_size);
    }
    // If pegged at upper limit and gradient is toward positive then reset gradient to zero 
    if (upegged && (sum < 0)) 
    {
      fjac[pitr.first].setZero(resid_size);
    }
  } 

  // Compute the QR factorization of the jacobian 
  // 1 = pivot , ipvt is identity matrix, Fjac is M x N (nfree) . wa1 (maybe Q), wa2 (diag),  wa3 (temp storage)
  mp_qrfac(fjac, 1, ipvt,  wa1, acnorm, wa3);
  /*
  // use eigen QR decomp ColPivHouseholderQR or FullPivHouseholderQR
  // Fill A with fjac
  int r=0;
  for(auto& itr : fjac)
  {
    // do this only if we call FullPivHouseholderQR, otherwise it is handled in mp_qrfac
    acnorm[r] = mp_enorm<T>(itr.second);
    for(int c = 0; c < resid_size; c++)
    {
      A(r,c) = itr.second(c);
    }
    r++;
  }
  Eigen::FullPivHouseholderQR<data_struct::MatrixTr<T>> qr(A);
  // Solve a linear system of equations using the QR decomposition
  //VectorXd x = qr.solve(b);
  //std::cout << "Reconstructed Matrix from QR Decomposition with Pivoting:\n" << qr.householderQ() * qr.matrixQR() << "\n";
  */

// on the first iteration and if mode is 1, scale according to the norms of the columns of the initial jacobian.

  if (iter == 1) 
  {
    if (options.douserscale == 0) 
    {
      diag = acnorm.unaryExpr([](T v) { return v == 0.0 ? (T)1.0 : v; });
    }

    for(auto& itr: fjac)
    {
      data_struct::Fit_Param<T>& param = params.at(itr.first);
      // on the first iteration, calculate the norm of the scaled x and initialize the step bound delta.
      wa3 = diag * param.value;
    }
    
    xnorm = mp_enorm<T>(wa3);
    delta = options.stepfactor*xnorm;
    if (delta == (T)0.0)
    {
       delta = options.stepfactor;
    }
  }

  // form (q transpose)*fvec and store the first n components in qtf.
  wa4 = out_resid;

  int j=0;
  for(auto &itr : fjac)
  {
    if (itr.second[j] != (T)0.0) 
    {
      T sum = (T)0.0;
      for (size_t i=j; i<resid_size; i++ ) 
      {
        sum += itr.second[i] * wa4[i];
      }
      temp = -sum / itr.second[j];
      for (size_t i=j; i<resid_size; i++ ) 
      {
        wa4[i] += itr.second[i] * temp;
      }
    }
    itr.second[j] = wa1[j];
    qtf[j] = wa4[j];
    j++;
  }

  // ( From this point on, only the square matrix, consisting of the triangle of R, is needed.) 

  if (options.nofinitecheck) 
  {
    // Check for overflow.  This should be a cheap test here since FJAC
    //   has been reduced to a (small) square matrix, and the test is
    //   O(N^2). 

    for (auto& itr: fjac) 
    {
	    if (itr.second.isInf().any()) 
      {
        result.status = Errors::NotANum;
        return result;
      }
    }

  }

  //  compute the norm of the scaled gradient.
  T gnorm = (T)0.0;
  if (fnorm != (T)0.0) 
  {
    int j=0;
    for (auto& itr: fjac) 
    {
      int l = ipvt[j];
      if (wa2[l] != (T)0.0) 
      {
        T sum = (itr.second * (qtf[j]/fnorm)).sum();
	      gnorm = std::max(gnorm,fabs(sum/wa2[l]));
      }
      j++;
    }
  }


  //	 test for convergence of the gradient norm.
  if (gnorm <= options.gtol)
  {
     result.status = Errors::OK_DIR;
  }
  if (result.status != Errors::INPUT)
  {
     ////goto L300;
  }
  if (options.maxiter == 0) 
  {
    result.status = Errors::MAXITER;
    ////goto L300;
  }

  //	 rescale if necessary.
  if (options.douserscale == 0) 
  {
    int j=0;
    for (auto& itr: fjac) 
    {
      diag[j] = std::max(diag[j],wa2[j]);
    }
  }
  
  //	 beginning of the inner loop.
  T ratio = (T)0.0; 
  do
  {
    //  determine the levenberg-marquardt parameter.
    mp_lmpar(fjac,ipvt,diag,qtf,delta,par,wa1,wa5,wa3,wa4);
    
    // store the direction p and x + p. calculate the norm of p.
    wa1 *= (T)-1.0;

    T alpha = 1.0;
    if (false == qanylim) 
    {
      // No parameter limits, so just move to new position WA5 
      for (auto &itr : fjac) 
      {
        data_struct::Fit_Param<T>& param = params.at(itr.first);
        wa5[j] = param.value + wa1[j];
      }

    }
    else
    {
      // Respect the limits.  If a step were to go out of bounds, then 
      // we should take a step in the same direction but shorter distance.
      // The step should take us right to the limit in that case.
      
      for (auto &itr : fjac) 
      {
        data_struct::Fit_Param<T>& param = params.at(itr.first);
        bool lpegged = false;
        bool upegged = false;
        if(param.bound_type == data_struct::Fit_Bound::LIMITED_LO || param.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
        {
          lpegged = (param.value <= param.min_val);
        }
        if(param.bound_type == data_struct::Fit_Bound::LIMITED_HI || param.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
        {
          upegged = (param.value >= param.max_val);
        }
        int dwa1 = fabs(wa1[j]) > MP_MachEp0<T>();
        
        if (lpegged && (wa1[j] < 0)) 
        {
          wa1[j] = 0;
        }
        if (upegged && (wa1[j] > 0))
        {
          wa1[j] = 0;
        }

        if(param.bound_type == data_struct::Fit_Bound::LIMITED_LO || param.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
        {
          if (dwa1 && ((param.value + wa1[j]) < param.min_val)) 
          {
            alpha = std::min(alpha, (param.min_val - param.value) / wa1[j]);
          }
        }
        if(param.bound_type == data_struct::Fit_Bound::LIMITED_HI || param.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
        {
          if (dwa1 && ((param.value + wa1[j]) > param.max_val)) 
          {
            alpha = std::min(alpha, (param.max_val - param.value) / wa1[j]);
          }
        }
      }
      
      // Scale the resulting vector, advance to the next position 
      for (auto &itr : fjac) 
      {
        data_struct::Fit_Param<T>& param = params.at(itr.first);
        T sgnu, sgnl;
        T ulim1, llim1;

        wa1[j] = wa1[j] * alpha;
        wa5[j] = param.value + wa1[j];

        // Adjust the output values.  If the step put us exactly
        // on a boundary, make sure it is exact.
        
        sgnu = (param.max_val >= 0) ? (+1) : (-1);
        sgnl = (param.min_val >= 0) ? (+1) : (-1);
        ulim1 = param.max_val * (1 - sgnu * MP_MachEp0<T>()) - ((param.max_val == 0)?(MP_MachEp0<T>()):0);
        llim1 = param.min_val * (1 + sgnl * MP_MachEp0<T>()) + ((param.min_val == 0)?(MP_MachEp0<T>()):0);

        if(param.bound_type == data_struct::Fit_Bound::LIMITED_HI || param.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
        {
          if (wa5[j] >= ulim1)
          {
            wa5[j] = param.max_val;
          }
        }
        if(param.bound_type == data_struct::Fit_Bound::LIMITED_LO || param.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI)
        {  
          if (wa5[j] <= llim1)
          {
            wa5[j] = param.min_val;
          }
        }
      }

    }

    wa3 = diag * wa1;    

    T pnorm = mp_enorm(wa3);
    
    //   on the first iteration, adjust the initial step bound.
    if (iter == 1) 
    {
      delta = std::min(delta, pnorm);
    }

    //   evaluate the function at x + p and calculate its norm.
    int iii = 0;
    for (auto &itr : fjac) 
    {
      saved_params.at(itr.first).value = wa5[iii];
      iii++;
    }

    FUNC_RET ret = funct(&saved_params, wa4, user_data);
    //iflag = mp_call(funct, m, npar, xnew, wa4, 0, private_data);
    result.nfev += 1;
    if (ret == FUNC_RET::USER_QUIT)
    {
      result.status = Errors::USER_QUIT;  
      return result;
    }


    fnorm1 = mp_enorm(wa4);

    //	    compute the scaled actual reduction.
    T actred = -(T)1.0;
    if (((T)0.1 * fnorm1) < fnorm)
    {
      temp = fnorm1 / fnorm;
      actred = (T)1.0 - temp * temp;
    }

    //   compute the scaled predicted reduction and the scaled directional derivative.
    j = 0;
    for (auto &itr : fjac) 
    {
      wa3[j] = (T)0.0;
      int l = ipvt[j];
      temp = wa1[l];
      for (int i=0; i<=j; i++ ) 
      {
        wa3[i] += itr.second[i]*temp; 
      }
      j++;
    }

    // Remember, alpha is the fraction of the full LM step actually taken
    

    T temp1 = mp_enorm(wa3) * alpha / fnorm;
    T temp2 = (sqrt(alpha * par) * pnorm) / fnorm;
    T prered = temp1 * temp1 + (temp2 * temp2) / (T)0.5;
    T dirder = -(temp1 * temp1 + temp2 * temp2);

    //  compute the ratio of the actual to the predicted reduction.
    if (prered != (T)0.0) 
    {
      ratio = actred/prered;
    }

    //	   update the step bound.
    
    
    if (ratio <= (T)0.25) 
    {
      if (actred >= (T)0.0) 
      {
        temp = (T)0.5; 
      }
      else
      {
        temp = (T)0.5*dirder/(dirder + (T)0.5*actred);
      }
      if ((((T)0.1 * fnorm1) >= fnorm) || (temp < (T)0.1) )
      {
        temp = (T)0.1;
      }
      delta = temp * std::min(delta,pnorm / (T)0.1);
      par = par/temp;
    }
    else 
    {
      if ((par == (T)0.0) || (ratio >= (T)0.75) ) 
      {
        delta = pnorm/(T)0.5;
        par = (T)0.5*par;
      }
    }

    //  test for successful iteration.
    if (ratio >= (T)1.0e-4)
    {
      
      //   successful iteration. update x, fvec, and their norms.
      j = 0;
      for (auto & itr : fjac) 
      {
        params.at(itr.first).value = wa5[j];
        wa5[j] = diag[j] * params.at(itr.first).value;
        j++;
      }
      out_resid = wa4;
      
      xnorm = mp_enorm(wa5);
      fnorm = fnorm1;
      iter += 1;
    }
    
    //   tests for convergence.

    if ((fabs(actred) <= options.ftol) && (prered <= options.ftol) &&  ((T)0.5*ratio <= (T)1.0) ) 
    {
      result.status = Errors::OK_CHI;
    }
    if (delta <= options.xtol*xnorm) 
    {
      result.status = Errors::OK_PAR;
    }
    if ((fabs(actred) <= options.ftol) && (prered <= options.ftol) && ((T)0.5 * ratio <= (T)1.0) && ( result.status == Errors::OK_PAR) ) 
    {
      result.status = Errors::OK_BOTH;
    }
    if (result.status == Errors::INPUT) 
    {   
      //  tests for termination and stringent tolerances.
      if ((options.maxfev > 0) && (result.nfev >= options.maxfev)) 
      {
        // Too many function evaluations 
        result.status = Errors::MAXITER;
      }
      if (iter >= options.maxiter) 
      {
        // Too many iterations 
        result.status = Errors::MAXITER;
      }
      if ((fabs(actred) <= MP_MachEp0<T>()) && (prered <= MP_MachEp0<T>()) && ((T)0.5*ratio <= (T)1.0) ) 
      {
        result.status = Errors::FTOL;
      }
      if (delta <= MP_MachEp0<T>() * xnorm) 
      {
        result.status = Errors::XTOL;
      }
      if (gnorm <= MP_MachEp0<T>()) 
      {
        result.status = Errors::GTOL;
      }
    }
  }  
  while (result.status == Errors::INPUT || ratio < (T)1.0e-4); // end of the inner loop. repeat if iteration unsuccessful.
  
  return result;
}


}
#endif