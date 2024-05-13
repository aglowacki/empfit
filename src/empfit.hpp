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
      ||  (param.side == data_struct::Derivative::AutoOneSide && (param.bound_type == data_struct::Fit_Bound::LIMITED_HI || param.bound_type == data_struct::Fit_Bound::LIMITED_LO_HI) && (temp > (param.max_val - h)))) 
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
void mp_qrfac(std::unordered_map<std::string, data_struct::ArrayTr<T> >& a, int pivot, data_struct::ArrayTr<int>& ipvt, data_struct::ArrayTr<T>&  rdiag, data_struct::ArrayTr<T>& acnorm, data_struct::ArrayTr<T>&  wa)
{
  T ajnorm,sum,temp;

  // compute the initial column norms and initialize several arrays.
  int j = 0;
  int m = 0;
  for(auto& itr: a)
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
  int n = a.size();
  int minmn = std::min(m,n);
  j = 0;
  for(auto& itr: a) 
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
Result<T> empfit(Callback_fuc<T> funct, size_t resid_size, data_struct::Fit_Parameters<T>& params, Options<T>* my_options, void *user_data)
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
/*
  // Make a new copy 
  for (i=0; i<npar; i++) {
    xnew[i] = xall[i];
  }

  // Initialize Levelberg-Marquardt parameter and iteration counter 
  par = 0.0;
  */
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
      for (int i=j; i<resid_size; i++ ) 
      {
        sum += itr.second[i] * wa4[i];
      }
      T temp = -sum / itr.second[j];
      for (int i=j; i<resid_size; i++ ) 
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
    int off = 0, nonfinite = 0;

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








  return result;
}


}
#endif