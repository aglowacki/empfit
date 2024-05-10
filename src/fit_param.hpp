/***
-Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.
-
-Copyright 2024. UChicago Argonne, LLC. This software was produced
-under U.S. Government contract DE-AC02-06CH11357 for Argonne National
-Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
-U.S. Department of Energy. The U.S. Government has rights to use,
-reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
-UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
-ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
-modified to produce derivative works, such modified software should
-be clearly marked, so as not to confuse it with the version available
-from ANL.
-
-Additionally, redistribution and use in source and binary forms, with
-or without modification, are permitted provided that the following
-conditions are met:
-
-    * Redistributions of source code must retain the above copyright
-      notice, this list of conditions and the following disclaimer.
-
-    * Redistributions in binary form must reproduce the above copyright
-      notice, this list of conditions and the following disclaimer in
-      the documentation and/or other materials provided with the
-      distribution.
-
-    * Neither the name of UChicago Argonne, LLC, Argonne National
-      Laboratory, ANL, the U.S. Government, nor the names of its
-      contributors may be used to endorse or promote products derived
-      from this software without specific prior written permission.
-
-THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
-"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
-LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
-FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
-Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
-INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
-BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
-CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
-LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
-ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
-POSSIBILITY OF SUCH DAMAGE.
-***/

#ifndef Fit_Param_HPP
#define Fit_Param_HPP

#include <string>
#include <unordered_map>

#if defined _WIN32 || defined __CYGWIN__
  #pragma warning( disable : 4251 4127 4996 4505 4244 )
  #define DIR_END_CHAR '\\'
  #define DIR_END_CHAR_OPPOSITE '/'
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define DLL_EXPORT __attribute__ ((dllexport))
    #else
      #define DLL_EXPORT __declspec(dllexport) 
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_EXPORT __attribute__ ((dllimport))
    #else
      #define DLL_EXPORT __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define DLL_LOCAL
#else
  #define DIR_END_CHAR '/'
  #define DIR_END_CHAR_OPPOSITE '\\'
  #if __GNUC__ >= 4
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DLL_EXPORT
    #define DLL_LOCAL
  #endif
#endif


namespace data_struct
{

//-----------------------------------------------------------------------------

enum class Fit_Bound {FIXED=1, LIMITED_LO_HI=2, LIMITED_LO=3, LIMITED_HI=4, FIT=5};

enum class Derivative {AutoOneSide=0, OneSide=1, NegOneSide=-1, TwoSide=2, User=3};
    // 0 - one-sided derivative computed automatically
	// 1 - one-sided derivative (f(x+h) - f(x)  )/h
    // -1 - one-sided derivative (f(x)   - f(x-h))/h
	// 2 - two-sided derivative (f(x+h) - f(x-h))/(2*h) 
	// 3 - user-computed analytical derivatives}
//-----------------------------------------------------------------------------

//template<typename T>
//using VectorTr = Eigen::Vector<T, Eigen::Dynamic>;

template<typename _T>
using ArrayTr = Eigen::Array<_T, Eigen::Dynamic, Eigen::RowMajor>;

//-----------------------------------------------------------------------------

/**
 * @brief The Fit_Param struct : Structure that holds a parameter which consists of a value, min, max, and if it should be used in the fit routine.
 *                                Many fit routines use arrays so there are convert to and from array functions.
 */
template<typename T>
struct DLL_EXPORT Fit_Param
{
    Fit_Param()
    {
        name = "N/A";
        min_val = std::numeric_limits<T>::quiet_NaN();
        max_val = std::numeric_limits<T>::quiet_NaN();
        value = std::numeric_limits<T>::quiet_NaN();
        step_size = std::numeric_limits<T>::quiet_NaN();
        relstep = std::numeric_limits<T>::quiet_NaN();
        side = Derivative::AutoOneSide;
        deriv_reltol = std::numeric_limits<T>::quiet_NaN();
        deriv_abstol = std::numeric_limits<T>::quiet_NaN();
        bound_type = Fit_Bound::FIXED;
        debug = false;
    }

    Fit_Param(const Fit_Param& param)
    {
        name = param.name;
        min_val = param.min_val;
        max_val = param.max_val;
        value = param.value;
        step_size = param.step_size;
        relstep = param.relstep;
        deriv_reltol = param.deriv_reltol;
        deriv_abstol = param.deriv_abstol;
        side = param.side;
        bound_type = param.bound_type;
        debug = param.debug;
    }
    /*
    Fit_Param(const Fit_Param&& param)
    {
        name = param.name;
        min_val = param.min_val;
        max_val = param.max_val;
        value = param.value;
        step_size = param.step_size;
        relstep = param.relstep;
        side = param.side;
        bound_type = param.bound_type;
    }
    */
    Fit_Param(std::string name_)
    {
        name = name_;
        min_val = std::numeric_limits<T>::quiet_NaN();
        max_val = std::numeric_limits<T>::quiet_NaN();
        value = std::numeric_limits<T>::quiet_NaN();
        step_size = std::numeric_limits<T>::quiet_NaN();
        relstep = std::numeric_limits<T>::quiet_NaN();
        deriv_reltol = std::numeric_limits<T>::quiet_NaN();
        deriv_abstol = std::numeric_limits<T>::quiet_NaN();
        side = Derivative::AutoOneSide;
        bound_type = Fit_Bound::FIXED;
        debug = false;
    }

    Fit_Param(std::string name_, T val_)
    {
        name = name_;
        min_val = std::numeric_limits<T>::min();
        max_val = std::numeric_limits<T>::max();
        step_size = val_ *(T)0.0001;
        relstep = val_ * (T)0.00001;
        value = val_;
        deriv_reltol = (T)0.0;
        deriv_abstol = (T)0.0;
        side = Derivative::AutoOneSide;
        bound_type = Fit_Bound::FIXED;
        debug = false;
    }

	Fit_Param(std::string name_, T val_, Fit_Bound b_type)
	{
		name = name_;
		min_val = std::numeric_limits<T>::min();
		max_val = std::numeric_limits<T>::max();
		step_size = val_ *(T)0.0001;
        relstep = val_ * (T)0.00001;
        deriv_reltol = std::numeric_limits<T>::quiet_NaN();
        deriv_abstol = std::numeric_limits<T>::quiet_NaN();
		value = val_;
        side = Derivative::AutoOneSide;
		bound_type = b_type;
        debug = false;
	}

    Fit_Param(std::string name_, T min_, T max_, T val_, T step_size_, T relstep_, Derivative side_, Fit_Bound b_type)
    {
        name = name_;
        min_val = min_;
        max_val = max_;
        value = val_;
        relstep = relstep_;
        bound_type = b_type;
        side = side_;
        step_size = step_size_;
        debug = false;
    }

    const std::string bound_type_str() const
    {
        switch (bound_type)
        {
            case Fit_Bound::FIXED:
                return "Fixed";
                break;
            case Fit_Bound::LIMITED_LO_HI:
                return "LIMITED LO HI";
                break;
            case Fit_Bound::LIMITED_LO:
                return "LIMITED LO";
                break;
            case Fit_Bound::LIMITED_HI:
                return "LIMITED HI";
                break;
            case Fit_Bound::FIT:
                return "FIT";
                break;
        }
        return "N/A";
    }

    std::string name;
    T min_val;
    T max_val;
    T value;
    T step_size;
    T relstep;
    T deriv_reltol;
    T deriv_abstol;
    Derivative side; 
    Fit_Bound bound_type;
    bool debug;
};


//-----------------------------------------------------------------------------

template<typename T>
using Fit_Parameters = std::unordered_map<std::string, Fit_Param<T> >;

//-----------------------------------------------------------------------------

} //namespace data_struct

#endif // Fit_Param_Hpp
