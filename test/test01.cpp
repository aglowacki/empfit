#include <iostream>

#include "empfit.hpp"
/*
using FVec = data_struct::ArrayTr<float>;
using FParams = data_struct::Fit_Parameters<float>;
using FResult = optimize::Result<float>;
*/

using DArr = data_struct::ArrayTr<double>;
using DParam = data_struct::Fit_Param<double>;
using DParams = data_struct::Fit_Parameters<double>;
using DResult = optimize::Result<double>;

// This is the private data structure which contains the data points and their uncertainties 
struct vars_struct
{
    vars_struct(DArr *_x, DArr* _y, DArr* _ey)
    {
        x = _x;
        y = _y;
        ey = _ey;
    }
    
    DArr* x;
    DArr* y;
    DArr* ey;
};

optimize::FUNC_RET linfunc(const DParams * const params, DArr& out, void* user_data)
{
    struct vars_struct *v = (struct vars_struct *) user_data;
    double a = params->at("a").value; 
    double b = params->at("b").value; 
    DArr f = a +  b *  *(v->x) ;
    out = ( *(v->y) - f ) / *(v->ey);

    return optimize::FUNC_RET::OK;
}

void test01()
{
    //std::function<void(const Fit_Parameters<T_real>* const, const  Range* const, Spectra<T_real>*)> gen_func = std::bind(&Matrix_Optimized_Fit_Routine<T_real>::model_spectrum, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

    DArr x(10);
    x<< -1.7237128E+00,1.8712276E+00,-9.6608055E-01,
        -2.8394297E-01,1.3416969E+00,1.3757038E+00,
        -1.3703436E+00,4.2581975E-02,-1.4970151E-01,
        8.2065094E-01;
    DArr y(10);
    y<< 1.9000429E-01,6.5807428E+00,1.4582725E+00,
        2.7270851E+00,5.5969253E+00,5.6249280E+00,
        0.787615,3.2599759E+00,2.9771762E+00,
        4.5936475E+00;

    DArr ey(10);
    ey<< .07,.07,.07,.07,.07,.07,.07,.07,.07,.07;
    /*      y = a - b*x    */
    /*              a    b */
    DParams params = { {"a", DParam("a", 1.0, data_struct::Fit_Bound::FIT)},
     {"b", DParam("b", 1.0, data_struct::Fit_Bound::FIT)} };

    struct vars_struct v(&x, &y, &ey);

    // Call fitting function for 10 data points and 2 parameters 
    DResult result = optimize::empfit<double>(linfunc, x.size(), params, nullptr, (void *) &v);

    std::cout<<"*** testlinfit status = "<<optimize::ErrorToString(result.status) <<"\n";
    std::cout<< "a = "<< params.at("a").value << " : Actual = 3.20\n";
    std::cout<< "b = "<< params.at("b").value << " : Actual = 1.78\n";
}

int main()
{
    test01();

    return 0;
}