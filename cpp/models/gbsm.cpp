#include "gbsm.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;


inline double norm_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}


vector<double> gbsm_value(
    const vector<bool> &is_call,
    const vector<double> &S,
    const vector<double> &K,
    const vector<double> &T,
    const vector<double> &r,
    const vector<double> &sigma,
    const vector<double> &b
)
{
    omp_set_num_threads(omp_get_max_threads());

    vector<double> values(S.size());
    double sqrtT, d1, d2;

    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(S.size()); ++i)
    {
        if (T[i] <= 0.0)
        {
            values[i] = is_call[i] ? max(S[i] - K[i], 0.0) : max(K[i] - S[i], 0.0);
            continue;
        }
        sqrtT = sqrt(T[i]);

        d1 = (log(S[i] / K[i]) + (b[i] + pow(sigma[i], 2) * T[i]) ) / (sigma[i] * sqrtT);
        d2 = d1 - sigma[i] * sqrtT;

        values[i] = (
            is_call[i] ? 
            S[i] * exp((b[i] - r[i]) * T[i]) * norm_cdf(d1) - K[i] * exp(-r[i] * T[i]) * norm_cdf(d2)
            : K[i] * exp(-r[i] * T[i]) * norm_cdf(-d2) - S[i] * exp((b[i] - r[i]) * T[i]) * norm_cdf(-d1)
        );
    }

    return values;
}



//g++ -O3 -march=native -ffast-math -funroll-loops -fopenmp -std=c++20 gbsm.cpp -o gbsm