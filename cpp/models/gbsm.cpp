#include "gbsm.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <omp.h>


inline double fast_norm_cdf(double x) {
    // Abramowitz & Stegun approximation
    static constexpr double a1 = 0.319381530;
    static constexpr double a2 = -0.356563782;
    static constexpr double a3 = 1.781477937;
    static constexpr double a4 = -1.821255978;
    static constexpr double a5 = 1.330274429;
    static constexpr double inv_sqrt_2pi = 0.3989422804014327;
    
    double L = std::fabs(x);
    double k = 1.0 / (1.0 + 0.2316419 * L);
    double w = 1.0 - inv_sqrt_2pi * std::exp(-L * L / 2) *
                    (a1 * k + a2 * k*k + a3 * std::pow(k,3) + a4 * std::pow(k,4) + a5 * std::pow(k,5));
    return (x < 0.0) ? 1.0 - w : w;
}


inline double norm_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}


std::vector<double> gbsm_value(
    const std::vector<uint8_t>& is_call,
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& T,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& b
)
{
    size_t N = S.size();
    if (is_call.size() != N || K.size() != N || T.size() != N || r.size() != N || sigma.size() != N || b.size() != N)
        throw std::invalid_argument("All input vectors must be the same length.");
        
    std::vector<double> values(S.size());
    double sqrtT, d1, d2;
    double ebrT, erT;

    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        double intrinsic = is_call[i] ? std::max(S[i] - K[i], 0.0) : std::max(K[i] - S[i], 0.0);

        sqrtT = sqrt(T[i]);
        ebrT = std::exp((b[i] - r[i]) * T[i]);
        erT  = std::exp(-r[i] * T[i]);

        d1 = (std::log(S[i] / K[i]) + (b[i] + 0.5 * std::pow(sigma[i], 2)) * T[i]) / (sigma[i] * sqrtT);
        d2 = d1 - sigma[i] * sqrtT;

        double val = (
            is_call[i] ? 
            S[i] * ebrT * fast_norm_cdf(d1) - K[i] * erT * fast_norm_cdf(d2)
            : K[i] * erT * fast_norm_cdf(-d2) - S[i] * ebrT * fast_norm_cdf(-d1)
        );

        bool expired = (T[i] <= 0.0);
        values[i] = expired ? intrinsic : val;
    }

    return values;
}



