#include "gbsm.h"
#include "../maths/stats.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <omp.h>

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
        double sigma_floor = sigma[i] <= 0.0 ? 1e-9 : sigma[i];

        sqrtT = sqrt(T[i]);
        ebrT = std::exp((b[i] - r[i]) * T[i]);
        erT  = std::exp(-r[i] * T[i]);

        d1 = (std::log(S[i] / K[i]) + (b[i] + 0.5 * std::pow(sigma_floor, 2)) * T[i]) / (sigma_floor * sqrtT);
        d2 = d1 - sigma_floor * sqrtT;

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



