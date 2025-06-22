#pragma once

#include <cmath>
#include <algorithm>
#include <immintrin.h>

inline double norm_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

inline double fast_norm_cdf(double x) {
    // Abramowitz & Stegun approximation
    static constexpr double a1 = 0.319381530, a2 = -0.356563782, a3 = 1.781477937;
    static constexpr double a4 = -1.821255978, a5 = 1.330274429, inv_sqrt_2pi = 0.3989422804014327;
    
    double L = std::fabs(x);
    double k = 1.0 / (1.0 + 0.2316419 * L);
    double poly = ((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k;
    double w = 1.0 - inv_sqrt_2pi * std::exp(-L * L / 2) * poly;
    return (x < 0.0) ? 1.0 - w : w;
}

