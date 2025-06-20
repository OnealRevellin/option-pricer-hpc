// gbsm.h
#ifndef GBSM_H
#define GBSM_H

#include <vector>
#include <cstdint>

inline double norm_cdf(double);
inline double fast_norm_cdf(double);

std::vector<double> gbsm_value(
    const std::vector<uint8_t>& is_call,
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& T,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& b
);

#endif // GBSM_H