#pragma once

#include <vector>
#include <cstdint>

class VanillaOptionsPricer
{
public:
    VanillaOptionsPricer(
        const std::vector<uint8_t>& is_call_,
        const std::vector<double>& S_,
        const std::vector<double>& K_,
        const std::vector<double>& T_,
        const std::vector<double>& r_,
        const std::vector<double>& sigma_,
        const std::vector<double>& b_
    );

    std::vector<double> values();

private:
    std::vector<uint8_t> is_call;
    std::vector<double> S, K, T, r, sigma, b;
};