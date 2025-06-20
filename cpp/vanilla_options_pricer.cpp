#include "vanilla_options_pricer.h"
#include "models/gbsm.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>



VanillaOptionsPricer::VanillaOptionsPricer(
    const std::vector<uint8_t> &is_call_,
    const std::vector<double> &S_,
    const std::vector<double> &K_,
    const std::vector<double> &T_,
    const std::vector<double> &r_,
    const std::vector<double> &sigma_,
    const std::vector<double> &b_
)
    : is_call(is_call_), S(S_), K(K_), T(T_), r(r_), sigma(sigma_), b(b_) {}

std::vector<double> values();


std::vector<double> VanillaOptionsPricer::values()
{
    return gbsm_value(
        is_call, S, K, T, r, sigma, b
    );
}

