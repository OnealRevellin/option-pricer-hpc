// gbsm.h
#ifndef GBSM_H
#define GBSM_H

#include <vector>
#include <cstdint>

inline double norm_cdf(double);
inline double fast_norm_cdf(double);


/**
 * @brief Computes the value of a European option using GBSM model.
 * 
 * Using this model, `b` can be adapted so the model matches the underlying type.
 * For example:
 * - `b = r`      ⇒ Black-Scholes model for stock options.
 * - `b = 0.0`    ⇒ Black76 model for options on futures.
 * - `b = r - q`  ⇒ Black-Scholes-Merton model for stock options with continuous dividend yield.
 *
 * @param is_call Vector indicating call (1) or put (0) options.
 * @param S Vector of spot prices.
 * @param K Vector of strike prices.
 * @param T Vector of times to maturity in years.
 * @param r Vector of risk-free interest rates.
 * @param sigma Vector of annualized volatilities.
 * @param b Vector of cost of carry values.
 * @return std::vector<double> Vector of option prices.
 * @throws std::invalid_argument if input vectors have different sizes.
 */
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