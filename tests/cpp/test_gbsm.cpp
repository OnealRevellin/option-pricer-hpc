#include <gtest/gtest.h>
#include "../../cpp/vanilla_options_pricer.h"
#include <cmath>

// Tolerance level used for floating point comparisons.
const double TOL = 1e-9;

// Simple Call & Put value check using known values of options.
TEST(GBSMTEST, CallOptionSimpleCase)
{
    std::vector<uint8_t> is_call = {1};
    std::vector<double> S = {100.0};
    std::vector<double> K = {100.0};
    std::vector<double> T = {1.0};
    std::vector<double> r = {0.05};
    std::vector<double> sigma = {0.2};
    std::vector<double> b = {0.05};

    VanillaOptionsPricer p = VanillaOptionsPricer(is_call, S, K, T, r, sigma, b);
    std::vector<double> result = p.values();

    double expected_value = 10.4506;
    EXPECT_NEAR(result[0], expected_value, 1e-3);
}

TEST(GBSMTEST, PutOptionSimpleCase)
{
    std::vector<uint8_t> is_call = {0};
    std::vector<double> S = {100.0};
    std::vector<double> K = {100.0};
    std::vector<double> T = {1.0};
    std::vector<double> r = {0.05};
    std::vector<double> sigma = {0.2};
    std::vector<double> b = {0.05};

    VanillaOptionsPricer p = VanillaOptionsPricer(is_call, S, K, T, r, sigma, b);
    std::vector<double> result = p.values();

    double expected_value = 5.5735;
    EXPECT_NEAR(result[0], expected_value, 1e-3);
}

// Call-Put Parity check: C + K * exp(-rT) = P + S
TEST(GBSMTEST, CallPutParity)
{
    std::vector<uint8_t> is_call = {1, 0};
    std::vector<double> S = {100.0, 100.0};
    std::vector<double> K = {100.0, 100.0};
    std::vector<double> T = {1.0, 1.0};
    std::vector<double> r = {0.05, 0.05};
    std::vector<double> sigma = {0.2, 0.2};
    std::vector<double> b = {0.05, 0.05};

    VanillaOptionsPricer p(is_call, S, K, T, r, sigma, b);
    std::vector<double> result = p.values();

    double call_price = result[0];
    double put_price = result[1];
    double lhs = call_price + K[0] * std::exp(-r[0] * T[0]);
    double rhs = put_price + S[0];

    EXPECT_NEAR(lhs, rhs, 1e-6);
}

// Sigma = 0 (no vol until expiry), call option should be discounted intrinsic value
TEST(GBSMTEST, SigmaZeroCall)
{
    std::vector<uint8_t> is_call = {1};
    std::vector<double> S = {120.0};
    std::vector<double> K = {100.0};
    std::vector<double> T = {1.0};
    std::vector<double> r = {0.05};
    std::vector<double> sigma = {0.0};
    std::vector<double> b = {0.05};

    VanillaOptionsPricer p(is_call, S, K, T, r, sigma, b);
    std::vector<double> result = p.values();

    double expected_value = std::max(
        S[0] * std::exp((b[0] - r[0]) * T[0]) - K[0] * std::exp(-r[0] * T[0]), 
        0.0
    );
    
    EXPECT_NEAR(result[0], expected_value, TOL);
}

// Expiry case (T=0), option values should be equal to intrinsic values
TEST(GBSMTEST, ExpiryIntrinsicValue)
{
    std::vector<uint8_t> is_call = {1, 0};
    std::vector<double> S = {90.0, 90.0};
    std::vector<double> K = {100.0, 100.0};
    std::vector<double> T = {0.0, 0.0};
    std::vector<double> r = {0.05, 0.05};
    std::vector<double> sigma = {0.2, 0.2};
    std::vector<double> b = {0.05, 0.05};

    VanillaOptionsPricer p(is_call, S, K, T, r, sigma, b);
    std::vector<double> result = p.values();

    double expected_call = std::max(S[0] - K[0], 0.0);
    double expected_put = std::max(K[1] - S[1], 0.0);

    EXPECT_NEAR(result[0], expected_call, TOL);
    EXPECT_NEAR(result[1], expected_put, TOL);
}