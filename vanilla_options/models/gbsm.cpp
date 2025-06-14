#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
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


int main()
{
    // Set threads to the max number of threads of the CPU for OMP
    omp_set_num_threads(omp_get_max_threads());

    const size_t N = 50'000'000;

    // inputs generator
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<> spot_dist(50, 150);
    uniform_real_distribution<> strike_dist(50, 150);
    uniform_real_distribution<> time_dist(0.01, 2.0);
    uniform_real_distribution<> rate_dist(0.0, 0.1);
    uniform_real_distribution<> vol_dist(0.1, 0.5);
    uniform_real_distribution<> carry_dist(-0.05, 0.05);
    bernoulli_distribution call_put_dist(0.5);

    // Generate random inputs
    vector<bool> is_call(N);
    vector<double> S(N), K(N), T(N), r(N), sigma(N), b(N);

    for (size_t i = 0; i < N; ++i) {
        is_call[i] = call_put_dist(gen);
        S[i] = spot_dist(gen);
        K[i] = strike_dist(gen);
        T[i] = time_dist(gen);
        r[i] = rate_dist(gen);
        sigma[i] = vol_dist(gen);
        b[i] = carry_dist(gen);
    }

    auto start = chrono::high_resolution_clock::now();
    vector<double> result = gbsm_value(is_call, S, K, T, r, sigma, b);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    cout << "Execution time: " << duration << " nanoseconds" << "\n";
    cout << "Execution time per option: " << (duration/N) << " nanoseconds" << "\n";

    for (int i = 0; i < 5; ++i) {
        cout << "Option " << i << ": Value = " << result[i] << "\n";
    }

    cout << "Option " << N << ": Value = " << result[N-1] << "\n";

    return 0;

}

//g++ -O3 -march=native -ffast-math -funroll-loops -fopenmp -std=c++20 gbsm.cpp -o gbsm