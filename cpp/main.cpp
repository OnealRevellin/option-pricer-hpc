#include "models/gbsm.h"

#include <iostream>
#include <random>
#include <chrono>

int main()
{
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