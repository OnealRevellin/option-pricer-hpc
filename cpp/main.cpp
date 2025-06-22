#include "vanilla_options_pricer.h"
#include "models/gbsm_simd.h"

#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

int main()
{
    omp_set_num_threads(omp_get_max_threads());

    // Number of options to price during the simulation.
    const size_t N = 50'000'000;
    // Inputs generator.
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> spot_dist(50, 150);
    std::uniform_real_distribution<> strike_dist(50, 150);
    std::uniform_real_distribution<> time_dist(0.01, 2.0);
    std::uniform_real_distribution<> rate_dist(0.0, 0.1);
    std::uniform_real_distribution<> vol_dist(0.1, 0.5);
    std::uniform_real_distribution<> carry_dist(-0.05, 0.05);
    std::bernoulli_distribution call_put_dist(0.5);

    // Generate N inputs to init the options pricer.
    std::vector<uint8_t> is_call(N);
    std::vector<double> S(N), K(N), T(N), r(N), sigma(N), b(N);

    for (size_t i = 0; i < N; ++i) {
        is_call[i] = call_put_dist(gen);
        S[i] = spot_dist(gen);
        K[i] = strike_dist(gen);
        T[i] = time_dist(gen);
        r[i] = rate_dist(gen);
        sigma[i] = vol_dist(gen);
        b[i] = carry_dist(gen);
    }

    VanillaOptionsPricer pricer = VanillaOptionsPricer(is_call, S, K, T, r, sigma, b);
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> result = pricer.values();
    //std::vector<double> result = gbsm_value_simd(is_call, S, K, T, r, sigma, b);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    
    std::cout << "Execution time: " << duration << " nanoseconds" << "\n";
    std::cout << "Execution time per option: " << (duration/N) << " nanoseconds" << "\n";
    
    
    for (int i = 0; i < 5; ++i) {
        std::cout << "Option " << i << ": Value = " << result[i] << "\n";
        std::cout << "Params : [" << static_cast<int>(is_call[i]) << ", " << S[i] << ", " << K[i] << ", " << T[i] << ", " << sigma[i] << ", " << r[i] << "]" << std::endl;
    }
    std::cout << "Option " << N << ": Value = " << result[N-1] << "\n";

    return 0;

}


//g++ -O3 -march=native -ffast-math -funroll-loops -fopenmp -std=c++20 main.cpp "vanilla_options_pricer.cpp" "models/gbsm.cpp" -o main