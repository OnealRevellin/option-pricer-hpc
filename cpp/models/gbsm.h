// gbsm.h
#ifndef GBSM_H
#define GBSM_H

#include <vector>
using namespace std;

inline double norm_cdf(double);
vector<double> gbsm_value(
    const vector<bool> &is_call,
    const vector<double> &S,
    const vector<double> &K,
    const vector<double> &T,
    const vector<double> &r,
    const vector<double> &sigma,
    const vector<double> &b
);

#endif // GBSM_H