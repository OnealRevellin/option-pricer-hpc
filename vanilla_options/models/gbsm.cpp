#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> gbsm_value(
    vector<bool> &is_call,
    vector<double> &S,
    vector<double> &K,
    vector<double> &T,
    vector<double> &r,
    vector<double> &sigma,
    vector<double> &b
)
{
    vector<double> values(S.size());
    for (size_t i = 0; i < S.size(); ++i)
    {
        if (T[i] <= 0.0)
        {
            values[i] = is_call[i] ? max(S[i] - K[i], 0.0) : max(K[i] - S[i], 0.0);
        }
    }

    return values;
}

int main()
{

    return 0;
}