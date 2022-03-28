import numpy as np
import statistics

def PathDepS(S_0, n, r, sigma, m, K, T):
    S = [[0 for _ in range(m)] for _ in range(n)]
    t = [i for i in np.arange(0,T, 1/m)]
    S_avg = []
    C_avg = []
    C = []
    for i in range(n):
        S[i][0] = S_0
        for j in range(1, m):
            Z = np.random.normal(0, 1)
            S[i][j] = S[i][j-1]*np.exp((r-0.5*sigma**2)*(1/m) + sigma*np.sqrt(1/m)*Z)
        avg = sum(S[i])/m
        S_avg.append(avg)
        C_avg.append(np.exp(-r*T)*max(avg - K, 0))
    C = sum(C_avg) / n
    var = statistics.variance(C_avg)
    return (C, var)

res = {}
ns = [1000, 10000, 100000, 1000000]

for ele in ns:
    res[str(ele)] = PathDepS(S_0 = 110, n = ele, r = 0.01, sigma = 0.3, m = 12, K = 100, T = 1)
print(res)
