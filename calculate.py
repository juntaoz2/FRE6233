import numpy as np
import statistics
import scipy
from scipy.stats import norm
def MC_model(S0, n, m, r, sigma, T, K):
    Sari = Sgeo = S0
    C1, C2 = [0 for _ in range(n)], [0 for _ in range(n)]
    S = [0 for _ in range(m)]
    S[0] = S0
    temp_expectation = [0 for _ in range(n)]

    for i in range(1, n):
        for j in range(1, m):
            Z = np.random.normal(0, 1)
            S[j] = S[j-1]*np.exp((r-0.5*sigma**2)*(1/m) + sigma*np.sqrt(1/m)*Z)
        Sari_mean = np.mean(S)
        Sgeo_mean = statistics.geometric_mean(S)

        C1[i] = np.exp(-r*T) * max(Sari_mean-K, 0) #C1 is arithmetic Y calculate sample variance
        C2[i] = np.exp(-r * T) * max(Sgeo_mean - K, 0) #C2 is geometric X
    C1_avg = sum(C1) / n
    C2_avg = sum(C2) / n

    C_1 = C_2 = 0

    for i in range(1, n):
        C_1 += (C1[i] - C1_avg) * (C2[i] - C2_avg)  #C1 = Y
        C_2 += (C2[i] - C2_avg) * (C2[i] - C2_avg)  #C2 = X
    b = C_1/C_2

    tbar = (1/m) * sum([i for i in np.arange(1/12,1+1/12,1/12)])

    temp = 0
    for i in range(1, m):
        temp += (2*i-1)*((m+1-i)/m)

    sigma_sq = ((sigma**2)/(m**2 * tbar)) * temp
    delta = 0.5*(sigma**2) - 0.5*(sigma_sq**2)

    d = (np.log(S0/K) + (r-delta+0.5*sigma_sq)*tbar)/(np.sqrt(sigma_sq*tbar))
    expectaton = (np.exp(-delta*tbar) * S0 * norm.cdf(d)) - np.exp(-r*tbar) * K * norm.cdf(d-np.sqrt(sigma_sq*tbar))

    for i in range(1, n):
        temp_expectation[i] = C1[i] - b*(C2[i]-expectaton)
    Ybar = sum(temp_expectation) / n

    #covariance and rho
    correlation = np.cov(C1, C2)[0,1]/ (np.var(C1)*np.var(C2))

    #calculating standard error
    var = statistics.variance(C1)

    return (Ybar, correlation, var)

res = {}
ns = [1000, 10000, 100000, 1000000]

for ele in ns:
    res[str(ele)] = MC_model(S0 = 110, n = ele, m = 12, r = 0.01, sigma = 0.3, T = 1, K = 100)
print(res)
