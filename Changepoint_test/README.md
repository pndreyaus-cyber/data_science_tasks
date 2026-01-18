# Testing if the sample has a breakpoint using empirical bridges

This task is to test the homogeneity of samples generated from two different distributions. 

## Purpose
This project implements and evaluates a statistical test for detecting structural breaks in time series data. The test is based on the **Empirical Bridge Process** — the normalized cumulative sum of residuals — whose asymptotic behavior under the null hypothesis follows a Brownian bridge.

## Problem Statement
Given a sample with a potential change-point at position 110 (110 observations from cosine distribution, 90 from hypsecant distribution), we:
1. Implement the empirical bridge test statistic
2. Calculate its asymptotic p-value
3. Estimate empirical power through Monte Carlo simulation
4. Analyze the effect of data transformations (x², x⁴) on test performance



## Results
### Original sample
![Original sample plot](img/sample_1.png "Original sample")

![Original sample empirical bridge](img/emp_bridge_1.png "Original sample empirical bridge")

![Calculated results for original sample](img/results_1.png "Calculated results for original sample")

### Sqaured sample
![Squared sample plot](img/sample_2.png "Squared sample")

![Squared sample empirical bridge](img/emp_bridge_2.png "Squared sample empirical bridge")

![Calculated results for squared sample](img/results_2.png "Calculated results for sqaured sample")

### Sample to the 4-th power
![Sample to the 4-th power plot](img/sample_4.png "Sample to the 4-th power")

![Sample to the 4-th power empirical bridge](img/emp_bridge_4.png "Sample to the 4-th power empirical bridge")

![Calculated results for sample to the 4-th power](img/results_4.png "Calculated results for sample to the 4-th power")

### Power, mean difference and variance difference
| Transformation	| Power |	Mean Difference |	Variance Difference |
| --- | --- | --- | --- |
| x¹ (original) |	0.043	| -0.0015 |	-1.1808 |
|x² (squared) |	0.545	| -1.1913 |	-21.7468 |
|x⁴ (4th power)	| 0.346 |	-27.5467 |	-50400.93 |

#### 1. Original data ($x$)


*   Power ≈ 0.043. Very low probability of rejecting $H_{0}$, when $H_{1}$ is true
*   Mean difference ≈ 0 — Cosine and hypsecant distributions have essentially identical means. That is probably why, $H_{0}$ would be accepted
*   Moderate variance difference (-1.1808) — but the test is insensitive to variance changes in raw data

The test falsely accepted $H_{0}$

#### 2. Squared data ($x^2$)
* Power ≈ 0.545. More than 50% of correctly rejecting $H_{0}$!!!
* Mean difference = -1.191 — Mean shift is detectible
* Large variance difference (-21.74) amplifies the effect after transformation

The test correctly rejected $H_{0}$

#### 3. Data powered to the 4-th power ($x^4$)
* Power ≈ 0.346. Less than 35% of correctly rejecting $H_{0}$
* Mean difference = -27.54 — Massive theoretical difference
* Extreme variance difference (-50400) — indicates heavy-tailed, unstable data
* Strange: despite larger mean difference than x², power is lower

The test falsely accepted $H_{0}$

### Conclusion
The squared transformation achieves the optimal balance for this test
