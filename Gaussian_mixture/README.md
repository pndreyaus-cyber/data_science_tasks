# Gaussian Mixture Model (EM from Scratch)

- Implemented EM algorithm for 1D Gaussian mixtures
- Supports multiple initialization strategies:
  - Random
  - Percentile-based
  - K-means-based
- Compared against `sklearn.mixture.GaussianMixture`

# Results
The original generated mixture is

$$ p(x) = 0.3 \, \mathcal{N}(x \mid -2, 0.16) + 0.5 \, \mathcal{N}(x \mid 0, 1.0) + 0.2 \, \mathcal{N}(x \mid 3, 0.09) $$

The test ran EM-algorithm with 3 different intialisation algorithms: random, k_means and percentile.

Random initialisation has these properties:
    - Means: Random points from data range or random data points
    - Weights: All equal (1/k)
    - Variances: All equal to overall data variance

Percentile-based initialisation:
    - Means: At percentiles of data (e.g., 25th, 50th, 75th)
    - Weights: Equal
    - Variances: Calculated from data around each percentile
  
K-means clustering
    - Use K-means cluster centers as initial means
    - Estimate weights from cluster sizes
    - Estimate variances from clusters

And also it compared EM-algorithm results with the include in the sklearn.mixture library GaussianMixture class. the results are show in the table below. 

| Source | Weight | Mean | Std |
|---|---|---|---|
| | | Left | | |
| sklearn | 0.370 | -1.91 | 0.47 |
| rand | 0.386 | -1.876 | 0.495 |
| kmeans | 0.316 | -1.996 | 0.407 |
| percentile | 0.346 | -1.953 | 0.438 |
| | | Middle | | |
| sklearn | 0.426 | 0.077 | 0.806 |
| rand | 0.409 | 0.123 | 0.778 |
| kmeans | 0.481 | -0.079 | 0.915 |
| percentile | 0.451 | 0.0074 | 0.853 |
| | | Right | | |
| sklearn | 0.204 | 3.008 | 0.293 |
| rand | 0.204 | 3.0072 | 0.294 |
| kmeans | 0.203 | 3.011 | 0.29 |
| percentile | 0.203 | 3.009 | 0.292 |

| Source | Log-likelihood |
| ---    | --- |
| sklearn | -1725.5 |
| rand | -1727.55 |
| kmeans | -1721.478 |
| percentile | -1722.853 |

## Conclusion
Initialisation matters. Random intialisation gives the worst value of log-likelihood. K-means gives the best log-likelihood value (it is even better, than the sklearn -- in this test)

Estimated parameters using EM-algorithm are very close to the original ones and to the sklearn estimated parameters.
