# Fuzzy clustering that finds the number of clusters on its own

**Master's thesis** · *Implementation and Assessment of Cluster Size-Adaptable Fuzzy Clustering Algorithms with Cluster Merging Capabilities*
Mireia Jaume Roldán · MSc in Intelligent Systems (MUSI), University of the Balearic Islands, 2025/26 · Supervisor: Alberto Ortiz Rodríguez

📄 **[Read the thesis (PDF)](paper_master_thesis.pdf)**

---

## In short

Most clustering algorithms need to be told how many clusters to look for. In practice that number is rarely known. This project tackles the problem for a family of **fuzzy clustering** algorithms, where each point belongs partly to several clusters instead of to exactly one.

What I did:

- **Implemented five fuzzy clustering algorithms from the research literature**, from scratch in NumPy: FCM, eFCM, QFCM, YFCM and eQFCM.
- **Extended four of them so they estimate the number of clusters themselves.** Only eFCM originally had this ability: it starts with too many clusters and merges similar ones as it runs. I added the same merging mechanism to the other four, and proposed a new, smoother (sigmoidal) rule for when two clusters should be merged.
- **Implemented every algorithm with two distance measures**: Euclidean, which suits round clusters, and the Gustafson–Kessel (Mahalanobis) distance, which adapts to elongated clusters. That gives ten algorithm–distance combinations.
- **Designed and ran a statistically careful benchmark**: 8 real datasets, 8 synthetic datasets, 30 resampled runs per configuration, and a rank-based selection of **one parameter setting per algorithm that has to work across all datasets**, rather than tuning each dataset separately.

## Key findings

| Algorithm | Mean V-measure, Euclidean | Mean V-measure, Mahalanobis |
|-----------|:---:|:---:|
| FCM   | 0.384 | **0.322** |
| eFCM  | 0.352 | 0.217 |
| QFCM  | 0.357 | 0.250 |
| YFCM  | **0.421** | 0.281 |
| eQFCM | 0.357 | 0.254 |

*Mean over the 8 real datasets, each value averaged over 30 runs. The V-measure ranges from 0 to 1; higher means better agreement with the true classes. FCM with the Mahalanobis distance is averaged over 7 datasets, because it failed numerically on Sonar.*

1. **YFCM was the most reliable algorithm.** It had the best average score, and it recovered the correct number of clusters on five of the eight real datasets starting from an upper bound of 10. On the Wine dataset it reached an adjusted Rand score of 0.90.
2. **The simplest model held its own.** Plain FCM, with no tuning parameters at all, was competitive with models that have two to four parameters.
3. **Choosing the number of clusters is the hard part, and the algorithms fail in opposite directions.** In these experiments, eFCM always merged down to a minimum number of clusters, while QFCM and eQFCM kept too many. Only FCM and YFCM reliably stopped at the right number.
4. **The flexible distance failed on real data, but the distance itself was not the problem.** The synthetic experiments showed that the Mahalanobis distance breaks down because estimating a covariance matrix per cluster is unreliable when there are many features and few points per cluster. On low-dimensional data with elongated clusters, the same settings worked very well.
5. **Preprocessing can matter more than tuning.** Compared with the paper that introduced eQFCM (Kanzawa, 2024), which tunes parameters separately for each dataset, my results were far better on Wine (adjusted Rand score 0.90 vs 0.47) and Heart (0.39 vs 0.05). Both are datasets whose features have very different scales, so the likely reason is that I standardised the features first. In Wine, a single unscaled feature accounts for 99.8% of the total variance.

The thesis also discusses the limitations of the study, such as the fact that parameter selection uses the true labels and that no formal significance tests were run. It ends with concrete proposals for future work.

## How the evaluation works

Each of the ten algorithm–distance combinations goes through the same two-stage protocol:

1. **Fuzziness parameters.** With the number of clusters fixed to the true value, sweep a parameter grid (up to 1,080 configurations for eQFCM). Rank the configurations on each dataset by V-measure, then add up the ranks across datasets (a Borda count) to pick the one that works best overall.
2. **Merging threshold.** Starting from the chosen parameters and an over-estimated number of clusters, sweep 11 merging rules (9 fixed thresholds, the original adaptive rule, and my sigmoidal rule) and select the best in the same way.

Every configuration is evaluated 30 times, each on a different 90% stratified subsample with a different random starting point. This measures how stable the result is, not just how good it is on average. The selected configurations are reported with 13 quality metrics, including chance-corrected scores (adjusted Rand score and adjusted mutual information), and then tested unchanged on synthetic data whose true structure is known.

## Repository structure

```
paper_master_thesis.pdf              # the full thesis
fuzzyclust_algorithms/
    base.py                          # shared iteration loop, cluster merging, Euclidean & Gustafson–Kessel distances
    fcm.py  efcm.py  qfcm.py         # one class per algorithm, each with a merging and a fixed-C mode
    yfcm.py  eqfcm.py
utils.py                             # resampling, parameter grids, V-measure ranking, Borda aggregation, validity indices
testing_real_datasets.py             # full two-stage protocol on the 8 real datasets
testing_toy_datasets.py              # generates the 8 synthetic datasets and tests the selected settings on them

Real_results_Euclidean/              # raw results per algorithm and dataset (mFalse = fixed C, mTrue = merging)
Real_results_Mahalanobis/            # same, with the Gustafson–Kessel distance
Real_results_BEST_COMBO/             # the selected configuration for each algorithm, with all metrics
Real_results_BEST_COMBO_filtered/    # the same, filtered to the metrics reported in the thesis
Toy_results_BEST_COMBO/              # results on the synthetic datasets
```

## Running the code

Requires Python 3.10 or later. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python testing_toy_datasets.py     # synthetic benchmark, about 10 seconds
uv run python testing_real_datasets.py    # full real-data benchmark, takes hours
```

Or with pip:

```bash
pip install numpy scipy scikit-learn pandas matplotlib
python testing_toy_datasets.py
```

The real datasets are downloaded automatically from scikit-learn and OpenML. Results are written as CSV files to the folders listed above.

## Tools

Python · NumPy · SciPy · pandas · scikit-learn · Matplotlib · uv

## License

The thesis document is licensed under Creative Commons BY-NC-ND.
