# thesis-clustering

Code accompanying the Master's thesis *"Implementation and Assessment of
Cluster size-Adaptable Fuzzy Clustering Algorithms with Cluster Merging
Capabilities"* (MUSI, University of the Balearic Islands).

The thesis extends the similarity-based cluster-merging mechanism of eFCM
(Kaymak & Setnes) — which lets an algorithm estimate the number of clusters
$C$ from an upper bound instead of requiring it up front — to four further
fuzzy clustering algorithms, and evaluates the resulting ten
algorithm/distance combinations on real and synthetic data.

## Algorithms

| Algorithm | Reference | `merge=True` variant |
|---|---|---|
| FCM   | Bezdek (1981) | ✅ |
| eFCM  | Kaymak & Setnes (2000, 2002) | ✅ (native) |
| QFCM  | Miyamoto, Ichihashi & Honda (2008) | ✅ |
| YFCM  | Yang (1993) | ✅ |
| eQFCM | Kanzawa (2024) | ✅ |

Each algorithm is implemented under both the **Euclidean** distance and the
volume-normalised **Gustafson–Kessel (Mahalanobis)** distance.

## Repository structure

```
fuzzyclust_algorithms/     # core implementations: fcm.py, efcm.py, qfcm.py, yfcm.py, eqfcm.py
utils.py                   # shared pipeline: bootstrap/resampling, parameter grids,
                           # V-measure-based ranking and selection, threshold sweeps, etc
testing_real_datasets.py   # calculates the best combination of parameters for the real datasets
testing_toy_datasets.py    # tests the combinations obtained from the real datasets to the toy datasets
Real_results_Euclidean/    # per-algorithm CSVs, Euclidean, merge=False and merge=True
Real_results_Mahalanobis/  # same, Gustafson–Kessel distance
Real_results_BEST_COMBO/            # selected (fuzzification, alpha) configuration per algorithm
Real_results_BEST_COMBO_filtered/   # same, filtered to the reported metrics
Toy_results_BEST_COMBO/    # results of the selected configurations on synthetic data
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) (see `uv.lock`):

```bash
uv sync
```

Alternatively, with `pip`:

```bash
pip install numpy scipy scikit-learn matplotlib
```

Requires Python ≥ 3.10.

## Reproducing the results

```bash
# Real datasets: parameter sweep, V-measure-based selection, then the
# merging-threshold sweep, for every algorithm and both distances
python testing_real_datasets.py

# Synthetic datasets, per algorithm/distance
python testing_toy_datasets.py
# ... etc.
```

Results are written as CSVs to the `Real_results_*` and `Toy_results_*`
folders described above.


## License

See the thesis document for the applicable license.
