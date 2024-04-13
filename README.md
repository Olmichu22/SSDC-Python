# SSDC-Python
Semi-supervised DenPeak Clustering algorithm (Python)
# Semi-supervised DenPeak Clustering (SSDC) Implementation

This section details the implementation of the Semi-supervised DenPeak Clustering (SSDC)[1] algorithm in Python. The original Matlab code developed by the authors can be found at [SSDC Matlab Implementation](https://github.com/Huxhh/SSDC). This Python implementation adapts the procedures to the target language and encapsulates them in a generic class, similar to other SSL algorithms.

## Overview

The SSDC algorithm aims to generate temporary centers based on density measures and the minimum distance to a denser point, followed by the assignment, merging, and splitting of clusters. This ensures that no 'cannot-link' constraints are violated. The pseudocode for the algorithm is detailed below, and it's worth noting that step 7.1.1 varies slightly from the original to accommodate scenarios where the density of most points is null. Additionally, the two methods for calculating density present in the original code are implemented: the 'cutoff' method counts the number of points within a threshold distance \(d\), while the 'gaussian' method uses the function \(\phi = e^{-(D_{ij}/d)^2}\) to calculate each point's contribution to the density.

### Halo Points

Halo points, or boundary points, are also considered in this implementation. These points are determined by calculating the average boundary density for each cluster (given by the maximum average density of two points from different clusters within the threshold distance) and comparing each point's density against this value. If it is lower, the point is considered part of the halo and is excluded from merger considerations. This aspect is a modification from the original algorithm, which calculated this before adjusting clusters for constraints (for unknown reasons), whereas in the Python implementation, this calculation is made before any final potential mergers to accommodate modifications made during the restructuring to ensure constraint compliance.

### Constraint Compliance

A key feature of this algorithm is its assurance of 100% compliance with 'cannot-link' constraints. Once the division phase is complete, ensuring that no 'cannot-link' constraints remain unmet, no further mergers are performed unless they do not violate any pre-validated 'cannot-link' constraints. As demonstrated in various configurations of the algorithm, the error rate for these types of constraints consistently remains at 0%.

## Pseudocode

Below is the pseudocode for the SSDC algorithm:

```algorithm
1. Input: Data matrix X, Must-link constraints ML, Cannot-link constraints CL, Distance threshold factor P, initial center factor D.
2. Initialization: Initial number of clusters (based on sqrt(n)/D), kernel options, distance options.
3. Step 1: Calculate distance matrix.
4. Step 2: Obtain threshold distance d, the position P*D where D is a vector of all unique, sorted distances.
5. Step 3: Calculate density ρ using the selected kernel and threshold distance d.
6. Step 4: Calculate minimum distances δ to the nearest denser neighbor.
7. Step 5: Determine initial centers by selecting the highest values of ρ * δ.
8. Step 6: Assign each point to the cluster of the nearest denser point.
9. Step 7: Address cannot-link and must-link constraints:
   - Repeat:
     - 7.1.1: Search for an unmet cannot-link constraint.
     - 7.1.2: Split the cluster into two, assigning the new central point to the point with the second highest value of ρ * δ.
     - 7.1.3: Reassign clusters for each point in the split cluster, assigning the cluster of the nearest denser point.
   - Until no cannot-link constraints remain unmet.
   - 7.2: For must-links, merge the clusters they belong to, provided no cannot-link constraints are violated.
10. Step 8: Calculate halo points.
11. Step 9: Merge clusters that have points within distance d and are not part of the halo, provided no cannot-link constraints are violated.
12. Output: Final clusters.
```
## References

[1] Y. Ren, X. Hu, S. Ke, G. Yu, D. Yao, and Z. Xu, "Semi-supervised DenPeak Clustering with Pairwise Constraints," in PRICAI 2018: Trends in Artificial Intelligence, pp. 837–850, 2018.


## License

This project is licensed under the GNU General Public License v3.0. This license allows you the freedom to run, study, share, and modify the software while ensuring that all derivatives remain free as well.

For more details, see the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
