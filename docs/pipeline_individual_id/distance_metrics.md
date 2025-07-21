# distance_metrics.py

`compute_distances()` calculates several metrics between two vectors. The implementation shows the supported functions:

```python
    distances = {
        "euclidean": euclidean(a, b),
        "manhattan": cityblock(a, b),
        "cosine": cosine(a, b),
        "chebyshev": chebyshev(a, b),
        "canberra": canberra(a, b),
        "braycurtis": braycurtis(a, b),
    }
```
【F:src/FIT_python/pipeline_individual_id/distance_metrics.py†L14-L29】

* **Euclidean** and **Manhattan** are the familiar $\ell_2$ and $\ell_1$ distances from classical geometry. They are easy to interpret and are the basis for many optimisation algorithms, but Euclidean distance in particular can be dominated by overall scale when features are not standardised.

* **Cosine** measures the angle between two vectors and is therefore independent of their magnitude. This metric is widely used in text retrieval and other high‑dimensional applications, see e.g. *Salton & McGill, 1983*. Its main drawback is that it ignores vector length entirely, which may be undesirable for some tasks.

* **Chebyshev** considers only the largest absolute component-wise difference between two vectors. It is useful when an error in any single dimension should be penalised strongly. However, it discards information from the remaining dimensions.

* **Canberra** and **Bray‑Curtis** normalise the differences by the absolute values of the components. They are particularly popular in ecology and compositional data analysis (e.g. *Bray & Curtis, 1957*). Their sensitivity to small changes around zero can be both a strength and a weakness depending on noise levels.

The helper function returns all metrics simultaneously as a dictionary for easy downstream consumption.

**References**
* Bray, J. R., & Curtis, J. T. (1957). "An ordination of the upland forest communities of southern Wisconsin." *Ecological Monographs*.
* Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw‑Hill.
