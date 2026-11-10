# Scaling test

Scaling test of weight conductance variation, with state-dependent (`*_dep`) and
state-independent (`*_indep`) Gaussian noise. The accuracy at each noise level is
interpolated from two anchor runs. The test compares it with full-dataset simulation
(paper Sec. 6.1.1, Table 4, Fig. 2).

```sh
python -m accuracim --config configs/2_scaling-test/r18_dep.yaml                   # interpolated
python -m accuracim --config configs/2_scaling-test/r18_dep.yaml --mode simulate   # full-dataset, to compare
```
