# Additivity test

Additivity test over a wide noise range. Memory-cell (state-dependent), DAC and ADC
(state-independent) noise are applied at the same time. The test compares the interpolated
accuracy with full-dataset simulation (paper Sec. 6.2, Table 6, Fig. 4).

One config per model (`r18`, `r50`, `gpt-s`, `gpt-m`, `gpt-l`):

```sh
python -m accuracim --config configs/1_additivity-test/r18.yaml                   # interpolated
python -m accuracim --config configs/1_additivity-test/r18.yaml --mode simulate   # full-dataset, to compare
```
