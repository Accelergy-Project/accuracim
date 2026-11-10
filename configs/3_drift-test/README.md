# Drift test

Weight conductance drift test (`W → W(1 − α)`). It is not validated yet, so these configs
run full-dataset simulation only (mode: simulate).

Input/ADC clipping ranges are calibrated once on the undrifted array and kept fixed across
drift values. `r18-noise.yaml` adds DAC (input) noise on top of drift.

```sh
python -m accuracim --config configs/3_drift-test/r18.yaml
```
