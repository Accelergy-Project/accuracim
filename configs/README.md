# Configs

```
configs/
├── quick_examples/      small runs that download their own data (Imagenette, WikiText-2)
├── 0_sram-cim-dse/      SRAM-CiM design space exploration (ResNet-18)
├── 1_additivity-test/   memory-cell, DAC and ADC noise together
├── 2_scaling-test/      memory-cell noise, state-dependent and -independent
├── 3_drift-test/        conductance drift (simulate only)
└── model_examples/      ADC noise sweeps for the other models in workloads/models.py
```

`run.mode` is `estimate` (a few anchor runs, interpolation for the rest) or `simulate`
(every point). Run from the repo root. `--mode` overrides the config:

```sh
python -m accuracim --config configs/1_additivity-test/r18.yaml
python -m accuracim --config configs/1_additivity-test/r18.yaml --mode simulate
```
