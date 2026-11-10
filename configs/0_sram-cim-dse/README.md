# SRAM-CiM design space exploration

DSE of an SRAM-CiM accelerator for ResNet-18 on ImageNet (paper Sec. 6.3).
Fixed settings: 1,152-row crossbar, 8b input/weight, differential weight cells, 12b ADC, 8b DAC.

- `dse_sweep.yaml`: full design space exploration over SRAM cell size × DAC × ADC
  (6 × 15 × 43 points), small-noise interpolated (mode: estimate)
- `eap-optimized-designs.yaml`: full-dataset accuracy measurement without interpolation
  for the selected EAP-optimized designs (mode: simulate)

```sh
python -m accuracim --config configs/0_sram-cim-dse/dse_sweep.yaml
python -m accuracim --config configs/0_sram-cim-dse/eap-optimized-designs.yaml
```

## Noise values

Each `args` value is the noise stdev relative to the full-scale range.
ADC/DAC values are converted from survey-reported SNDR: `σ = 1 / sqrt(8 · 10^(SNDR/10))`.

**SRAM cell** (`weight_physical_noise`): 6-T SRAM device variation [3], scaled to 32nm [4].
Cell size ×1 to ×32 gives 12.04% to 2.128% (σ ∝ 1/√size).

**8b DAC** (`input_physical_noise`) [2], SNDR (dB) in config order:
68.0, 61.0, 60.0, 54.0, 48.0, 43.0, 42.0, 40.8, 40.0, 38.0, 37.0, 34.0, 33.0, 30.0, 29.0

**12b ADC** (`adc_physical_noise`) [1], SNDR (dB) in config order:
101.9, 98.0, 73.9, 72.6, 71.0, 70.4, 70.0, 69.7, 69.1, 68.2, 68.1, 68.0, 67.5, 67.1, 67.0,
66.6, 66.0, 65.7, 65.4, 65.3, 65.0, 64.4, 64.4, 64.0, 63.5, 63.2, 63.0, 62.8, 62.6, 62.1,
62.0, 61.1, 61.0, 60.8, 60.5, 60.4, 60.2, 60.1, 60.0, 60.0, 59.6, 59.3, 59.0

## References

1. B. Murmann, "ADC Performance Survey 1997–2025." https://github.com/bmurmann/ADC-survey
2. P. Caragiulo, C. Daigle, B. Murmann, "DAC Performance Survey 1996–2020." https://github.com/pietro-caragiulo/survey-DAC
3. T. Fischer et al., "A 65 nm test structure for SRAM device variability and NBTI statistics," ESSDERC 2009.
4. H. S. Yang et al., "Scaling of 32nm low power SRAM with high-K metal gate," IEDM 2008.
