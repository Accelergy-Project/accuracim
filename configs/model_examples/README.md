# Model examples

State-independent ADC noise sweeps, clipping at the 100th percentile, at each model's
input / weight bits (chosen by the bit searches in `bits/`).

```sh
configs/model_examples/run.sh                   # every ADC noise sweep
configs/model_examples/run.sh ResNet34 ViT-B16  # these only
configs/model_examples/run.sh --bits            # bit searches
GPUS="0 1" configs/model_examples/run.sh        # several GPUs
```

| Model | Input / weight bits | ADC noise |
|---|---|---|
| ResNet34 | 6 / 8 | 6e-5 – 1.7e-4 |
| ResNet101 | 10 / 10 | 1.4e-5 – 4.4e-5 |
| ResNet152 | 7 / 8 | 3e-5 – 9.5e-5 |
| VGG16-BN | 8 / 10 | 2.8e-5 – 8.8e-5 |
| MobileNetV2 | 10 / 12 | 1.8e-5 – 5.7e-5 |
| EfficientNet-B0 | 12 / 14 | 6e-7 – 2e-6 |
| ViT-B16 | 10 / 7 | 6e-6 – 1.7e-5 |
| DistilGPT2 | 10 / 10 | 1e-8 – 1e-3 |
| Pythia-160M | 10 / 12 | 5e-7 – 1.6e-6 |
| SmolLM2-135M | 14 / 10 | 1.9e-6 – 5.5e-6 |
| SmolLM2-360M | 16 / 10 | 9e-7 – 2.5e-6 |
| Qwen2.5-0.5B | 14 / 10 | 1.6e-6 – 4.8e-6 |
| GPT2-XL | 10 / 8 | 2.8e-6 – 9.3e-6 |
