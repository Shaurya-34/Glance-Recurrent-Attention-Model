# Glance — Recurrent Attention Model

A PyTorch implementation of a **Recurrent Attention Model (RAM)** that learns where to look before deciding what it sees.

Instead of processing an entire image at full resolution, the model repeatedly extracts small, location-dependent **multi-scale glimpses**, integrates them with an LSTM, and learns a policy for selecting the next location using **REINFORCE**.

The notebook develops and evaluates the model on standard MNIST, translated MNIST, and cluttered translated MNIST.

## Why this project?

Conventional image classifiers process the whole input repeatedly. RAM takes a different approach: it treats visual classification as a sequential decision problem.

At each timestep:

1. A location selects a small region of the image.
2. The glimpse sensor extracts that region at multiple scales.
3. The glimpse and its location are encoded into a compact representation.
4. An LSTM updates the model's recurrent state.
5. The location head proposes where to look next.
6. After the final glimpse, the action head predicts the digit class.

During training, the location policy is sampled from a Gaussian distribution and optimized with a REINFORCE objective. A learned baseline reduces the variance of the policy-gradient estimate.

## Architecture

```text
                        ┌──────────────────────┐
                        │       Input Image     │
                        └──────────┬───────────┘
                                   │
                              location l_t
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │     Glimpse Sensor       │
                     │  multi-scale crop/resize │
                     └────────────┬─────────────┘
                                  │
                         glimpse + location
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │     Glimpse Network      │
                     │      128-d embedding     │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │       LSTMCell           │
                     │      hidden size 256     │
                     └───────┬─────────┬────────┘
                             │         │
                    ┌────────┘         └───────────┐
                    ▼                              ▼
             Location Head                    Baseline Head
              2-D mean (tanh)                  reward estimate
                    │
                    ▼
             Gaussian policy
                    │
                    ▼
              next location

                         final hidden state
                                │
                                ▼
                         Action / Classifier
                              10 classes
```

### Multi-scale glimpse sensor

The sensor uses `torch.nn.functional.affine_grid` and `grid_sample` to extract a crop centered at the current normalized location. Larger crops are resized back to the same `patch_size`, then concatenated across scales.

The default model configuration uses:

- `patch_size = 8`
- `zoom_factors = [1.0, 2.0]`
- `glimpse_dim = 128`
- `hidden_dim = 256`
- `num_classes = 10`
- policy standard deviation `std = 0.25` in the translated/cluttered experiments
- `6` glimpses for standard MNIST
- `8` glimpses for translated MNIST experiments

Locations are represented in normalized `[-1, 1]` coordinates. During training they are sampled from a Gaussian centered on the location head's output and then clamped to the valid range.

### Loss

The notebook combines three terms:

```text
Total Loss = Cross-Entropy
           + REINFORCE Policy Loss
           + Baseline MSE Loss
```

The terminal reward is `1` for a correct classification and `0` otherwise. The learned baseline is subtracted from this reward to form the advantage used by the policy-gradient term.

## Experiments

### 1. Standard MNIST

The baseline experiment trains on the normal `28×28` MNIST images.

Configuration:

```python
model = RAM(
    channels=1,
    patch_size=8,
    zoom_factors=[1.0, 2.0],
    hidden_dim=256,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Training uses a batch size of `64`, `6` glimpses, and `5` epochs.

**Recorded notebook result:** **94.09% test accuracy**.

### 2. Translated MNIST

Each `28×28` digit is randomly placed inside a `60×60` canvas. This forces the attention policy to search rather than relying on the digit being centered.

The experiment uses `8` glimpses and `std=0.25` for the stochastic location policy.

The notebook also visualizes the learned sequence of glimpse locations over the image.

### 3. Cluttered + Translated MNIST

The cluttered dataset places the target digit randomly on a `60×60` canvas and adds several random `8×8` patches sampled from other MNIST digits.

The attention model is then trained to find the relevant digit while ignoring distractors.

After an additional `25` training epochs, the notebook records:

**Final test accuracy: 67.39%**

The notebook also plots the attention trajectories, making it possible to inspect where the policy looks over time.

> These are notebook-recorded results, not a controlled benchmark. The repository does not currently include checkpoint files, a reproducible experiment script, or a systematic hyperparameter sweep.

## Visualizing attention

The notebook includes a `visualize_glimpses(...)` helper that overlays the sequence of normalized attention locations on input images.

This makes the model's behavior directly inspectable:

```text
Image → glimpse 1 → glimpse 2 → ... → glimpse T → prediction
                         ↘ learned search trajectory ↗
```

## Running the notebook

The project is currently organized as a **Google Colab notebook** rather than a Python package.

### Requirements

The notebook uses:

- Python 3
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Kaggle API for the initial raw-MNIST download

A CUDA-enabled GPU is used when available.

### Option 1 — Google Colab

Open `Glance.ipynb` in Google Colab and run the cells from top to bottom.

The notebook was authored with GPU execution in mind and records a Colab `T4` runtime configuration.

### Option 2 — Local environment

Install the dependencies:

```bash
pip install torch torchvision numpy matplotlib kaggle
```

Then launch Jupyter:

```bash
jupyter notebook Glance.ipynb
```

The first section downloads the MNIST dataset through Kaggle. If you use that path, configure your Kaggle API credentials first.

## Repository structure

```text
.
├── Glance.ipynb   # complete implementation, experiments, training and visualization
└── README.md      # project documentation
```

## Implementation notes

This repository is intentionally notebook-driven. The same notebook contains several iterations of the model, including the standard-MNIST version and a version adapted for translated/cluttered inputs.

A few details worth knowing when reading or extending the code:

- The initial attention location is randomly sampled in `[-1, 1]` for each training example in the translated/cluttered model.
- In evaluation mode, the model uses the predicted location mean directly instead of sampling from the Gaussian policy.
- The final class prediction is produced from the **last recurrent hidden state**, after the configured number of glimpses.
- The current reward is terminal and based only on whether the final prediction is correct.
- The notebook contains exploratory/demo cells in addition to the final training loops, so it should be read as an experimental implementation rather than a polished training framework.

## Background

This project follows the core idea from:

> Mnih et al., *Recurrent Models of Visual Attention* (2014)

The key idea is to learn a task-dependent sequence of visual observations instead of processing the full image at every step.

Paper: https://arxiv.org/abs/1406.6247

## Future directions

Natural next steps would be:

- move the model and datasets into reusable `.py` modules
- add deterministic experiment configurations and seeds
- save/load checkpoints
- log per-epoch validation metrics
- compare against a CNN baseline
- test different glimpse counts, scales, and policy variances
- improve the reward/policy formulation for cluttered scenes

## License

No license file is currently included in the repository.

## Author

**Shaurya-34**

Repository: https://github.com/Shaurya-34/Glance-Recurrent-Attention-Model
