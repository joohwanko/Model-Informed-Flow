# Model-Informed Flow for Bayesian Inference

This repository contains the implementation of the paper [Model-Informed Flows for Bayesian Inference](https://arxiv.org/abs/2505.24243) (Joohwan Ko and Justin Domke, NeurIPS 2025).

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Train with Gaussian
uv run train.py --model 8schools --family gaussian

# Train with IAF
uv run train.py --model 8schools --family iaf --num_layers 5

# Train with MIF (Model-Informed Flow) - affine by default
uv run train.py --model funnel --family mif --num_layers 5

# Train with MIF + deeper network
uv run train.py --model funnel --family mif --num_layers 5 --hidden_units 32 --deep_net

# Train with custom FAF + prior info
uv run train.py --model funnel --family faf --num_layers 5 --use_prior_info --use_t
```

## Available Models

- `8schools`: Eight Schools hierarchical model
- `seeds`: Seed germination
- `sonar`: Sonar classification
- `ionosphere`: Radar signal classification
- `funnel`: Funnel distribution
## Variational Families

- `gaussian`: Mean-field or full-rank Gaussian
- `faf`: Forward Autoregressive Flow
- `iaf`: Inverse Autoregressive Flow 
- `mif`: Model-Informed Flow (FAF with prior info + translation term)

## NCP Methods

- `CP`: Centered Parameterization
- `NCP`: Non-Centered Parameterization
- `VIP`: Variationally Inferred Parameters (learns optimal λ)
- `Dual-VIP`: Learns both λ and ν

## Python API

```python
import jax
from model_informed_flow import get_model, create_variational_family, train_model

jax.config.update("jax_enable_x64", True)

model = get_model("8schools")
variational_family = create_variational_family(
    family_type="faf",
    gaussian_param="mean-field",
    u_latent_size=model.u_latent_size,
    ncp_distribution="variational_ncp",
    num_flow_layers=5,
)

params, final_elbo, _ = train_model(model, variational_family, ncp_method="VIP")
print(f"ELBO: {final_elbo:.4f}")
```

## Examples

```bash
uv run examples/basic_training.py
uv run examples/flow_comparison.py
uv run examples/ncp_methods.py
```

## Arguments

```bash
# Model
--model MODEL          # Model name (default: 8schools)
--funnel_dim DIM       # Funnel dimension (default: 10)

# Variational family
--family FAMILY        # gaussian, faf, iaf, or mif (default: gaussian)
--gaussian_param TYPE  # mean-field or full-rank (default: mean-field)
--ncp_method METHOD    # CP, NCP, VIP, or Dual-VIP (default: CP)

# Flow parameters (for faf/iaf/mif)
--num_layers N         # Number of flow layers (default: 1)
--hidden_units N       # Hidden units in MLP, 0=linear (default: 32)
--use_prior_info       # Use model prior f_i, g_i (MIF uses this)
--use_t                # Use translation term t_i (MIF uses this)
--deep_net             # Use deep MLP vs linear
--epsilon_t_input      # Use epsilon as input to t network
--train_base_dist      # Train base distribution
--unknown_order        # Reverse variable order

# Training
--num_steps N          # Training steps (default: 100000)
--lr RATE              # Learning rate (default: 0.001)
--seed N               # Random seed (default: 0)
--print_every N        # Print frequency (default: 10000)
```

## MIF Configuration

MIF automatically sets:
- `use_prior_info=True` (uses model's f_i and g_i functions)
- `use_t=True` (includes translation term)
- `epsilon_t_input=True` (uses epsilon as input to t network)
- `unknown_order=False` (respects hierarchical order)
- `mlp_hidden_unit=0` (affine flow: linear layers only, no hidden units)
- `deep_net=False` (affine flow by default)

You can override these with `--hidden_units` and `--deep_net` flags for deeper networks.

## Directory Structure

```
model_informed_flow/
├── models.py              
├── variational_families.py 
└── training.py            
examples/
└── basic_training.py     
data/
└── *.data
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ko2025model,
  title={Model-Informed Flows for Bayesian Inference},
  author={Ko, Joohwan and Domke, Justin},
  journal={arXiv preprint arXiv:2505.24243},
  year={2025}
}
```
