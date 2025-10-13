"""
Training utilities for Model-Informed Flow

This module contains functions for training variational inference models
using various optimization strategies.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import time
from typing import Dict, Optional, Tuple

from .variational_families import VariationalFamily, ForwardAutoregressiveFlow, InverseAutoregressiveFlow, model_log_prob
from .models import HierarchicalBayesianModel


def create_variational_family(
    family_type: str,
    gaussian_param: str,
    u_latent_size: int,
    ncp_distribution: str,
    **kwargs
) -> VariationalFamily:
    """
    Factory function to create variational families.
    
    Args:
        family_type: Type of variational family ("gaussian", "faf", "iaf", "mif")
        gaussian_param: "mean-field" or "full-rank"
        u_latent_size: Dimension of latent space
        ncp_distribution: NCP distribution type
        **kwargs: Additional arguments for specific family types
            For FAF/IAF/MIF:
                - num_flow_layers: Number of flow layers (default: 5)
                - mlp_hidden_unit: Hidden units in MLP (default: 0)
                - use_prior_info: Use model prior information (default: False)
                - use_t: Use translation term (default: False)
                - unknown_order: Reverse variable order (default: False)
                - train_base_dist: Train base distribution (default: False)
                - deep_net: Use deep network (default: False)
                - epsilon_t_input: Use epsilon as input to t (default: False)
        
    Returns:
        A variational family instance
    """
    if family_type == "gaussian":
        return VariationalFamily(gaussian_param, u_latent_size, ncp_distribution)
    
    elif family_type == "faf":
        return ForwardAutoregressiveFlow(
            gaussian_param, u_latent_size, ncp_distribution, **kwargs
        )
    
    elif family_type == "iaf":
        return InverseAutoregressiveFlow(
            gaussian_param, u_latent_size, ncp_distribution, **kwargs
        )
    
    elif family_type == "mif":
        # MIF is FAF with specific settings based on the paper
        # Initialize with affine flow (linear only, no hidden units)
        mif_kwargs = {
            "use_prior_info": True,
            "unknown_order": False,
            "train_base_dist": False,
            "use_t": True,
            "epsilon_t_input": True,
            "deep_net": False,  # Affine flow: no deep network
            "mlp_hidden_unit": 0,  # Affine flow: no hidden units, pure linear
            "num_flow_layers": kwargs.get("num_flow_layers", 1),
        }
        # Allow overriding with user-specified values
        if "deep_net" in kwargs:
            mif_kwargs["deep_net"] = kwargs["deep_net"]
        if "mlp_hidden_unit" in kwargs:
            mif_kwargs["mlp_hidden_unit"] = kwargs["mlp_hidden_unit"]
        
        return ForwardAutoregressiveFlow(
            gaussian_param, u_latent_size, ncp_distribution, **mif_kwargs
        )
    
    else:
        raise ValueError(f"Unknown variational family type: {family_type}")


def estimate_elbo(key, params, lambda_list, nu_list, model, variational_family):
    """Estimate the ELBO for a single sample."""
    latent, log_q = variational_family.sample_and_log_prob(
        key, params, model=model, lambda_list=lambda_list, nu_list=nu_list
    )
    
    if variational_family.ncp_distribution == "model_ncp":
        log_p = model_log_prob(model, latent, lambda_list, nu_list, variational_ncp=False)
    elif variational_family.ncp_distribution == "variational_ncp":
        log_p = model_log_prob(model, latent, lambda_list, nu_list, variational_ncp=True)
    
    elbo = log_p - log_q
    return elbo


def make_loss_function(model, ncp_method, variational_family, batchsize):
    """Create a JIT-compiled loss function."""
    u_latent_size = model.u_latent_size

    @jit
    def loss_function(params, key):
        keys = jax.random.split(key, batchsize)
        
        # Determine lambda_list based on ncp_method
        if ncp_method == "VIP":
            lambda_unconstrained = params['lambda_unconstrained']
            lambda_list = jax.nn.sigmoid(lambda_unconstrained)
            nu_list = lambda_list
        elif ncp_method == "Dual-VIP":
            lambda_unconstrained = params['lambda_unconstrained']
            nu_unconstrained = params['nu_unconstrained']
            lambda_list = jax.nn.sigmoid(lambda_unconstrained)
            nu_list = jax.nn.sigmoid(nu_unconstrained)
        elif ncp_method == "CP":
            lambda_list = jnp.ones(u_latent_size)
            nu_list = jnp.ones(u_latent_size)
        elif ncp_method == "NCP":
            lambda_list = jnp.zeros(u_latent_size)
            nu_list = jnp.zeros(u_latent_size)
        else:
            raise ValueError(f"Invalid ncp_method: {ncp_method}")

        # Vectorized ELBO estimation
        elbos = jax.vmap(estimate_elbo, in_axes=(0, None, None, None, None, None))(
            keys, params, lambda_list, nu_list, model, variational_family
        )
        elbo = jnp.mean(elbos)
        return -elbo

    return loss_function


def compute_elbo(params, key, num_samples, batch_size, model, ncp_method, variational_family):
    """Compute the ELBO over multiple samples in batches."""
    u_latent_size = model.u_latent_size
    
    # Determine lambda_list based on ncp_method
    if ncp_method == "VIP":
        lambda_unconstrained = params['lambda_unconstrained']
        lambda_list = jax.nn.sigmoid(lambda_unconstrained)
        nu_list = lambda_list
    elif ncp_method == "Dual-VIP":
        lambda_unconstrained = params['lambda_unconstrained']
        nu_unconstrained = params['nu_unconstrained']
        lambda_list = jax.nn.sigmoid(lambda_unconstrained)
        nu_list = jax.nn.sigmoid(nu_unconstrained)
    elif ncp_method == "CP":
        lambda_list = jnp.ones(u_latent_size)
        nu_list = jnp.ones(u_latent_size)
    elif ncp_method == "NCP":
        lambda_list = jnp.zeros(u_latent_size)
        nu_list = jnp.zeros(u_latent_size)
    else:
        raise ValueError(f"Invalid ncp_method: {ncp_method}")

    num_batches = num_samples // batch_size
    remainder = num_samples % batch_size
    total_elbo = 0.0

    @jit
    def batch_elbo(keys_batch):
        elbos = jax.vmap(estimate_elbo, in_axes=(0, None, None, None, None, None))(
            keys_batch, params, lambda_list, nu_list, model, variational_family
        )
        return jnp.sum(elbos)

    # Compute ELBO over batches
    for _ in range(num_batches):
        key, subkey = random.split(key)
        keys = random.split(subkey, batch_size)
        total_elbo += batch_elbo(keys)

    # Compute ELBO for remaining samples
    if remainder > 0:
        key, subkey = random.split(key)
        keys = random.split(subkey, remainder)
        total_elbo += batch_elbo(keys)

    final_elbo = total_elbo / num_samples
    return final_elbo


def train_model(
    model: HierarchicalBayesianModel,
    variational_family: VariationalFamily,
    ncp_method: str = "CP",
    num_steps: int = 10000,
    batch_size: int = 1,
    learning_rate: float = 0.01,
    scheduler: str = "piecewise",
    lambda_init: float = 0.5,
    seed: int = 0,
    final_elbo_num_samples: int = 10000,
    final_elbo_batch_size: int = 256,
    print_every: int = 1000,
    wandb_log: bool = False,
) -> Tuple[Dict, float, list]:
    """
    Train a variational inference model.
    
    Args:
        model: The hierarchical Bayesian model
        variational_family: The variational family to optimize
        ncp_method: NCP method ("CP", "NCP", "VIP", "Dual-VIP")
        num_steps: Number of optimization steps
        batch_size: Batch size for ELBO estimation
        learning_rate: Initial learning rate
        scheduler: Learning rate scheduler ("piecewise", "exponential", "none")
        lambda_init: Initial value for lambda (VIP methods)
        seed: Random seed
        final_elbo_num_samples: Number of samples for final ELBO estimation
        final_elbo_batch_size: Batch size for final ELBO estimation
        print_every: Print progress every N steps
        wandb_log: Whether to log to Weights & Biases
        
    Returns:
        params: Optimized parameters
        final_elbo: Final ELBO value
        loss_history: Loss history during training
    """
    start_time = time.time()
    key = random.PRNGKey(seed)
    
    # Create loss function
    loss_function = make_loss_function(model, ncp_method, variational_family, batch_size)

    print(f"\nTraining {model.name} with {type(variational_family).__name__}")
    print(f"NCP method: {ncp_method}, Learning rate: {learning_rate}")
    
    # Setup optimizer
    if scheduler == "piecewise":
        schedule = optax.piecewise_constant_schedule(
            init_value=learning_rate,
            boundaries_and_scales={
                int(num_steps * 1/3): 0.1,
                int(num_steps * 2/3): 0.1
            }
        )
        optimizer = optax.adam(schedule)
    elif scheduler == "exponential":
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=int(num_steps // 100),
            decay_rate=0.99,
            staircase=False
        )
        optimizer = optax.adam(schedule)
    elif scheduler == "none":
        optimizer = optax.adam(learning_rate)
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")
    
    # Initialize parameters
    params = variational_family._init_variational_params()
    
    # Initialize lambda parameters for VIP methods
    if ncp_method == "VIP":
        if not (0 < lambda_init < 1):
            raise ValueError("lambda_init must be in the range (0, 1)")
        lambda_init_value = jnp.log(lambda_init / (1 - lambda_init))
        params['lambda_unconstrained'] = jnp.ones(model.u_latent_size) * lambda_init_value

    if ncp_method == "Dual-VIP":
        if not (0 < lambda_init < 1):
            raise ValueError("lambda_init must be in the range (0, 1)")
        lambda_init_value = jnp.log(lambda_init / (1 - lambda_init))
        params['lambda_unconstrained'] = jnp.ones(model.u_latent_size) * lambda_init_value
        params['nu_unconstrained'] = jnp.ones(model.u_latent_size) * lambda_init_value

    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state, key):
        loss, grads = value_and_grad(loss_function)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    loss_history = []
    train_time = time.time()
    
    for step_num in range(num_steps):
        key, subkey = random.split(key)
        params, opt_state, loss = train_step(params, opt_state, subkey)
        loss_history.append(float(loss))

        # Log to W&B if enabled
        if wandb_log:
            try:
                import wandb
                if step_num % 20 == 0:
                    current_time = time.time()
                    elapsed_time = current_time - train_time if step_num > 0 else 0
                    wandb.log({"train_loss": float(loss), "wall_clock_time": elapsed_time}, step=step_num)
            except ImportError:
                pass

        # Print progress
        if step_num % print_every == 0 or step_num == num_steps - 1:
            avg_loss = jnp.mean(jnp.array(loss_history[-50:]))
            current_time = time.time()
            elapsed_time = current_time - train_time
            print(f"Step {step_num}/{num_steps}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
    
    # Compute final ELBO
    key, subkey = random.split(key)
    final_elbo = compute_elbo(
        params, subkey, final_elbo_num_samples,
        final_elbo_batch_size, model, ncp_method, variational_family
    )
    print(f"\nFinal ELBO: {final_elbo:.4f}")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Compute lambda and nu means if applicable
    if ncp_method in ["VIP", "Dual-VIP"]:
        lambda_mean = float(jnp.mean(jax.nn.sigmoid(params['lambda_unconstrained'])))
        print(f"Lambda mean: {lambda_mean:.4f}")
        
        if ncp_method == "Dual-VIP":
            nu_mean = float(jnp.mean(jax.nn.sigmoid(params['nu_unconstrained'])))
            print(f"Nu mean: {nu_mean:.4f}")
    
    if wandb_log:
        try:
            import wandb
            wandb.log({"final_elbo": float(final_elbo), "total_time": total_time})
            if ncp_method in ["VIP", "Dual-VIP"]:
                wandb.log({"lambda_mean": lambda_mean})
                if ncp_method == "Dual-VIP":
                    wandb.log({"nu_mean": nu_mean})
        except ImportError:
            pass
    
    return params, float(final_elbo), loss_history

