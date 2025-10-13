"""
Variational Families for Model-Informed Flow

This module implements various variational families for variational inference:
- Gaussian (mean-field and full-rank)
- Forward Autoregressive Flow (FAF)
- Inverse Autoregressive Flow (IAF)
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from jax.flatten_util import ravel_pytree
from jax.scipy.stats.norm import logpdf
from jax import flatten_util
from typing import List


# ============================================================================
# Helper: Autoregressive MLP
# ============================================================================

class AutoregressiveMLP(nn.Module):
    """
    Simple MLP for autoregressive flows.
    
    Attributes:
        hidden_unit: Number of hidden units (0 for linear layer only)
        output_dim: Output dimension
        mlp_init_std: Standard deviation for weight initialization
    """
    hidden_unit: int  # Single hidden layer size
    output_dim: int = 1
    mlp_init_std: float = 0.01
    
    @nn.compact
    def __call__(self, x):
        # Direct path from input to output
        direct_output = nn.Dense(
            features=self.output_dim,
            kernel_init=jax.nn.initializers.normal(stddev=self.mlp_init_std)
        )(x)
        
        # Add hidden path if hidden_unit > 0
        if self.hidden_unit > 0:
            hidden = nn.Dense(
                features=self.hidden_unit,
                kernel_init=jax.nn.initializers.normal(stddev=self.mlp_init_std)
            )(x)
            hidden = nn.relu(hidden)
            hidden_output = nn.Dense(
                features=self.output_dim,
                kernel_init=jax.nn.initializers.normal(stddev=self.mlp_init_std)
            )(hidden)
            return direct_output + hidden_output
        else:
            return direct_output


# ============================================================================
# Base Variational Family: Gaussian
# ============================================================================

class VariationalFamily:
    """
    Base Gaussian variational family (mean-field or full-rank).
    
    This class implements a Gaussian distribution parameterized by:
    - mean: Location parameter
    - L_param: Scale parameter (diagonal for mean-field, lower triangular for full-rank)
    
    Args:
        gaussian_param: "mean-field" or "full-rank"
        u_latent_size: Dimension of the latent space
        ncp_distribution: NCP distribution type
    """

    def __init__(self, gaussian_param: str, u_latent_size: int, ncp_distribution: str):
        assert gaussian_param in ["mean-field", "full-rank"], \
            "gaussian_param must be either 'mean-field' or 'full-rank'."
        
        self.gaussian_param = gaussian_param
        self.u_latent_size = u_latent_size
        self.ncp_distribution = ncp_distribution

    def _init_variational_params(self):
        """Initialize variational parameters."""
        mean = jnp.zeros(self.u_latent_size)
        
        # Initialize L_param to get softplus(L_param) = 1.0
        initial_L_param_value = jnp.log(jnp.exp(1.0) - 1.0)
        
        if self.gaussian_param == "mean-field":
            L_param = jnp.full(self.u_latent_size, initial_L_param_value)
        else:
            # Full-rank: initialize as lower triangular matrix
            L_param = jnp.diag(jnp.full(self.u_latent_size, initial_L_param_value))
        
        params = {"mean": mean, "L_param": L_param}
        flat_params, _ = ravel_pytree(params)
        total_params = flat_params.size
        print(f"Total number of Gaussian parameters: {total_params}")
        return params

    def _build_L(self, L_param):
        """Build the scale matrix L from parameters."""
        if L_param.ndim == 1:
            # Mean-field: diagonal matrix
            diag_L_values = jax.nn.softplus(L_param)
            L = diag_L_values
        else:
            # Full-rank: lower triangular matrix
            diag_L_values = jax.nn.softplus(jnp.diag(L_param))
            diag_L = jnp.diag(diag_L_values)
            L = jnp.tril(L_param, -1) + diag_L
        return L

    def get_loc_scale(self, params):
        """Extract location and scale parameters."""
        m = params["mean"]
        L_param = params["L_param"]
        L = self._build_L(L_param)
        return m, L

    def sample_and_log_prob(self, key, params, model=None, lambda_list=None, nu_list=None):
        """
        Sample from the Gaussian distribution and compute log probability.
        
        Args:
            key: JAX random key
            params: Variational parameters
            model: Optional model (used for NCP transformations)
            lambda_list: Optional lambda parameters for NCP
            nu_list: Optional nu parameters for NCP
            
        Returns:
            sample: A sample from the distribution
            log_q: Log probability of the sample
        """
        # Sample from base Gaussian
        m, L = self.get_loc_scale(params)
        epsilon = random.normal(key, shape=m.shape)
        
        if self.gaussian_param == "mean-field":
            u_tilde_latent = m + L * epsilon
            if L.ndim == 1:
                log_det_cov = 2.0 * jnp.sum(jnp.log(L))
            else:
                raise ValueError("For mean-field, L is 1D.")
        else:
            u_tilde_latent = m + L @ epsilon
            log_det_cov = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        
        k = m.shape[0]
        log2pi = jnp.log(2.0 * jnp.pi)
        log_q_u = -0.5 * (jnp.sum(epsilon**2) + k * log2pi + log_det_cov)

        # Handle variational NCP if specified
        if self.ncp_distribution == "variational_ncp":
            if model is None or lambda_list is None or nu_list is None:
                raise ValueError(
                    "model, lambda_list, and nu_list must be provided for variational_ncp."
                )

            # Apply NCP transformation
            u_tilde_latent_unflatten = model.unflatten(u_tilde_latent)
            lambda_list_unflatten = model.unflatten(lambda_list)
            nu_list_unflatten = model.unflatten(nu_list)
            z_list = []
            sum_log_sigma_terms = 0.0

            for i in range(len(u_tilde_latent_unflatten)):
                u_tilde_i = u_tilde_latent_unflatten[i]
                lambda_i = lambda_list_unflatten[i]
                nu_i = nu_list_unflatten[i]

                mu_i = model.f_i(z_list, i)
                sigma_i = model.g_i(z_list, i)

                z_i = mu_i + (u_tilde_i - lambda_i * mu_i) * (sigma_i ** (1 - nu_i))
                z_list.append(z_i)

                sum_log_sigma_terms += jnp.sum((1 - nu_i) * jnp.log(sigma_i))

            z, _ = flatten_util.ravel_pytree(z_list)
            log_q = log_q_u - sum_log_sigma_terms
            return z, log_q
        else:
            return u_tilde_latent, log_q_u


# ============================================================================
# Forward Autoregressive Flow (FAF)
# ============================================================================

class ForwardAutoregressiveFlow(VariationalFamily):
    """
    Forward Autoregressive Flow.
    
    An autoregressive normalizing flow that transforms a base Gaussian distribution
    through a series of autoregressive transformations.
    
    Args:
        gaussian_param: "mean-field" or "full-rank"
        u_latent_size: Dimension of latent space
        ncp_distribution: NCP distribution type
        num_flow_layers: Number of flow layers
        activation: Activation function (not used in current implementation)
        use_prior_info: Whether to use prior information from model
        unknown_order: Whether to reverse the order of outputs
        train_base_dist: Whether to train the base distribution
        use_t: Whether to use t parameter in the flow
        mlp_hidden_sizes: Hidden layer sizes (not used in current implementation)
        deep_net: Whether to use deep network (not used if False)
        mlp_init_std: Standard deviation for MLP initialization
        epsilon_t_input: Whether to use epsilon as input to t network
        mlp_hidden_unit: Number of hidden units in MLP
    """
    
    def __init__(
        self,
        gaussian_param: str,
        u_latent_size: int,
        ncp_distribution: str,
        num_flow_layers: int = 10,
        activation: str = "relu",
        use_prior_info: bool = False,
        unknown_order: bool = False,
        train_base_dist: bool = False,
        use_t: bool = False,
        mlp_hidden_sizes: List[int] = [64, 64],
        deep_net: bool = False,
        mlp_init_std: float = 0.01,
        epsilon_t_input: bool = False,
        mlp_hidden_unit: int = 0,
        **kwargs,
    ):
        super().__init__(gaussian_param, u_latent_size, ncp_distribution)
        self.num_flow_layers = num_flow_layers
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.mlp_models = {}
        self.use_t = use_t
        self.unknown_order = unknown_order
        self.train_base_dist = train_base_dist
        self.use_prior_info = use_prior_info
        self.deep_net = deep_net
        if not self.deep_net:
            self.mlp_hidden_sizes = []
        self.mlp_init_std = mlp_init_std
        self.epsilon_t_input = epsilon_t_input
        self.mlp_hidden_unit = mlp_hidden_unit
        
        print(f"Forward Autoregressive Flow:")
        print(f"  num_flow_layers: {self.num_flow_layers}")
        print(f"  unknown_order: {self.unknown_order}")
        print(f"  train_base_dist: {self.train_base_dist}")
        print(f"  use_prior_info: {self.use_prior_info}")
        print(f"  use_t: {self.use_t}")
        print(f"  deep_net: {self.deep_net}")
        print(f"  epsilon_t_input: {epsilon_t_input}")
        print(f"  mlp_hidden_unit: {mlp_hidden_unit}")
        
    def _init_variational_params(self):
        """Initialize variational parameters including flow parameters."""
        D = self.u_latent_size
        self.mlp_models = {}
        
        for layer_idx in range(self.num_flow_layers):
            master_key = random.PRNGKey(1234 + layer_idx * D)

            # Create keys for each model type and coordinate
            keys = random.split(master_key, 3 * D)
            keys = keys.reshape(3, D, 2)

            # Create dummy inputs
            dummy_input = jnp.ones((D,))
            if self.epsilon_t_input:
                dummy_input_t = jnp.ones((D * 2,))
            else:
                dummy_input_t = dummy_input
                
            if self.use_prior_info:
                dummy_input = jnp.concatenate([dummy_input, jnp.zeros((2,))])
                dummy_input_t = jnp.concatenate([dummy_input_t, jnp.zeros((2,))])
            
            # Initialize models for each coordinate
            def create_model(key):
                model = AutoregressiveMLP(
                    hidden_unit=self.mlp_hidden_unit,
                    output_dim=1,
                    mlp_init_std=self.mlp_init_std
                )
                params = model.init(key, dummy_input)
                return params
            
            def create_model_t(key):
                model = AutoregressiveMLP(
                    hidden_unit=self.mlp_hidden_unit,
                    output_dim=1,
                    mlp_init_std=self.mlp_init_std
                )
                params = model.init(key, dummy_input_t)
                return params
            
            # Apply vmap over coordinates
            shift_params_all = jax.vmap(create_model)(keys[0])
            scale_params_all = jax.vmap(create_model)(keys[1])
            t_params_all = jax.vmap(create_model_t)(keys[2])

            all_model_params = {
                'shift': shift_params_all,
                'scale': scale_params_all,
                't': t_params_all
            }
        
        # Add base distribution parameters if training base dist
        if self.train_base_dist:
            base_params = super()._init_variational_params()
            all_model_params['base'] = base_params
            
        flat_params, self.unravel_fn = ravel_pytree(all_model_params)
        total_params = flat_params.size
        print(f"Total number of variational parameters: {total_params}")
        return all_model_params
    
    def sample_and_log_prob(self, key, params, model=None, lambda_list=None, nu_list=None):
        """Sample from the flow and compute log probability."""
        # Sample from base distribution
        if self.train_base_dist:
            z_current, log_q_base = super().sample_and_log_prob(
                key, params['base'], model=model, lambda_list=lambda_list, nu_list=nu_list
            )
        else:
            z_current = random.normal(key, shape=(self.u_latent_size,))
            D = self.u_latent_size
            log_const = -0.5 * D * jnp.log(2.0 * jnp.pi)
            log_q_base = -0.5 * jnp.sum(z_current**2, axis=-1) + log_const

        flow_params = params
        D = self.u_latent_size
        mlp_model = AutoregressiveMLP(
            hidden_unit=self.mlp_hidden_unit,
            output_dim=1,
            mlp_init_std=self.mlp_init_std
        )
        
        # Precompute autoregressive masks
        ar_masks = jnp.tril(jnp.ones((D, D)), k=-1)
        
        # Transform each coordinate
        @jax.jit
        def transform_coordinate(i, z_in, z_next, flow_params):
            """Transform a single coordinate using the flow."""
            mask = ar_masks[i]
            masked_input = z_next * mask
            masked_z_in = z_in * mask
            
            if self.use_prior_info:
                masked_input_unflatten = model.unflatten(z_next)
                f_i_list = []
                g_i_list = []
                for j in range(len(masked_input_unflatten)):
                    f_i = model.f_i(masked_input_unflatten, j)
                    g_i = model.g_i(masked_input_unflatten, j)
                    f_i_list.append(f_i)
                    g_i_list.append(g_i)
                f_i, _ = ravel_pytree(f_i_list) 
                f_i = f_i[i]
                g_i, _ = ravel_pytree(g_i_list) 
                g_i = g_i[i]
                masked_input = jnp.concatenate((masked_input, jnp.array([f_i, g_i])))
            
            # Get model parameters for this coordinate
            shift_params_i = jax.tree.map(lambda x: x[i], flow_params["shift"])
            scale_params_i = jax.tree.map(lambda x: x[i], flow_params["scale"])

            shift_i = mlp_model.apply(shift_params_i, masked_input)[0]
            raw_scale_i = mlp_model.apply(scale_params_i, masked_input)[0]
            scale_i = jax.nn.softplus(raw_scale_i) + 1e-6
            
            # Apply transformation
            if self.use_t:
                t_params_i = jax.tree.map(lambda x: x[i], flow_params["t"])
                if self.epsilon_t_input:
                    input_t_i = jnp.concatenate((masked_z_in, masked_input))
                    t_i = mlp_model.apply(t_params_i, input_t_i)[0]
                else:
                    t_i = mlp_model.apply(t_params_i, masked_input)[0]
                new_z_i = shift_i + scale_i * (z_in[i] - t_i)
            else:
                new_z_i = shift_i + scale_i * z_in[i]
                
            return new_z_i, jnp.log(scale_i)
        
        # Coordinate processing step
        def coord_step(carry, i):
            z_next, log_det, z_in = carry
            new_z_i, log_scale_i = transform_coordinate(i, z_in, z_next, flow_params)
            z_next = z_next.at[i].set(new_z_i)
            log_det = log_det + log_scale_i
            return (z_next, log_det, z_in), None
        
        # Layer processing function
        @jax.jit
        def process_layer(z_in):
            z_next = jnp.zeros_like(z_in)
            log_det = 0.0
            (z_next, log_det, _), _ = jax.lax.scan(
                coord_step, (z_next, log_det, z_in), jnp.arange(D)
            )
            return z_next, log_det
        
        # Layer step for scan
        def layer_step(carry, _):
            z, total_log_det = carry
            z_next, layer_log_det = process_layer(z)
            return (z_next, total_log_det + layer_log_det), None
        
        # Process all layers
        init_carry = (z_current, 0.0)
        (z_final, total_log_det), _ = jax.lax.scan(
            layer_step, init_carry, jnp.arange(self.num_flow_layers)
        )
        
        if self.unknown_order:
            z_final = z_final[::-1]

        # Compute final log probability
        log_q_T = log_q_base - total_log_det
        return z_final, log_q_T


# ============================================================================
# Inverse Autoregressive Flow (IAF)
# ============================================================================

class InverseAutoregressiveFlow(VariationalFamily):
    """
    Inverse Autoregressive Flow.
    
    An autoregressive normalizing flow that can be computed in parallel
    for efficient sampling (unlike FAF which requires sequential computation).
    
    Args:
        Same as ForwardAutoregressiveFlow
    """
    
    def __init__(
        self,
        gaussian_param: str,
        u_latent_size: int,
        ncp_distribution: str,
        num_flow_layers: int = 10,
        activation: str = "relu",
        use_prior_info: bool = False,
        unknown_order: bool = False,
        train_base_dist: bool = False,
        use_t: bool = False,
        mlp_hidden_sizes: List[int] = [64, 64],
        deep_net: bool = False,
        mlp_init_std: float = 0.01,
        epsilon_t_input: bool = False,
        mlp_hidden_unit: int = 0,
        **kwargs,
    ):
        super().__init__(gaussian_param, u_latent_size, ncp_distribution)
        self.num_flow_layers = num_flow_layers
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.mlp_models = {}
        self.use_t = use_t
        self.unknown_order = unknown_order
        self.train_base_dist = train_base_dist
        self.use_prior_info = use_prior_info
        self.deep_net = deep_net
        if not self.deep_net:
            self.mlp_hidden_sizes = []
        self.mlp_init_std = mlp_init_std
        self.epsilon_t_input = epsilon_t_input
        self.mlp_hidden_unit = mlp_hidden_unit
        
        print(f"Inverse Autoregressive Flow:")
        print(f"  num_flow_layers: {self.num_flow_layers}")
        print(f"  unknown_order: {self.unknown_order}")
        print(f"  train_base_dist: {self.train_base_dist}")
        print(f"  use_prior_info: {self.use_prior_info}")
        print(f"  use_t: {self.use_t}")
        print(f"  deep_net: {self.deep_net}")
        print(f"  epsilon_t_input: {epsilon_t_input}")
        print(f"  mlp_hidden_unit: {mlp_hidden_unit}")
        
    def _init_variational_params(self):
        """Initialize variational parameters including flow parameters."""
        D = self.u_latent_size
        all_layers_params = {}
        
        for layer_idx in range(self.num_flow_layers):
            master_key = random.PRNGKey(1234 + layer_idx * D)

            # Create keys for each model type and coordinate
            keys = random.split(master_key, 3 * D)
            keys = keys.reshape(3, D, 2)

            # Create dummy inputs
            dummy_input = jnp.ones((D,))
            if self.epsilon_t_input:
                dummy_input_t = jnp.ones((D * 2,))
            else:
                dummy_input_t = dummy_input
                
            if self.use_prior_info:
                dummy_input = jnp.concatenate([dummy_input, jnp.zeros((2,))])
                dummy_input_t = jnp.concatenate([dummy_input_t, jnp.zeros((2,))])
                
            # Initialize models
            def create_model(key):
                model = AutoregressiveMLP(
                    hidden_unit=self.mlp_hidden_unit,
                    output_dim=1,
                    mlp_init_std=self.mlp_init_std
                )
                params = model.init(key, dummy_input)
                return params
                
            def create_model_t(key):
                model = AutoregressiveMLP(
                    hidden_unit=self.mlp_hidden_unit,
                    output_dim=1,
                    mlp_init_std=self.mlp_init_std
                )
                params = model.init(key, dummy_input_t)
                return params
                
            # Apply vmap over coordinates
            shift_params_all = jax.vmap(create_model)(keys[0])
            scale_params_all = jax.vmap(create_model)(keys[1])
            t_params_all = jax.vmap(create_model_t)(keys[2])

            layer_params = {
                'shift': shift_params_all,
                'scale': scale_params_all,
                't': t_params_all
            }
            
            all_layers_params[f'layer_{layer_idx}'] = layer_params

        # Add base distribution parameters if training base dist
        if self.train_base_dist:
            base_params = super()._init_variational_params()
            all_layers_params['base'] = base_params

        flat_params, self.unravel_fn = ravel_pytree(all_layers_params)
        total_params = flat_params.size
        print(f"Total number of variational parameters: {total_params}")
        return all_layers_params
    
    def sample_and_log_prob(self, key, params, model=None, lambda_list=None, nu_list=None):
        """Sample from the IAF and compute log probability."""
        # Sample from base distribution
        if self.train_base_dist:
            u_current, log_q_base = super().sample_and_log_prob(
                key, params['base'], model=model, lambda_list=lambda_list, nu_list=nu_list
            )
        else:
            u_current = random.normal(key, shape=(self.u_latent_size,))
            D = self.u_latent_size
            log_const = -0.5 * D * jnp.log(2.0 * jnp.pi)
            log_q_base = -0.5 * jnp.sum(u_current**2, axis=-1) + log_const

        D = self.u_latent_size
        mlp_model = AutoregressiveMLP(
            hidden_unit=self.mlp_hidden_unit,
            output_dim=1,
            mlp_init_std=self.mlp_init_std
        )
        
        # Precompute autoregressive masks
        ar_masks = jnp.tril(jnp.ones((D, D)), k=-1)
        
        @jax.jit
        def process_layer_parallel(u_in, layer_params):
            """Process entire layer in parallel (key advantage of IAF)."""
            # Create all masked inputs in parallel
            masked_inputs = u_in[None, :] * ar_masks
            
            if self.use_prior_info:
                # Add prior information
                u_in_unflatten = model.unflatten(u_in)
                f_i_list = []
                g_i_list = []
                for j in range(len(u_in_unflatten)):
                    f_i = model.f_i(u_in_unflatten, j)
                    g_i = model.g_i(u_in_unflatten, j)
                    f_i_list.append(f_i)
                    g_i_list.append(g_i)
                f_i_flat, _ = ravel_pytree(f_i_list)
                g_i_flat, _ = ravel_pytree(g_i_list)
                
                # Add prior info to each masked input
                prior_info = jnp.stack([f_i_flat, g_i_flat], axis=1)
                masked_inputs = jnp.concatenate([masked_inputs, prior_info], axis=1)
            
            # Apply neural networks in parallel
            def apply_shift_net(i):
                shift_params_i = jax.tree.map(lambda x: x[i], layer_params["shift"])
                return mlp_model.apply(shift_params_i, masked_inputs[i])[0]
            
            def apply_scale_net(i):
                scale_params_i = jax.tree.map(lambda x: x[i], layer_params["scale"])
                raw_scale = mlp_model.apply(scale_params_i, masked_inputs[i])[0]
                return jax.nn.softplus(raw_scale) + 1e-6
            
            # Compute all shifts and scales in parallel
            shifts = jax.vmap(apply_shift_net)(jnp.arange(D))
            scales = jax.vmap(apply_scale_net)(jnp.arange(D))
            
            if self.use_t:
                def apply_t_net(i):
                    t_params_i = jax.tree.map(lambda x: x[i], layer_params["t"])
                    if self.epsilon_t_input:
                        input_t_i = jnp.concatenate([u_in, masked_inputs[i]])
                        return mlp_model.apply(t_params_i, input_t_i)[0]
                    else:
                        return mlp_model.apply(t_params_i, masked_inputs[i])[0]
                
                t_values = jax.vmap(apply_t_net)(jnp.arange(D))
                z_out = shifts + scales * (u_in - t_values)
            else:
                z_out = shifts + scales * u_in
            
            # Compute log determinant
            log_det = jnp.sum(jnp.log(scales))
            return z_out, log_det
        
        # Process all layers sequentially (but each layer is parallel)
        total_log_det = 0.0
        current_input = u_current
        
        for layer_idx in range(self.num_flow_layers):
            layer_params = params[f'layer_{layer_idx}']
            current_input, layer_log_det = process_layer_parallel(current_input, layer_params)
            total_log_det += layer_log_det
        
        z_final = current_input
        
        if self.unknown_order:
            z_final = z_final[::-1]

        # Compute final log probability
        log_q_T = log_q_base - total_log_det
        return z_final, log_q_T


# ============================================================================
# Utility Functions
# ============================================================================

def model_log_prob(model, latent, lambda_list, nu_list, variational_ncp=False):
    """
    Compute log p(z, x) for the given model.
    
    Args:
        model: The hierarchical Bayesian model
        latent: Latent variables
        lambda_list: Lambda parameters for NCP
        nu_list: Nu parameters for NCP
        variational_ncp: If True, assume latent is already transformed
        
    Returns:
        log_p_val: Log probability of the model
    """
    if variational_ncp:
        # latent corresponds to z directly (already transformed)
        z_unflatten = model.unflatten(latent)

        log_p_val = 0.0
        # Compute joint log probability
        for i in range(len(z_unflatten)):
            mu_i = model.f_i(z_unflatten[:i], i)
            sigma_i = model.g_i(z_unflatten[:i], i)
            z_i = z_unflatten[i]
            log_p_i = logpdf(z_i, loc=mu_i, scale=sigma_i)
            log_p_val += jnp.sum(log_p_i)

        # Compute likelihood for observed data
        log_p_input = model.compute_log_p_input(z_unflatten)
        log_p_val += jnp.sum(log_p_input)

        return log_p_val

    else:
        # Original model NCP scenario
        u_tilde_latent_unflatten = model.unflatten(latent)
        lambda_list_unflatten = model.unflatten(lambda_list)
        nu_list_unflatten = model.unflatten(nu_list)
        u_list = []

        log_p_val = 0

        for i in range(len(u_tilde_latent_unflatten)):
            u_tilde_i = u_tilde_latent_unflatten[i]
            lambda_i = lambda_list_unflatten[i]
            nu_i = nu_list_unflatten[i]

            mu_i = model.f_i(u_list, i)
            sigma_i = model.g_i(u_list, i)
            u_i = mu_i + (u_tilde_i - lambda_i * mu_i) * sigma_i ** (1 - nu_i)
            u_list.append(u_i)
            log_p_i = logpdf(u_tilde_i, loc=lambda_i * mu_i, scale=sigma_i**nu_i)
            log_p_val += jnp.sum(log_p_i)

        # Compute log probability for observed data
        log_p_input = model.compute_log_p_input(u_list)
        log_p_val += jnp.sum(log_p_input)

        return log_p_val

