"""
Hierarchical Bayesian Models for Bayesian Inference

This module contains implementations of various hierarchical Bayesian models
that can be used with the Model-Informed Flow framework.
"""

import jax
import jax.numpy as jnp
from jax import flatten_util
from jax.scipy.stats.norm import logpdf
import numpy as np
import pickle


# ============================================================================
# Helper Functions
# ============================================================================

def bernoulli_logpmf(k, logits):
    """Bernoulli log probability mass function using logits."""
    return k * jax.nn.log_sigmoid(logits) + (1 - k) * jax.nn.log_sigmoid(-logits)


def binomial_logpmf(k, n, logits):
    """Binomial log probability mass function using logits."""
    log_p = -jax.nn.softplus(-logits)
    log_1_minus_p = -jax.nn.softplus(logits)
    log_binom_coeff = jax.scipy.special.gammaln(n + 1) - jax.scipy.special.gammaln(k + 1) - jax.scipy.special.gammaln(n - k + 1)
    return log_binom_coeff + k * log_p + (n - k) * log_1_minus_p


def pad_with_const(X):
    """Add a constant column of ones for bias term."""
    extra = np.ones((X.shape[0], 1))
    return np.hstack([extra, X])


def standardize_and_pad(X):
    """Standardize features and add bias term."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.
    X = (X - mean) / std
    return pad_with_const(X)


# ============================================================================
# Base Model Class
# ============================================================================

class HierarchicalBayesianModel:
    """
    Base class for hierarchical Bayesian models.
    
    All models should inherit from this class and implement:
    - f_i: prior mean function
    - g_i: prior standard deviation function
    - compute_mu_input: mean function for observations
    - compute_sigma_input: standard deviation for observations (optional)
    - compute_log_p_input: log probability of observations
    """
    
    def __init__(self, name: str):
        self.name = name
        self.N = None  # Number of observations
        self.u_latent_size = None  # Number of latent variables
        self.u_size = None  # Total variables (latent + observations)
        print(f"Initialized {name} model")
    
    def f_i(self, prev_u, i):
        """Prior mean for latent variable i given previous variables."""
        raise NotImplementedError
    
    def g_i(self, prev_u, i):
        """Prior standard deviation for latent variable i given previous variables."""
        raise NotImplementedError


# ============================================================================
# Model Implementations
# ============================================================================

class EightSchools(HierarchicalBayesianModel):
    """
    Eight Schools model (Rubin, 1981).
    
    A hierarchical model for estimating the effect of coaching programs
    on SAT scores in eight schools.
    """
    
    def __init__(self):
        super().__init__("8schools")
        
        # Observed data from the eight schools study
        self.u_input = jnp.array([28., 8., -3., 7., -1., 1., 18., 12.])
        self.sigma_input = jnp.array([15., 10., 16., 11., 9., 11., 10., 18.])
        self.N = 8
        
        # Latent variables: [log_tau, mu, theta_1, ..., theta_8]
        self.u_latent_size = 2 + self.N
        self.u_size = self.u_latent_size + self.N
        _, self.unflatten = flatten_util.ravel_pytree([0.0, 0.0, jnp.zeros(self.N)])
        
    def f_i(self, prev_u, i):
        """Prior mean function."""
        if i == 0:
            # log_tau
            return 0.0
        elif i == 1:
            # mu
            return 0.0
        elif i == 2:
            # theta_i depends on mu
            mu = prev_u[1]
            return jnp.repeat(mu, self.N)
        else:
            raise IndexError("Index out of bounds in f_i")
    
    def g_i(self, prev_u, i):
        """Prior standard deviation function."""
        if i == 0:
            # log_tau
            return 5.0
        elif i == 1:
            # mu
            return 5.0
        elif i == 2:
            # theta_i depends on tau
            log_tau = prev_u[0]
            tau = jnp.exp(log_tau)
            return jnp.repeat(tau, self.N)
        else:
            raise IndexError("Index out of bounds in g_i")
    
    def compute_mu_input(self, u_list):
        """Mean for observations."""
        theta = jnp.array(u_list[2])
        return theta
    
    def compute_sigma_input(self, u_list):
        """Standard deviation for observations."""
        return self.sigma_input
    
    def compute_log_p_input(self, u_list):
        """Log probability of observations."""
        mu_input = self.compute_mu_input(u_list)
        sigma_input = self.compute_sigma_input(u_list)
        log_p_input = logpdf(self.u_input, loc=mu_input, scale=sigma_input)
        return log_p_input


class SeedsModel(HierarchicalBayesianModel):
    """
    Seeds germination model.
    
    Hierarchical binomial regression for seed germination data.
    """
    
    def __init__(self):
        super().__init__("Seeds")

        # Seed germination data
        data = {
            "R": [10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3],
            "N": [39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7],
            "X1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "X2": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "tot": 21
        }

        self.R = jnp.array(data['R'])
        self.N = jnp.array(data['N'])
        self.X1 = jnp.array(data['X1'])
        self.X2 = jnp.array(data['X2'])
        self.tot = data['tot']

        self.u_input = self.R

        # Latent variables: [log_tau, a_0, a_1, a_2, a_12, b[tot]]
        self.u_latent_size = 5 + self.tot
        self.u_size = self.u_latent_size + self.tot
        _, self.unflatten = flatten_util.ravel_pytree([0.0, 0.0, 0.0, 0.0, 0.0, jnp.zeros(self.tot)])
        
    def f_i(self, prev_u, i):
        """Prior mean function."""
        if i in [0, 1, 2, 3, 4]:
            return 0.0
        elif i == 5:
            return jnp.zeros(self.tot)
        else:
            raise IndexError("Index out of bounds in f_i")

    def g_i(self, prev_u, i):
        """Prior standard deviation function."""
        if i == 0:
            # log_tau
            return 10.0
        elif i in [1, 2, 3, 4]:
            # a_0, a_1, a_2, a_12
            return 10.0
        elif i == 5:
            # b_i depends on tau
            log_tau = prev_u[0]
            sigma_b = jnp.exp(-0.5 * log_tau)
            return jnp.repeat(sigma_b, self.tot)
        else:
            raise IndexError("Index out of bounds in g_i")

    def compute_log_p_input(self, u_list):
        """Log probability of observations."""
        a_0 = u_list[1]
        a_1 = u_list[2]
        a_2 = u_list[3]
        a_12 = u_list[4]
        b = u_list[5]
        logits = a_0 + a_1 * self.X1 + a_2 * self.X2 + a_12 * self.X1 * self.X2 + b
        log_p_input = binomial_logpmf(self.u_input, self.N, logits)
        return log_p_input


class SonarModel(HierarchicalBayesianModel):
    """
    Sonar classification model.
    
    Logistic regression for sonar signal classification (rocks vs mines).
    """
    
    def __init__(self):
        super().__init__("Sonar")

        # Load sonar data
        with open('data/sonar_full.pkl', 'rb') as f:
            X, Y = pickle.load(f)
        Y = (Y + 1) // 2  # Convert labels to 0 and 1
        X = standardize_and_pad(X)

        self.X = jnp.array(X)
        self.Y = jnp.array(Y)
        self.n_data = self.X.shape[0]
        self.dim = self.X.shape[1]

        self.u_input = self.Y

        # Latent variables: weights
        self.u_latent_size = self.dim
        self.u_size = self.u_latent_size + self.n_data
        _, self.unflatten = flatten_util.ravel_pytree([jnp.zeros(self.dim)])
        
    def f_i(self, prev_u, i):
        """Prior mean function."""
        if i == 0:
            return jnp.zeros(self.dim)
        else:
            raise IndexError("Index out of bounds in f_i")

    def g_i(self, prev_u, i):
        """Prior standard deviation function."""
        if i == 0:
            return jnp.ones(self.dim)
        else:
            raise IndexError("Index out of bounds in g_i")

    def compute_log_p_input(self, u_list):
        """Log probability of observations."""
        w = u_list[0]
        logits = jnp.dot(self.X, w)
        log_p_input = bernoulli_logpmf(self.u_input, logits)
        return log_p_input


class IonosphereModel(HierarchicalBayesianModel):
    """
    Ionosphere classification model.
    
    Logistic regression for ionosphere radar signal classification.
    """
    
    def __init__(self):
        super().__init__("Ionosphere")

        # Load ionosphere data
        with open('data/ionosphere_full.pkl', 'rb') as f:
            X, Y = pickle.load(f)
        Y = (Y + 1) // 2  # Convert labels to binary (0 and 1)
        X = standardize_and_pad(X)

        self.X = jnp.array(X)
        self.Y = jnp.array(Y)
        self.n_data = self.X.shape[0]
        self.dim = self.X.shape[1]

        self.u_input = self.Y

        # Latent variables: weights
        self.u_latent_size = self.dim
        self.u_size = self.u_latent_size + self.n_data
        _, self.unflatten = flatten_util.ravel_pytree([jnp.zeros(self.dim)])
        
    def f_i(self, prev_u, i):
        """Prior mean function."""
        if i == 0:
            return jnp.zeros(self.dim)
        else:
            raise IndexError("Index out of bounds in f_i")

    def g_i(self, prev_u, i):
        """Prior standard deviation function."""
        if i == 0:
            return jnp.ones(self.dim)
        else:
            raise IndexError("Index out of bounds in g_i")

    def compute_log_p_input(self, u_list):
        """Log probability of observations."""
        w = u_list[0]
        logits = jnp.dot(self.X, w)
        log_p_input = bernoulli_logpmf(self.u_input, logits)
        return log_p_input


class FunnelModel(HierarchicalBayesianModel):
    """
    Funnel distribution (Neal, 2003).
    
    A challenging distribution with strong dependencies:
        x_1 ~ Normal(0, sigma_f^2)
        x_i ~ Normal(0, exp(x_1)) for i > 1
    
    This creates a funnel-shaped distribution that is difficult to sample from.
    """

    def __init__(self, d=10, sigma_f=3.0):
        """
        Args:
            d: Total dimension (default=10). Must be >= 2.
            sigma_f: Standard deviation for x_1 (default=3.0).
        """
        super().__init__(f"funnel_d{d}")
        
        if d < 2:
            raise ValueError("Funnel dimension must be >= 2 for a funnel shape.")

        self.d = d
        self.sigma_f = sigma_f

        # No observed data
        self.u_input = jnp.array([])
        self.u_latent_size = d
        self.u_size = self.u_latent_size

        # For flatten/unflatten
        example_tree = [0.0] * d
        _, self.unflatten = flatten_util.ravel_pytree(example_tree)

    def f_i(self, prev_u, i):
        """Prior mean function (always 0)."""
        return 0.0

    def g_i(self, prev_u, i):
        """Prior standard deviation function."""
        if i == 0:
            # x_1 ~ Normal(0, sigma_f)
            return self.sigma_f
        else:
            # x_i ~ Normal(0, exp(x_1))
            x1_val = prev_u[0]
            return jnp.exp(x1_val / 2.0)

    def compute_mu_input(self, u_list):
        """No observations."""
        return jnp.array([])

    def compute_sigma_input(self, u_list):
        """No observations."""
        return jnp.array([])

    def compute_log_p_input(self, u_list):
        """No observations, so log-likelihood is 0."""
        return jnp.array([])


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {
    "8schools": EightSchools,
    "seeds": SeedsModel,
    "sonar": SonarModel,
    "ionosphere": IonosphereModel,
    "funnel": FunnelModel,
}


def get_model(name: str, **kwargs):
    """
    Factory function to get a model by name.
    
    Args:
        name: Name of the model (e.g., "8schools", "seeds", "sonar", "ionosphere", "funnel")
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        An instance of the requested model
        
    Raises:
        ValueError: If the model name is not recognized
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    
    return MODEL_REGISTRY[name](**kwargs)

