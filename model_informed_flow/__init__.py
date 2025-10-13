"""
Model-Informed Flow for Bayesian Inference

A framework for variational inference using normalizing flows informed by
hierarchical Bayesian model structure.
"""

from .models import (
    HierarchicalBayesianModel,
    EightSchools,
    SeedsModel,
    SonarModel,
    IonosphereModel,
    FunnelModel,
    get_model,
    MODEL_REGISTRY,
)

from .variational_families import (
    VariationalFamily,
    ForwardAutoregressiveFlow,
    InverseAutoregressiveFlow,
    model_log_prob,
)

from .training import (
    train_model,
    create_variational_family,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "HierarchicalBayesianModel",
    "EightSchools",
    "SeedsModel",
    "SonarModel",
    "IonosphereModel",
    "FunnelModel",
    "get_model",
    "MODEL_REGISTRY",
    # Variational Families
    "VariationalFamily",
    "ForwardAutoregressiveFlow",
    "InverseAutoregressiveFlow",
    "model_log_prob",
    # Training
    "train_model",
    "create_variational_family",
]

