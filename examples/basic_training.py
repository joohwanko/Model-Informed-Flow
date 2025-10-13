"""Basic training example"""
import jax
from model_informed_flow import get_model, create_variational_family, train_model

jax.config.update("jax_enable_x64", True)

# Create model
model = get_model("8schools")

# Train with CP
print("Training with CP...")
vf_cp = create_variational_family("gaussian", "mean-field", model.u_latent_size, "variational_ncp")
_, elbo_cp, _ = train_model(model, vf_cp, "CP", num_steps=100000, print_every=10000)

# Train with VIP
print("\nTraining with VIP...")
vf_vip = create_variational_family("gaussian", "mean-field", model.u_latent_size, "variational_ncp")
params, elbo_vip, _ = train_model(model, vf_vip, "VIP", num_steps=100000, print_every=10000)

# Compare
print(f"\nCP ELBO:  {elbo_cp:.4f}")
print(f"VIP ELBO: {elbo_vip:.4f}")
print(f"Improvement: {elbo_vip - elbo_cp:.4f}")
