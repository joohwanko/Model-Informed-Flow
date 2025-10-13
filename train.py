"""Train Model-Informed Flow"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
import jax
from model_informed_flow import get_model, create_variational_family, train_model

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description="Train Model-Informed Flow")
    
    # Model
    parser.add_argument("--model", type=str, default="8schools",
                       choices=["8schools", "seeds", "sonar", "ionosphere", "funnel"])
    parser.add_argument("--funnel_dim", type=int, default=10)
    
    # Variational family
    parser.add_argument("--family", type=str, default="mif", 
                       choices=["gaussian", "faf", "iaf", "mif"])
    parser.add_argument("--gaussian_param", type=str, default="mean-field", 
                       choices=["mean-field", "full-rank"])
    parser.add_argument("--ncp_method", type=str, default="CP", 
                       choices=["CP", "NCP", "VIP", "Dual-VIP"])
    
    # Flow parameters (for faf/iaf/mif)
    parser.add_argument("--num_layers", type=int, default=1,
                       help="Number of flow layers")
    parser.add_argument("--hidden_units", type=int, default=None,
                       help="Hidden units in MLP (default: 32 for faf/iaf, 0 for mif)")
    parser.add_argument("--use_prior_info", action="store_true",
                       help="Use model prior information (f_i, g_i)")
    parser.add_argument("--use_t", action="store_true",
                       help="Use translation term t_i")
    parser.add_argument("--train_base_dist", action="store_true",
                       help="Train base distribution")
    parser.add_argument("--unknown_order", action="store_true",
                       help="Reverse variable order")
    parser.add_argument("--deep_net", action="store_true",
                       help="Use deep MLP (vs linear)")
    parser.add_argument("--epsilon_t_input", action="store_true",
                       help="Use epsilon as input to t network")
    
    # Training
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10000)
    
    args = parser.parse_args()
    
    # Create model
    model = get_model(args.model, d=args.funnel_dim) if args.model == "funnel" else get_model(args.model)
    
    # Create variational family
    flow_kwargs = {}
    if args.family == "mif":
        # For MIF, only pass num_layers; MIF uses affine flow by default (no hidden units)
        # User can override with --hidden_units and --deep_net
        flow_kwargs = {
            "num_flow_layers": args.num_layers,
        }
        # Only pass if user explicitly set these
        if args.hidden_units is not None:
            flow_kwargs["mlp_hidden_unit"] = args.hidden_units
        if args.deep_net:
            flow_kwargs["deep_net"] = args.deep_net
            
    elif args.family in ["faf", "iaf"]:
        # For FAF/IAF, use default of 32 hidden units if not specified
        hidden_units = args.hidden_units if args.hidden_units is not None else 32
        flow_kwargs = {
            "num_flow_layers": args.num_layers,
            "mlp_hidden_unit": hidden_units,
            "use_prior_info": args.use_prior_info,
            "use_t": args.use_t,
            "train_base_dist": args.train_base_dist,
            "unknown_order": args.unknown_order,
            "deep_net": args.deep_net,
            "epsilon_t_input": args.epsilon_t_input,
        }
    
    variational_family = create_variational_family(
        family_type=args.family,
        gaussian_param=args.gaussian_param,
        u_latent_size=model.u_latent_size,
        ncp_distribution="variational_ncp",
        **flow_kwargs
    )
    
    # Train
    params, final_elbo, _ = train_model(
        model=model,
        variational_family=variational_family,
        ncp_method=args.ncp_method,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        seed=args.seed,
        print_every=args.print_every,
    )
    
    print(f"\nFinal ELBO: {final_elbo:.4f}")


if __name__ == "__main__":
    main()
