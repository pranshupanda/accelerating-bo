# Visualization
# Credit: Prof Aryan Deshwal

import os
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood,
)

from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.kernels import ScaleKernel
from botorch.models.kernels.categorical import CategoricalKernel

# CLI

parser = argparse.ArgumentParser(
    description="GP visualization for LABS objective"
)

parser.add_argument(
    "--kernel",
    type=str,
    default="rbf",
    choices=["rbf", "categorical"],
    help="Kernel type",
)

parser.add_argument(
    "--n",
    type=int,
    default=10,
    help="Input dimensionality",
)

parser.add_argument(
    "--train-size",
    type=int,
    default=20,
    help="Number of training points",
)

parser.add_argument(
    "--test-size",
    type=int,
    default=20,
    help="Number of test points",
)

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed",
)

parser.add_argument(
    "--binary",
    action="store_true",
    help="Use binary inputs instead of continuous random inputs",
)

parser.add_argument(
    "--normalize-input",
    action="store_true",
    help="Apply Normalize transform to inputs",
)

parser.add_argument(
    "--standardize-output",
    action="store_true",
    help="Apply Standardize transform to outputs",
)

parser.add_argument(
    "--dpi",
    type=int,
    default=300,
    help="PNG DPI",
)

parser.add_argument(
    "--output-dir",
    type=str,
    default="plots",
    help="Directory to save plots",
)

parser.add_argument(
    "--output-name",
    type=str,
    default=None,
    help="Output PNG filename",
)

args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.output_dir, exist_ok=True)


def cv_plot(q_middle, q_lower, q_upper, Y_true, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    min_val = (torch.minimum(Y_true, q_lower)).min()
    max_val = (torch.maximum(Y_true, q_upper)).max()

    min_val = min_val - 0.1 * (max_val - min_val)
    max_val = max_val + 0.1 * (max_val - min_val)

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "b--",
        lw=2,
    )

    yerr1 = q_middle - q_lower
    yerr2 = q_upper - q_middle

    yerr = torch.cat(
        (
            yerr1.unsqueeze(0),
            yerr2.unsqueeze(0),
        ),
        dim=0,
    ).squeeze(-1)

    markers, caps, bars = ax.errorbar(
        Y_true.cpu().numpy(),
        q_middle.cpu().numpy(),
        yerr=yerr.cpu().numpy(),
        fmt=".",
        capsize=4,
        elinewidth=2.0,
        ms=14,
        c="k",
        ecolor="gray",
    )

    [bar.set_alpha(0.8) for bar in bars]
    [cap.set_alpha(0.8) for cap in caps]

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    ax.grid(True)


def cv_plots(gp_model, test_x, test_y):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    posterior = gp_model.posterior(test_x)

    mean = posterior.mean.detach().squeeze()
    var = posterior.variance.detach().squeeze()

    q1 = mean - 1.96 * var.sqrt()
    q2 = mean + 1.96 * var.sqrt()

    cv_plot(
        q_middle=mean,
        q_lower=q1,
        q_upper=q2,
        Y_true=test_y,
        ax=ax,
    )

    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Predictions")

    return fig


def labs_merit(x):
    total = 0

    for k in range(1, len(x)):
        a_k = torch.sum(
            (2 * x[:-k] - 1)
            * (2 * x[k:] - 1)
        )

        total += a_k ** 2

    return total.item()


def sample(n, num_points, binary=False):
    if binary:
        x = torch.randint(
            0,
            2,
            (num_points, n),
            dtype=torch.double,
        )
    else:
        x = torch.rand(
            num_points,
            n,
            dtype=torch.double,
        )

    y = torch.tensor(
        [[labs_merit(xi)] for xi in x],
        dtype=torch.double,
    )

    return x, y


train_x, train_y = sample(
    args.n,
    args.train_size,
    binary=args.binary,
)

test_x, test_y = sample(
    args.n,
    args.test_size,
    binary=args.binary,
)


model_kwargs = {}

# Input transform
if args.normalize_input:
    model_kwargs["input_transform"] = Normalize(
        d=train_x.shape[-1]
    )

# Output transform
if args.standardize_output:
    model_kwargs["outcome_transform"] = Standardize(
        m=1
    )

# Kernel selection
if args.kernel == "categorical":
    covar_module = ScaleKernel(
        base_kernel=CategoricalKernel(
            ard_num_dims=args.n
        )
    )

    model_kwargs["covar_module"] = covar_module

# Build model
model = SingleTaskGP(
    train_x,
    train_y,
    **model_kwargs,
)

mll = ExactMarginalLogLikelihood(
    model.likelihood,
    model,
)

fit_gpytorch_mll(mll)

fig = cv_plots(
    model,
    test_x,
    test_y,
)

if args.output_name is None:
    filename = f"{args.kernel}_kernel_results.png"
else:
    filename = args.output_name

output_path = os.path.join(
    args.output_dir,
    filename,
)

fig.savefig(
    output_path,
    dpi=args.dpi,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved plot to: {output_path}")