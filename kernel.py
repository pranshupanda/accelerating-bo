# Visualization
# Credit: Prof Aryan Deshwal 

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel
from botorch.models.kernels.categorical import CategoricalKernel
import os 

os.makedirs("plots", exist_ok=True)

def cv_plot(q_middle, q_lower, q_upper, Y_true, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    min_val = (torch.minimum(Y_true, q_lower)).min()
    max_val = (torch.maximum(Y_true, q_upper)).max()
    min_val, max_val = min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val)
    ax.plot([min_val, max_val], [min_val, max_val], "b--", lw=2)

    yerr1, yerr2 = q_middle - q_lower, q_upper - q_middle
    yerr = torch.cat((yerr1.unsqueeze(0), yerr2.unsqueeze(0)), dim=0).squeeze(-1)
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
#    ax.set_xlabel("True value", fontsize=24)
#    ax.set_ylabel("Predicted value", fontsize=24)
    ax.grid(True)

def cv_plots(gp_model, test_x, test_y):
    mean = gp_model.posterior(test_x).mean
    var = gp_model.posterior(test_x).variance
    mean = mean.detach().squeeze()
    var = var.detach().squeeze()
    q1 = mean - 1.96 * var.sqrt()
    q2 = mean + 1.96 * var.sqrt()
    cv_plot(q_middle=mean, q_lower=q1, q_upper=q2, Y_true = test_y)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")

# Default GP run - RBF Kernel

# LABS Objective
def labs_merit(x):
    total = 0
    for k in range(1, len(x)):
        a_k = torch.sum((2 * x[:-k] - 1) * (2 * x[k:] - 1))
        total += a_k ** 2
    return total.item()

def sample(n, num_points):
    x = torch.rand(num_points, n)
    y = torch.tensor([[labs_merit(xi)] for xi in x])
    return x, y

n = 5
train_x, train_y = sample(n, 10)
test_x, test_y = sample(n, 10)

train_x = train_x.double()
train_y = train_y.double()
test_x = test_x.double()
test_y = test_y.double()

y_mean = train_y.mean()
y_std = train_y.std()
train_y_norm = (train_y - y_mean) / y_std

model = SingleTaskGP(train_x, train_y, input_transform=Normalize(d=n))
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

fig1 = cv_plots(model, test_x, test_y)
fig1.savefig("plots/rbf_kernel_results.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

# Run with Categorical Kernel (https://archive.botorch.org/api/_modules/botorch/models/kernels/categorical.html#CategoricalKernel)

n = 5
np.random.seed(1)
train_x, train_y = sample(n, 5)
test_x, test_y = sample(n, 5)


covar_module = ScaleKernel(
    base_kernel=CategoricalKernel(ard_num_dims=n)
)

model = SingleTaskGP(
    train_x,
    train_y,
    covar_module=covar_module,
    input_transform=Normalize(d=train_x.shape[-1]),  
    outcome_transform=Standardize(m=1),              
)


mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
fig2 = cv_plots(model, test_x, test_y)
fig2.savefig("plots/categorical_kernel_results.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
