import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from time import time

# ====================================================
# GLOBALS
# ====================================================
OPTIMAL_VALUES = {10: 3.8462, 20: 7.692, 30: 7.627, 40: 7.407}
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================================
# LABS MERIT FUNCTION (GPU)
# ====================================================
def labs_merit(x):
    x = x.squeeze()
    if x.ndim != 1:
        raise ValueError(f"Input must be 1D, got shape {x.shape}")
    n = x.shape[0]
    if n < 2:
        return torch.tensor([[0.0]], dtype=torch.float64, device=device)

    x_pm = 1 - 2 * x
    C = torch.tensor([torch.dot(x_pm[:n - k], x_pm[k:]) for k in range(1, n)], dtype=torch.float64, device=device)
    denom = torch.sum(C * C)
    val = n ** 2 / (2 * denom) if denom > 0 else 0.0
    return torch.tensor([[val]], dtype=torch.float64, device=device)

# ====================================================
# FAST CHM + BCA (GPU)
# ====================================================
def fast_chm(x, r, f, beta):
    x_mutated = x.clone()
    best = f(x.unsqueeze(0)).item()
    y = x.clone()
    n = x.shape[0]
    a = torch.randint(0, n, (1,), device=device).item()

    for i in range(n):
        if torch.rand(1, device=device).item() < r:
            idx = (a + i) % n
            x_mutated[idx] = 1.0 - x_mutated[idx]
        pi = max(min(i + 1, n - i) ** (-beta), 1 / n)
        if torch.rand(1, device=device).item() < pi:
            curr = f(x_mutated.unsqueeze(0)).item()
            if curr >= best:
                best = curr
                y = x_mutated.clone()
    return y, best

def fast_bca(fitness, n, r, beta, max_iters):
    x = torch.randint(0, 2, (n,), dtype=torch.float64, device=device)
    fofx = fitness(x.unsqueeze(0)).item()
    for _ in range(max_iters):
        y, fofy = fast_chm(x, r, fitness, beta=beta)
        if fofy >= fofx:
            x, fofx = y, fofy
    return x.unsqueeze(0), fofx

def bca_acq_optimizer(acq_func, dim, r=0.9, beta=3.095, max_iters=30, restarts=20):
    best_x, best_val = None, float("-inf")
    for _ in range(restarts):
        x_try, val = fast_bca(acq_func, dim, r=r, beta=beta, max_iters=max_iters)
        if val > best_val:
            best_x, best_val = x_try, val
    return best_x.to(dtype=torch.float64, device=device)

# ====================================================
# BO TRIAL (GPU)
# ====================================================
def run_bo_trial(dim, init_points=10, seed=None, max_iters=100, retrain_every=10, use_categorical=False):
    if seed is not None:
        torch.manual_seed(seed)

    X = torch.randint(0, 2, (init_points, dim), dtype=torch.float64, device=device)
    Y = torch.cat([labs_merit(x.unsqueeze(0)) for x in X], dim=0)
    best_val = Y.max().item()
    history = [best_val]

    # Initialize GP model
    if use_categorical:
        covar_module = ScaleKernel(CategoricalKernel(num_categories=[2] * dim)).to(device)
        model = SingleTaskGP(X, Y, covar_module=covar_module, outcome_transform=Standardize(m=1)).to(device)
    else:
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1)).to(device)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    for t in range(max_iters):
        if t % retrain_every == 0 and t > 0:
            if use_categorical:
                covar_module = ScaleKernel(CategoricalKernel(num_categories=[2] * dim)).to(device)
                model = SingleTaskGP(X, Y, covar_module=covar_module, outcome_transform=Standardize(m=1)).to(device)
            else:
                model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1)).to(device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

        acq_func = qLogExpectedImprovement(model, best_f=Y.max())
        new_x = bca_acq_optimizer(acq_func, dim, max_iters=30, restarts=3)
        new_y = labs_merit(new_x)
        X = torch.cat([X, new_x], dim=0)
        Y = torch.cat([Y, new_y], dim=0)

        if new_y.item() > best_val:
            best_val = new_y.item()
        history.append(best_val)

    return {"best_val": best_val, "history": history, "final_gap": OPTIMAL_VALUES[dim] - best_val}

# ====================================================
# RESULTS SUMMARY (GPU)
# ====================================================
def summarize_results(results, dims):
    print("\n" + "=" * 80)
    print("AVERAGED RESULTS ACROSS TRIALS")
    print("=" * 80)

    for d in dims:
        default_vals = [r["best_val"] for r in results["default"][d]]
        categorical_vals = [r["best_val"] for r in results["categorical"][d]]
        default_gaps = [r["final_gap"] for r in results["default"][d]]
        categorical_gaps = [r["final_gap"] for r in results["categorical"][d]]

        avg_default = np.mean(default_vals)
        std_default = np.std(default_vals)
        avg_categorical = np.mean(categorical_vals)
        std_categorical = np.std(categorical_vals)

        avg_gap_default = np.mean(default_gaps)
        avg_gap_categorical = np.mean(categorical_gaps)

        print(f"\nDimension {d} (Optimal: {OPTIMAL_VALUES[d]:.4f}):")
        print(f"  Default Kernel:")
        print(f"    Mean Merit Factor: {avg_default:.4f} ± {std_default:.4f}")
        print(f"    Mean Gap to Optimal: {avg_gap_default:.4f}")
        print(f"  Categorical Kernel:")
        print(f"    Mean Merit Factor: {avg_categorical:.4f} ± {std_categorical:.4f}")
        print(f"    Mean Gap to Optimal: {avg_gap_categorical:.4f}")
        print(f"  Improvement: {avg_categorical - avg_default:.4f} "
              f"({(avg_categorical - avg_default) / avg_default * 100:.2f}%)")

# ====================================================
# MAIN EXPERIMENT LOOP (GPU)
# ====================================================
def run_comparison(dims=[10, 20, 30, 40], n_trials=5, max_iters=100):
    results = {"default": {d: [] for d in dims}, "categorical": {d: [] for d in dims}}

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        for d in dims:
            seed = 42 + trial * 1000 + d
            res_default = run_bo_trial(dim=d, seed=seed, max_iters=max_iters, use_categorical=False)
            res_categorical = run_bo_trial(dim=d, seed=seed, max_iters=max_iters, use_categorical=True)

            results["default"][d].append(res_default)
            results["categorical"][d].append(res_categorical)

            print(f"  dim={d}: Default={res_default['best_val']:.4f}, "
                  f"Categorical={res_categorical['best_val']:.4f}")

    summarize_results(results, dims)
    return results

# ====================================================
# ENTRY POINT
# ====================================================
if __name__ == "__main__":
    start = time()
    results = run_comparison(dims=[10, 20, 30, 40], n_trials=5, max_iters=100)
    print(f"\nTotal time: {time() - start:.1f} seconds on {device.upper()}")
