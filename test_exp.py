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
from concurrent.futures import ProcessPoolExecutor, as_completed

OPTIMAL_VALUES = {10: 3.8462, 20: 7.692, 30: 7.627, 40: 7.407}

def labs_merit(x):
    x_np = np.round(x.squeeze().cpu().numpy()).astype(int)
    if x_np.ndim != 1:
        raise ValueError(f"Input must be 1D, got shape {x_np.shape}")
    n = len(x_np)
    if n < 2:
        return torch.tensor([[0.0]], dtype=torch.float64)

    x_pm = 1 - 2 * x_np
    # Vectorized autocorrelation computation
    C = np.array([np.dot(x_pm[:n - k], x_pm[k:]) for k in range(1, n)])
    denom = np.sum(C * C)
    val = n ** 2 / (2 * denom) if denom > 0 else 0.0
    return torch.tensor([[val]], dtype=torch.float64)

def fast_chm(x, r, f, beta):
    x_mutated = x.clone()
    best = f(x.unsqueeze(0)).item()
    y = x.clone()
    n = len(x)
    a = torch.randint(0, n, (1,)).item()
    for i in range(n):
        if torch.rand(1).item() < r:
            idx = (a + i) % n
            x_mutated[idx] = 1.0 - x_mutated[idx]
        pi = max(min(i + 1, n - i) ** (-beta), 1 / n)
        if torch.rand(1).item() < pi:
            curr = f(x_mutated.unsqueeze(0)).item()
            if curr >= best:
                best = curr
                y = x_mutated.clone()
    return y, best

def fast_bca(fitness, n, r, beta, max_iters, device="cpu"):
    x = torch.randint(0, 2, (n,), dtype=torch.float64, device=device)
    fofx = fitness(x.unsqueeze(0)).item()
    for _ in range(max_iters):
        y, fofy = fast_chm(x, r, fitness, beta=beta)
        if fofy >= fofx:
            x, fofx = y, fofy
    return x.unsqueeze(0), fofx

def bca_acq_optimizer(acq_func, dim, r=0.9, beta=3.095053400066849, max_iters=30, restarts=20, device="cpu"):
    best_x, best_val = None, float("-inf")
    for _ in range(restarts):
        x_try, val = fast_bca(acq_func, dim, r=r, beta=beta, max_iters=max_iters, device=device)
        if val > best_val:
            best_x, best_val = x_try, val
    return best_x.to(dtype=torch.float64)

def run_bo_trial(dim, init_points=10, device="cpu", seed=None, max_iters=100, 
                 retrain_every=10, use_categorical=False):
    if seed is not None:
        torch.manual_seed(seed)
    
    X = torch.randint(0, 2, (init_points, dim), dtype=torch.float64, device=device)
    Y = torch.cat([labs_merit(x.unsqueeze(0)) for x in X], dim=0)
    best_val = Y.max().item()
    history = [best_val]
    
    # Build model with or without categorical kernel
    if use_categorical:
        covar_module = ScaleKernel(CategoricalKernel(num_categories=[2] * dim))
        model = SingleTaskGP(X, Y, covar_module=covar_module, outcome_transform=Standardize(m=1))
    else:
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    for t in range(max_iters):
        if t % retrain_every == 0 and t > 0:
            if use_categorical:
                covar_module = ScaleKernel(CategoricalKernel(num_categories=[2] * dim))
                model = SingleTaskGP(X, Y, covar_module=covar_module, outcome_transform=Standardize(m=1))
            else:
                model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

        acq_func = qLogExpectedImprovement(model, best_f=Y.max())
        new_x = bca_acq_optimizer(acq_func, dim, max_iters=30, restarts=3, device=device)
        new_y = labs_merit(new_x)
        X = torch.cat([X, new_x], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        if new_y.item() > best_val:
            best_val = new_y.item()
        history.append(best_val)

    return {"best_val": best_val, "history": history, "final_gap": OPTIMAL_VALUES[dim] - best_val}

def run_comparison(dims=[10, 20, 30, 40], n_trials=10, max_iters=150, parallel=True):
    results = {
        "default": {d: [] for d in dims},
        "categorical": {d: [] for d in dims}
    }
    
    print("Running comparison across 10 trials...")
    print("=" * 80)
    
    if parallel:
        # Parallel execution
        tasks = []
        with ProcessPoolExecutor() as executor:
            for trial in range(n_trials):
                for d in dims:
                    seed = 42 + trial * 1000 + d
                    tasks.append((executor.submit(run_bo_trial, d, 10, "cpu", seed, max_iters, 10, False), 
                                  d, trial, "default"))
                    tasks.append((executor.submit(run_bo_trial, d, 10, "cpu", seed, max_iters, 10, True), 
                                  d, trial, "categorical"))
            
            completed = 0
            total = len(tasks)
            for future, d, trial, kernel_type in tasks:
                res = future.result()
                results[kernel_type][d].append(res)
                completed += 1
                if completed % 8 == 0:
                    print(f"Progress: {completed}/{total} runs completed")
    else:
        # Sequential execution (original)
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            for d in dims:
                seed = 42 + trial * 1000 + d
                res_default = run_bo_trial(dim=d, seed=seed, max_iters=max_iters, use_categorical=False)
                results["default"][d].append(res_default)
                
                res_categorical = run_bo_trial(dim=d, seed=seed, max_iters=max_iters, use_categorical=True)
                results["categorical"][d].append(res_categorical)
                
                print(f"  dim={d}: Default={res_default['best_val']:.4f}, Categorical={res_categorical['best_val']:.4f}")
    
    # Calculate and print averages
    print("\n" + "=" * 80)
    print("AVERAGED RESULTS ACROSS 10 TRIALS")
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
        print(f"  Improvement: {avg_categorical - avg_default:.4f} ({(avg_categorical - avg_default) / avg_default * 100:.2f}%)")
    
    return results

if __name__ == "__main__":
    start = time()
    results = run_comparison(dims=[10, 20, 30, 40], n_trials=10, max_iters=150, parallel=True)
    print(f"\nTotal time: {time() - start:.1f} seconds")