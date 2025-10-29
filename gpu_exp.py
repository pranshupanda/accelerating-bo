#!/usr/bin/env python3
"""
gpu_exp_oop.py

OOP, GPU-aware, vectorized acquisition optimizers (BCA, Binary PSO, Firefly)
integrated with BoTorch SingleTaskGP for acquisition optimization.
Prints timing info only every 100 iterations.
"""

import time
import torch
import numpy as np
import torch.nn.functional as F  # Added for conv1d
from time import perf_counter
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, ExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# -------------------------
# Configuration
# -------------------------
torch.set_num_threads(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
# --- OPTIMIZATION: Changed from float64 to float32 for GPU performance ---
dtype = torch.float32
torch.set_default_dtype(dtype)

CONFIG = {
    "dims": [10, 20, 30, 40, 50, 60],
    "n_trials": 3,
    "max_iters": 2000,        # BO iterations per trial (adjust)
    "init_points": 10,
    "retrain_every": 25,
    "seed_offset": 42,
    "print_every": 100,       # print frequency for iterative loops
    "optimizer_defaults": {
        "BCA": {"pop_size": 64, "r": 0.95, "beta": 3.095, "max_iters": 2000},
        "PSO": {"n_particles": 64, "w": 0.7, "c1": 1.5, "c2": 1.5, "max_iters": 2000},
        "Firefly": {"n_fireflies": 64, "alpha": 0.2, "beta0": 1.0, "gamma": 1.0, "max_iters": 2000},
    }
}

OPTIMAL_VALUES = {10: 3.8462, 20: 7.692, 30: 7.627, 40: 7.407, 50: 8.170, 60: 8.257}

# -------------------------
# Utility: safe evaluation of acquisition functions
# Handles scalar or batched outputs robustly and returns 1D tensor [batch]
# -------------------------
def safe_acq_eval(acq_func, X):
    """
    acq_func: a BoTorch acquisition module that accepts X shape (batch, dim)
    X: tensor of shape (batch, dim)
    returns: 1D tensor shape (batch,)
    """
    with torch.no_grad():
        out = acq_func(X)
        if not torch.is_tensor(out):
            out = torch.tensor(out, dtype=dtype, device=device)
        out = out.view(-1)  # flatten to 1D

        # if acquisition returns a scalar for the whole batch, replicate it
        if out.numel() == 1 and X.shape[0] > 1:
            out = out.repeat(X.shape[0])

        return out


# -------------------------
# LABS merit function (vectorized; used if running direct optimization (not necessary for acquisition optimizers),
# kept here for completeness — adapted to batch input)
# -------------------------
def labs_merit_batch(X):
    # --- OPTIMIZATION: Replaced list comprehension with conv1d ---
    # X shape: (batch, n) with binary {0,1}
    X = X.to(dtype=dtype, device=device)
    n = X.shape[1]
    x_pm = 1 - 2 * X  # map 0->1, 1->-1. Shape (batch, n)

    # Use conv1d for vectorized autocorrelation
    batch_size = X.shape[0]

    # Reshape for conv1d: (batch, C_in, L_in)
    x_in = x_pm.unsqueeze(0)

    # Kernel is x_pm itself, reshaped to (C_out, C_in/groups, L_kernel)
    # We set C_out=batch_size and groups=batch_size
    x_kernel = x_pm.view(batch_size, 1, n) # (batch, 1, n)

    # Pad to get all (n-1) correlation lags
    all_corrs = F.conv1d(x_in, x_kernel, padding=n - 1, groups=batch_size)

    # all_corrs has shape (batch, 1, 2n-1).
    # The 0-lag (C_0) is at index (n-1).
    # We want C_1 to C_{n-1}, which are at indices n to 2n-2.
    C = all_corrs[0, :, n:] # Shape (batch, n-1)

    denom = torch.sum(C * C, dim=1)
    vals = torch.where(denom > 0, (n ** 2) / (2.0 * denom), torch.zeros_like(denom))
    return vals.unsqueeze(1)  # shape (batch,1)

# -------------------------
# Base optimizer class (abstract)
# -------------------------
class AcquisitionOptimizerBase:
    def __init__(self, dim, acq_func, device=device, dtype=dtype, print_every=100):
        self.dim = dim
        self.acq_func = acq_func
        self.device = device
        self.dtype = dtype
        self.print_every = print_every

    def optimize(self):
        raise NotImplementedError

# -------------------------
# BCA optimizer (population-based, vectorized CHM)
# -------------------------
class BCAOptimizer(AcquisitionOptimizerBase):
    def __init__(self, dim, acq_func, pop_size=64, r=0.95, beta=3.095, max_iters=2000, **kwargs):
        super().__init__(dim, acq_func, **kwargs)
        self.pop_size = pop_size
        self.r = r
        self.beta = beta
        self.max_iters = max_iters

    def _mutate_population(self, X):
        # X: (pop, dim)
        # mask flips with prob r
        flip_mask = torch.rand_like(X) < self.r
        X_mut = X.clone()
        X_mut[flip_mask] = 1.0 - X_mut[flip_mask]
        return X_mut

    def optimize(self):
        # Initialize population
        X = torch.randint(0, 2, (self.pop_size, self.dim), dtype=self.dtype, device=self.device)
        vals = safe_acq_eval(self.acq_func, X)           # shape (pop,)
        best_val, best_idx = torch.max(vals, dim=0)
        best_x = X[best_idx].clone()

        t0 = perf_counter()
        for it in range(1, self.max_iters + 1):
            # generate mutated candidates
            X_mut = self._mutate_population(X)          # (pop, dim)
            # acceptance probability pi per position depends on index distance; here vectorize using uniform mask
            # We'll compute a probabilistic additional flip mask using a per-dim probability pi
            i = torch.arange(self.dim, device=self.device, dtype=self.dtype)
            # pi_k = max(min(i+1, n-i) ** (-beta), 1/n) but vectorize
            minvals = torch.minimum(i + 1.0, self.dim - i).clamp(min=1.0)
            pi = (minvals.pow(-self.beta)).clamp(min=1.0/self.dim)
            # expand pi to pop x dim and generate mask
            pi_mask = torch.rand((self.pop_size, self.dim), device=self.device) < pi.unsqueeze(0)
            X_mut[pi_mask] = 1.0 - X_mut[pi_mask]

            # Evaluate batch
            new_vals = safe_acq_eval(self.acq_func, X_mut)

            # Replace if improved or equal
            replace_mask = new_vals >= vals
            if replace_mask.any():
                X[replace_mask] = X_mut[replace_mask]
                vals[replace_mask] = new_vals[replace_mask]

            # Update global best
            local_best_val, local_best_idx = torch.max(vals, dim=0)
            if local_best_val > best_val:
                best_val = local_best_val
                best_x = X[local_best_idx].clone()

            if (it % self.print_every) == 0:
                print(f"  [BCA iter {it}] Best={best_val.item():.6f} | elapsed={perf_counter()-t0:.3f}s")

        return best_x.unsqueeze(0).to(dtype=self.dtype, device=self.device), best_val.item()

# -------------------------
# Binary PSO optimizer (vectorized)
# -------------------------
class BinaryPSOOptimizer(AcquisitionOptimizerBase):
    def __init__(self, dim, acq_func, n_particles=64, w=0.7, c1=1.5, c2=1.5, max_iters=2000, **kwargs):
        super().__init__(dim, acq_func, **kwargs)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters

    def optimize(self):
        X = torch.randint(0, 2, (self.n_particles, self.dim), dtype=self.dtype, device=self.device)
        V = torch.zeros_like(X, dtype=self.dtype, device=self.device)
        vals = safe_acq_eval(self.acq_func, X)
        p_best = X.clone()
        p_best_vals = vals.clone()
        g_best_val, g_best_idx = torch.max(p_best_vals, dim=0)
        g_best = p_best[g_best_idx].clone()

        t0 = perf_counter()
        for it in range(1, self.max_iters + 1):
            r1 = torch.rand_like(X)
            r2 = torch.rand_like(X)
            V = (self.w * V
                 + self.c1 * r1 * (p_best - X)
                 + self.c2 * r2 * (g_best.unsqueeze(0) - X))
            prob = torch.sigmoid(V)
            X = torch.where(torch.rand_like(prob) < prob, torch.ones_like(X), torch.zeros_like(X))

            vals = safe_acq_eval(self.acq_func, X)
            better = vals > p_best_vals
            if better.any():
                p_best[better] = X[better].clone()
                p_best_vals[better] = vals[better].clone()

            local_g_val, local_g_idx = torch.max(p_best_vals, dim=0)
            if local_g_val > g_best_val:
                g_best_val = local_g_val
                g_best = p_best[local_g_idx].clone()

            if (it % self.print_every) == 0:
                print(f"  [PSO iter {it}] Best={g_best_val.item():.6f} | elapsed={perf_counter()-t0:.3f}s")

        return g_best.unsqueeze(0).to(dtype=self.dtype, device=self.device), g_best_val.item()

# -------------------------
# Firefly optimizer (vectorized)
# -------------------------
class FireflyOptimizer(AcquisitionOptimizerBase):
    def __init__(self, dim, acq_func, n_fireflies=64, alpha=0.2, beta0=1.0, gamma=1.0, max_iters=2000, **kwargs):
        super().__init__(dim, acq_func, **kwargs)
        self.n_fireflies = n_fireflies
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.max_iters = max_iters

    def optimize(self):
        X = torch.randint(0, 2, (self.n_fireflies, self.dim), dtype=self.dtype, device=self.device)
        vals = safe_acq_eval(self.acq_func, X)

        t0 = perf_counter()
        for it in range(1, self.max_iters + 1):
            # For each firefly i, be attracted to fireflies with higher fitness
            # Vectorize pairwise differences
            diff = X.unsqueeze(1) - X.unsqueeze(0)   # (n, n, dim)
            dist = torch.norm(diff.float(), dim=2)  # (n, n)
            beta = self.beta0 * torch.exp(-self.gamma * dist ** 2)  # (n, n)

            # For each i, compute mask of those with higher fitness
            better_mask = (vals.unsqueeze(1) < vals.unsqueeze(0))  # (n, n) true where j better than i

            # Attraction probability per (i,j) pair (broadcast to dims)
            attract_prob = torch.sigmoid(beta).unsqueeze(2).expand(-1, -1, self.dim)  # (n,n,dim)
            random_attract = torch.rand_like(attract_prob)
            moves = (random_attract < attract_prob) & better_mask.unsqueeze(2)

            # Apply moves: for each i, if any j attracts at position k, flip that position in X[i]
            any_move = moves.any(dim=1)   # (n, dim) — True where at least one better j attracted i at pos k
            X = torch.where(any_move, 1.0 - X, X)

            # Random perturbation + rounding
            X = (X + self.alpha * torch.randn_like(X)).clamp(0.0, 1.0).round()

            vals = safe_acq_eval(self.acq_func, X)

            if (it % self.print_every) == 0:
                bestv = vals.max().item()
                print(f"  [Firefly iter {it}] Best={bestv:.6f} | elapsed={perf_counter()-t0:.3f}s")

        best_val, best_idx = torch.max(vals, dim=0)
        return X[best_idx].unsqueeze(0).to(dtype=self.dtype, device=self.device), best_val.item()

# -------------------------
# BO trial runner (OOP)
# -------------------------
class BOTrialRunner:
    def __init__(self, dim, cfg=CONFIG, use_categorical=False, optimizer_name="BCA", device=device):
        self.dim = dim
        self.cfg = cfg
        self.device = device
        self.use_categorical = use_categorical
        self.optimizer_name = optimizer_name
        self.dtype = dtype

    def _build_model(self, X, Y):
        # X, Y are already on device and dtype
        if self.use_categorical:
            covar_module = ScaleKernel(CategoricalKernel(num_categories=[2] * self.dim)).to(self.device)
            model = SingleTaskGP(X, Y, covar_module=covar_module, outcome_transform=Standardize(m=1)).to(self.device)
        else:
            model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1)).to(self.device)
        return model

    def _make_optimizer(self, acq_func):
        defaults = self.cfg["optimizer_defaults"]
        common_kwargs = {"dim": self.dim, "acq_func": acq_func, "device": self.device,
                         "dtype": self.dtype, "print_every": self.cfg["print_every"]}
        if self.optimizer_name == "BCA":
            p = defaults["BCA"]
            return BCAOptimizer(pop_size=p["pop_size"], r=p["r"], beta=p["beta"],
                                max_iters=p["max_iters"], **common_kwargs)
        elif self.optimizer_name == "PSO":
            p = defaults["PSO"]
            return BinaryPSOOptimizer(n_particles=p["n_particles"], w=p["w"], c1=p["c1"], c2=p.get("c2", 1.5), # .get for safety
                                      max_iters=p["max_iters"], **common_kwargs)
        elif self.optimizer_name == "Firefly":
            p = defaults["Firefly"]
            return FireflyOptimizer(n_fireflies=p["n_fireflies"], alpha=p["alpha"], beta0=p["beta0"],
                                    gamma=p["gamma"], max_iters=p["max_iters"], **common_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def run_trial(self, seed=None):
        cfg = self.cfg
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize X, Y (binary search space for LABS-like problems)
        X = torch.randint(0, 2, (cfg["init_points"], self.dim), dtype=self.dtype, device=self.device)
        # For BO we need a real-valued Y; using labs_merit_batch to produce Y shape (n,1)
        Y = labs_merit_batch(X).to(self.device)

        best_val = Y.max().item()
        history = [best_val]

        print(f"\n[Start BO trial] dim={self.dim}, optimizer={self.optimizer_name}, kernel={'Categorical' if self.use_categorical else 'Default'}")
        t0 = perf_counter()

        # Initial GP fit
        model = self._build_model(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        print(f"  [Init fit] {perf_counter() - t0:.3f}s")

        # BO main loop
        for t in range(cfg["max_iters"]):
            iter_start = perf_counter()
            # Retrain conditionally
            if t % cfg["retrain_every"] == 0 and t > 0:
                t_fit = perf_counter()
                model = self._build_model(X, Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                # print retrain only at large intervals (keeps output sparse)
                if (t % cfg["print_every"]) == 0:
                    print(f"  [Retrain @ iter {t}] {perf_counter() - t_fit:.3f}s")

            # Build acquisition function (we'll use qLogExpectedImprovement here for consistency with earlier code)
            # Some BoTorch acquisition modules accept 'best_f' as tensor; pass Y.max().
            try:
                acq_fn = qLogExpectedImprovement(model, best_f=Y.max())
            except Exception:
                # fallback
                acq_fn = ExpectedImprovement(model, best_f=Y.max())

            # Create optimizer for acquisition
            acq_optimizer = self._make_optimizer(acq_fn)

            # Run optimizer -> returns candidate x (1, dim) and its acq value
            new_x, acq_val = acq_optimizer.optimize()

            # Evaluate true objective (labs merit)
            new_y = labs_merit_batch(new_x).to(self.device)

            # Append
            X = torch.cat([X, new_x.to(dtype=self.dtype, device=self.device)], dim=0)
            Y = torch.cat([Y, new_y], dim=0)

            # update best
            if new_y.item() > best_val:
                best_val = new_y.item()
            history.append(best_val)

            # print only every print_every iterations
            if ((t + 1) % cfg["print_every"]) == 0:
                print(f"  Iter {t+1}/{cfg['max_iters']} | Best={best_val:.6f} | iter_time={perf_counter()-iter_start:.3f}s")

        print(f"  [Total trial time] {perf_counter() - t0:.3f}s\n")
        return {"best_val": best_val, "history": history, "final_gap": OPTIMAL_VALUES.get(self.dim, float("nan")) - best_val}

# -------------------------
# Experiment orchestration (similar to prior run_comparison)
# -------------------------
def run_comparison(cfg=CONFIG, optimizer_name="BCA"):
    results = {"default": {d: [] for d in cfg["dims"]}, "categorical": {d: [] for d in cfg["dims"]}}
    for trial in range(cfg["n_trials"]):
        for d in cfg["dims"]:
            seed = cfg["seed_offset"] + trial * 1000 + d
            # Default kernel
            runner_def = BOTrialRunner(dim=d, cfg=cfg, use_categorical=False, optimizer_name=optimizer_name)
            res_def = runner_def.run_trial(seed=seed)
            # Categorical kernel
            runner_cat = BOTrialRunner(dim=d, cfg=cfg, use_categorical=True, optimizer_name=optimizer_name)
            res_cat = runner_cat.run_trial(seed=seed)
            print(f"Trial {trial + 1}/{cfg['n_trials']} dim={d}: Default={res_def['best_val']:.6f}, Categorical={res_cat['best_val']:.6f}")
            results["default"][d].append(res_def)
            results["categorical"][d].append(res_cat)
    # Summarize (simple)
    print("\n" + "=" * 60)
    print("AVERAGED RESULTS")
    print("=" * 60)
    for d in cfg["dims"]:
        def_vals = [r["best_val"] for r in results["default"][d]]
        cat_vals = [r["best_val"] for r in results["categorical"][d]]
        print(f"\nDim {d}: Default mean={np.mean(def_vals):.4f} ± {np.std(def_vals):.4f}, "
              f"Categorical mean={np.mean(cat_vals):.4f} ± {np.std(cat_vals):.4f}")
    return results

# -------------------------
# Main guard
# -------------------------
if __name__ == "__main__":
    
    optimizers_to_test = ["BCA", "PSO", "Firefly"]
    
    for opt_name in optimizers_to_test:
        print("\n" + "="*80)
        print(f"STARTING EXPERIMENT RUN FOR OPTIMIZER: {opt_name.upper()}")
        print("="*80 + "\n")
        
        start = time.time()
        results = run_comparison(CONFIG, optimizer_name=opt_name)
        print(f"\nTotal wall time for {opt_name}: {time.time() - start:.1f}s on device={device.upper()}")