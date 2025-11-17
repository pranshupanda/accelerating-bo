import time
import csv
import torch
import numpy as np
from tqdm import tqdm
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# max_iters - max BO iterations 
# retrain_every - number of iterations to refit GP after
# # max_iters for BCA, PSO and Firefly - max iterations for each acquisition function optimization

CONFIG = {
    "dims": [10, 20, 30, 40, 50, 60],
    "n_trials": 10,
    "max_iters": 100,
    "init_points": 10,
    "retrain_every": 1,
    "seed_offset": 42,
    "print_every": 10,
    "optimizer_defaults": {
        "BCA": {"r": 0.9, "beta": 3.095, "max_iters": 200},
        "PSO": {"n_particles": 32, "w": 0.7, "c1": 1.5, "c2": 1.5, "max_iters": 200},
        "Firefly": {"n_fireflies": 32, "alpha": 0.2, "beta0": 1.0, "gamma": 1.0, "max_iters": 200},
    },
}

OPTIMAL_VALUES = {10: 3.8462, 20: 7.692, 30: 7.627, 40: 7.407, 50: 8.170, 60: 8.257}

torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def labs_merit(x):
    x = x.squeeze()
    n = x.shape[0]
    x_pm = 1 - 2 * x
    C = torch.tensor(
        [torch.dot(x_pm[:n - k], x_pm[k:]) for k in range(1, n)],
        dtype=torch.float64,
        device=device
    )
    denom = torch.sum(C * C)
    val = n ** 2 / (2 * denom) if denom > 0 else 0.0
    return torch.tensor([[val]], dtype=torch.float64, device=device)
    
# wrapper
def labs_merit_batch(X):
    return torch.cat([labs_merit(x.unsqueeze(0)) for x in X], dim=0)

def safe_acq_eval(acq_func, X):
    X = X.to(device=device, dtype=torch.float64)
    Xq = X.unsqueeze(1)  
    with torch.no_grad():
        y = acq_func(Xq)
    return y.view(-1)

class OptBase:
    def __init__(self, dim, acq_func, max_iters):
        self.dim = dim
        self.acq_func = acq_func
        self.max_iters = max_iters

class BCA(OptBase):
    def __init__(self, dim, acq_func, max_iters, r, beta):
        super().__init__(dim, acq_func, max_iters)
        self.r = r
        self.beta = beta

    def fast_chm(self, x):
        x_mutated = x.clone()
        best = safe_acq_eval(self.acq_func, x.unsqueeze(0)).item()
        y = x.clone()
        n = x.shape[0]
        a = torch.randint(0, n, (1,), device=device).item()

        for i in range(n):
            if torch.rand(1, device=device).item() < self.r:
                idx = (a + i) % n
                x_mutated[idx] = 1.0 - x_mutated[idx]
            pi = max(min(i + 1, n - i) ** (-self.beta), 1 / n)
            if torch.rand(1, device=device).item() < pi:
                curr = safe_acq_eval(self.acq_func, x_mutated.unsqueeze(0)).item()
                if curr >= best:
                    best = curr
                    y = x_mutated.clone()
        return y, best


    def optimize(self):
        t_opt_start = time.time()
        x = (torch.rand(self.dim, device=device) < 0.5).float()
        fofx = safe_acq_eval(self.acq_func, x.unsqueeze(0)).item()

        for _ in range(self.max_iters):
            y, fofy = self.fast_chm(x)
            if fofy >= fofx:
                x, fofx = y, fofy
        
        t_opt = time.time() - t_opt_start            
        print(f"    [BCA] Acquisition optimizer run time: {t_opt:.2f}s")
        return x.unsqueeze(0), fofx


class PSO(OptBase):
    def __init__(self, dim, acq_func, n_particles, w, c1, c2, max_iters):
        super().__init__(dim, acq_func, max_iters)
        self.n = n_particles
        self.w, self.c1, self.c2 = w, c1, c2

    def optimize(self):
        t_opt_start = time.time()
        X = (torch.rand(self.n, self.dim, device=device) < 0.5).float()
        V = torch.zeros_like(X)
        vals = safe_acq_eval(self.acq_func, X)
        p_best, p_best_vals = X.clone(), vals.clone()
        g_best = X[vals.argmax()].clone()

        for _ in range(self.max_iters):
            r1, r2 = torch.rand_like(X), torch.rand_like(X)
            V = self.w * V + self.c1 * r1 * (p_best - X) + self.c2 * r2 * (g_best - X)
            X = (torch.rand_like(V) < torch.sigmoid(V)).float()

            vals = safe_acq_eval(self.acq_func, X)
            better = vals > p_best_vals
            p_best[better], p_best_vals[better] = X[better], vals[better]

            g_idx = p_best_vals.argmax()
            g_best = p_best[g_idx].clone()
        
        t_opt = time.time() - t_opt_start            
        print(f"    [PSO] Acquisition optimizer run time: {t_opt:.2f}s")   
        return g_best.unsqueeze(0), p_best_vals.max().item()

class Firefly(OptBase):
    def __init__(self, dim, acq_func, n_fireflies, alpha, beta0, gamma, max_iters):
        super().__init__(dim, acq_func, max_iters)
        self.n = n_fireflies
        self.alpha, self.beta0, self.gamma = alpha, beta0, gamma

    def optimize(self):
        t_opt_start = time.time()
        X = (torch.rand(self.n, self.dim, device=device) < 0.5).float()
        vals = safe_acq_eval(self.acq_func, X)
        for _ in range(self.max_iters):
            dist = torch.cdist(X, X, p=1)
            beta = self.beta0 * torch.exp(-self.gamma * dist**2)
            better = vals.unsqueeze(1) < vals.unsqueeze(0)
            move = (beta.unsqueeze(2) * (X.unsqueeze(0) - X.unsqueeze(1)) * better.unsqueeze(2)).sum(1)
            noise = self.alpha * (torch.rand_like(X) - 0.5)
            logits = move + noise
            X = (torch.rand_like(X) < torch.sigmoid(logits)).float()
            vals = safe_acq_eval(self.acq_func, X)
        idx = vals.argmax()
        t_opt = time.time() - t_opt_start            
        print(f"    [Firefly] Acquisition optimizer run time: {t_opt:.2f}s")  
        return X[idx].unsqueeze(0), vals[idx].item()

class BORunner:
    def __init__(self, dim, use_categorical, opt_class):
        self.dim = dim
        self.use_cat = use_categorical
        self.opt_class = opt_class

    def run(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        t_trial_start = time.time()

        X = (torch.rand(CONFIG["init_points"], self.dim, device=device) < 0.5).double() 
        Y = labs_merit_batch(X)

        covar = None
        if self.use_cat:
            covar = ScaleKernel(CategoricalKernel(num_categories=[2]*self.dim)).to(device)

        model = SingleTaskGP(X, Y, covar_module=covar, outcome_transform=Standardize(m=1)).to(device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            fit_gpytorch_mll(mll)

        for it in range(CONFIG["max_iters"]):
            t_iter_start = time.time()
            if it > 0 and it % CONFIG["retrain_every"] == 0:
                model = SingleTaskGP(X, Y, covar_module=covar, outcome_transform=Standardize(m=1)).to(device)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    fit_gpytorch_mll(mll)

            acq_fn = qLogExpectedImprovement(model, best_f=Y.max())
            opt = self.opt_class(self.dim, acq_fn, **CONFIG["optimizer_defaults"][self.opt_class.__name__])
            new_x, _ = opt.optimize()
            new_y = labs_merit_batch(new_x)
            X = torch.cat([X, new_x], dim=0)
            Y = torch.cat([Y, new_y], dim=0)
            t_iter = time.time() - t_iter_start
            if (it + 1) % CONFIG["print_every"] == 0 or it == CONFIG["max_iters"] - 1:
                print(f"  [Iter {it+1:02d}] Iter Time={t_iter:.2f}s")

        
        t_trial = time.time() - t_trial_start        
        print(f"[Trial Done] Dim={self.dim}, Opt={self.opt_class.__name__}, Kernel={'Cat' if self.use_cat else 'Default'} | Total Trial Time={t_trial:.2f}s\n")  
        return Y.max().item()

def run_all():
    results = []
    optimizers = {"BCA": BCA, "PSO": PSO, "Firefly": Firefly}
    start_total = time.time()
    filename = "bo_results_checkpoint.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Optimizer", "Dimension", "Kernel", "MeanBest", "Std", "Runtime(s)"])

    for name, opt_cls in optimizers.items():
        for d in CONFIG["dims"]:
            for use_cat in [True]:
                vals = []
                t_start = time.time()
                kernel_name = "Categorical" if use_cat else "Default"
                print(f"\n--- {name} D={d} ({kernel_name}) ---")
                for t in range(CONFIG["n_trials"]):
                    seed = CONFIG["seed_offset"] + 1000 * t + d
                    vals.append(BORunner(d, use_cat, opt_cls).run(seed))
                elapsed = time.time() - t_start
                mean, std = float(np.mean(vals)), float(np.std(vals))
                results.append([name, d, kernel_name, mean, std, elapsed])
                print(f"{name:<10} | D={d:<3} | {kernel_name:<12} | mean={mean:.4f} | std={std:.4f} | time={elapsed/60:.2f}m")
                with open(filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, d, kernel_name, mean, std, elapsed])

    print("\n" + "="*90)
    print(f"{'Optimizer':<10} | {'Dim':<4} | {'Kernel':<12} | {'Mean Best':<12} | {'Std':<8} | {'Runtime(s)':<10}")
    print("="*90)
    for r in results:
        print(f"{r[0]:<10} | {r[1]:<4} | {r[2]:<12} | {r[3]:<12.4f} | {r[4]:<8.4f} | {r[5]:<10.2f}")
    print("="*90)
    print(f"Results checkpointed to '{filename}'")
    print(f"Total wall time: {(time.time()-start_total)/3600:.2f} hours on {device.upper()}")

if __name__ == "__main__":
    run_all()