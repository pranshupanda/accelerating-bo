import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt

CONFIG = {
    "dims": [10, 20, 30, 40, 50, 60],
    "n_trials": 15,
    "max_iters": 100,
    "seed_offset": 43,

    "BCA": {
        "r": 0.95,
        "beta": 3.095
    },
    "Firefly": {
        "n_fireflies": 20,
        "alpha": 0.2,
        "beta0": 1.0,
        "gamma": 1.0
    },
    "PSO": {
        "n_particles": 20,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5
    }
}

OPTIMAL_VALUES = {10: 3.8462, 20: 7.692, 30: 7.627, 40: 7.407, 50: 8.170, 60: 8.257,}
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

class OptimizationAlgorithm:
    def __init__(self, dim, fitness, max_iters=100):
        self.dim = dim
        self.fitness = fitness
        self.max_iters = max_iters
        self.best_val = -float("inf")
        self.best_x = None

    def optimize(self):
        raise NotImplementedError

class BCA(OptimizationAlgorithm):
    def __init__(self, dim, fitness, max_iters, r, beta):
        super().__init__(dim, fitness, max_iters)
        self.r = r
        self.beta = beta

    def fast_chm(self, x):
        x_mutated = x.clone()
        best = self.fitness(x.unsqueeze(0)).item()
        y = x.clone()
        n = x.shape[0]
        a = torch.randint(0, n, (1,), device=device).item()

        for i in range(n):
            if torch.rand(1, device=device).item() < self.r:
                idx = (a + i) % n
                x_mutated[idx] = 1.0 - x_mutated[idx]
            pi = max(min(i + 1, n - i) ** (-self.beta), 1 / n)
            if torch.rand(1, device=device).item() < pi:
                curr = self.fitness(x_mutated.unsqueeze(0)).item()
                if curr >= best:
                    best = curr
                    y = x_mutated.clone()
        return y, best

    def optimize(self):
        x = torch.randint(0, 2, (self.dim,), dtype=torch.float64, device=device)
        fofx = self.fitness(x.unsqueeze(0)).item()
        for _ in range(self.max_iters):
            y, fofy = self.fast_chm(x)
            if fofy >= fofx:
                x, fofx = y, fofy
        self.best_x, self.best_val = x, fofx
        return self.best_val

class FireflyAlgorithm(OptimizationAlgorithm):
    def __init__(self, dim, fitness, max_iters, n_fireflies, alpha, beta0, gamma):
        super().__init__(dim, fitness, max_iters)
        self.n_fireflies = n_fireflies
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def optimize(self):
        pop = torch.randint(0, 2, (self.n_fireflies, self.dim), dtype=torch.float64, device=device)
        fitness_values = torch.tensor([self.fitness(x.unsqueeze(0)).item() for x in pop], device=device)

        for _ in range(self.max_iters):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness_values[j] > fitness_values[i]:
                        rij = torch.sum((pop[i] - pop[j]) ** 2).sqrt()
                        beta = self.beta0 * torch.exp(-self.gamma * rij ** 2)
                        move = torch.rand(self.dim, device=device) < torch.sigmoid(beta * (pop[j] - pop[i]))
                        pop[i] = torch.where(move, 1 - pop[i], pop[i])
                        pop[i] = (pop[i] + self.alpha * torch.randn(self.dim, device=device)).clamp(0, 1).round()
            fitness_values = torch.tensor([self.fitness(x.unsqueeze(0)).item() for x in pop], device=device)

        best_idx = torch.argmax(fitness_values)
        self.best_x = pop[best_idx]
        self.best_val = fitness_values[best_idx].item()
        return self.best_val

class BinaryPSO(OptimizationAlgorithm):
    def __init__(self, dim, fitness, max_iters, n_particles, w, c1, c2):
        super().__init__(dim, fitness, max_iters)
        self.n_particles = n_particles
        self.w, self.c1, self.c2 = w, c1, c2

    def optimize(self):
        X = torch.randint(0, 2, (self.n_particles, self.dim), dtype=torch.float64, device=device)
        V = torch.zeros_like(X)
        personal_best = X.clone()
        personal_best_vals = torch.tensor([self.fitness(x.unsqueeze(0)).item() for x in X], device=device)

        g_best_idx = torch.argmax(personal_best_vals)
        g_best = personal_best[g_best_idx].clone()
        g_best_val = personal_best_vals[g_best_idx].item()

        for _ in range(self.max_iters):
            for i in range(self.n_particles):
                V[i] = self.w * V[i] + self.c1 * torch.rand(1, device=device) * (personal_best[i] - X[i]) + \
                        self.c2 * torch.rand(1, device=device) * (g_best - X[i])
                prob = torch.sigmoid(V[i])
                X[i] = torch.where(torch.rand(self.dim, device=device) < prob, 1.0, 0.0)
                f = self.fitness(X[i].unsqueeze(0)).item()

                if f > personal_best_vals[i]:
                    personal_best[i] = X[i].clone()
                    personal_best_vals[i] = f
                    if f > g_best_val:
                        g_best, g_best_val = X[i].clone(), f

        self.best_x, self.best_val = g_best, g_best_val
        return self.best_val

def run_comparison():
    cfg = CONFIG
    all_results = []
    for d in cfg["dims"]:
        for t in range(cfg["n_trials"]):
            torch.manual_seed(cfg["seed_offset"] + t * 1000 + d)
            algos = {
                "BCA": BCA(d, labs_merit, cfg["max_iters"], **cfg["BCA"]),
                "Firefly": FireflyAlgorithm(d, labs_merit, cfg["max_iters"], **cfg["Firefly"]),
                "PSO": BinaryPSO(d, labs_merit, cfg["max_iters"], **cfg["PSO"])
            }
            result_entry = {d: {}}
            print(f"\nRunning trial {t + 1}/{cfg['n_trials']} for dim={d}...")
            for name, algo in algos.items():
                best_val = algo.optimize()
                result_entry[d][name] = {
                    "best_val": best_val,
                    "final_gap": OPTIMAL_VALUES[d] - best_val
                }
                print(f"  {name}: best_val={best_val:.4f}, gap={OPTIMAL_VALUES[d] - best_val:.4f}")
            all_results.append(result_entry)
    summarize_results(all_results, cfg["dims"])

def summarize_results(results, dims):
    print("\n" + "=" * 80)
    print("AVERAGED RESULTS ACROSS TRIALS")
    print("=" * 80)
    algos = ["BCA", "Firefly", "PSO"]

    all_vals = {d: {a: [] for a in algos} for d in dims}

    for d in dims:
        print(f"\nDimension {d} (Optimal: {OPTIMAL_VALUES[d]:.4f}):")
        for name in algos:
            vals = [r[d][name]["best_val"] for r in results if d in r]
            all_vals[d][name] = vals
            gaps = [r[d][name]["final_gap"] for r in results if d in r]
            print(f"  {name}: Mean={np.mean(vals):.4f} ± {np.std(vals):.4f} | Mean Gap={np.mean(gaps):.4f}")

    for d in dims:
        means = [np.mean(all_vals[d][a]) for a in algos]
        stds = [np.std(all_vals[d][a]) for a in algos]
        plt.figure(figsize=(8,5))
        plt.bar(algos, means, yerr=stds, capsize=5)
        plt.axhline(OPTIMAL_VALUES[d], color='r', linestyle='--', label='Optimal Value')
        plt.title(f"Dimension {d}: Best Value ± Std")
        plt.ylabel("Best Value")
        plt.legend()
        plt.show()

    for d in dims:
        data = [all_vals[d][a] for a in algos]
        plt.figure(figsize=(8,5))
        plt.boxplot(data, labels=algos)
        plt.axhline(OPTIMAL_VALUES[d], color='r', linestyle='--', label='Optimal Value')
        plt.title(f"Dimension {d}: Distribution of Best Values")
        plt.ylabel("Best Value")
        plt.legend()
        plt.show()

    means_by_algo = {a: [] for a in algos}
    stds_by_algo = {a: [] for a in algos}
    for d in dims:
        for a in algos:
            means_by_algo[a].append(np.mean(all_vals[d][a]))
            stds_by_algo[a].append(np.std(all_vals[d][a]))

    plt.figure(figsize=(10,6))
    for a in algos:
        plt.errorbar(dims, means_by_algo[a], yerr=stds_by_algo[a], marker='o', capsize=5, label=a)
    plt.plot(dims, [OPTIMAL_VALUES[d] for d in dims], 'r--', label="Optimal Value")
    plt.xlabel("Dimension")
    plt.ylabel("Best Value")
    plt.title("Algorithm Performance vs Dimension")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    start = time()
    run_comparison()
    print(f"\nTotal time: {time() - start:.1f} seconds on {device.upper()}")
