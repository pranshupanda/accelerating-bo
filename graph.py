import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("bo_results_checkpoint.csv")

# Add optimal values
dims = [10, 20, 30, 40, 50, 60]
optimal = [3.8462, 7.6920, 7.6270, 7.4070, 8.1700, 8.2570]


# Define colors and linestyles
colors = {
    "BCA": "#1f77b4",
    "PSO": "#2ca02c",
    "Firefly": "#d62728",
    "LocalSearch": "#17becf"
}
linestyles = {
    "Default": "-",
    "Categorical": "--"
}

# ---------- Plot 1: Mean Best vs Dimension ----------
plt.figure(figsize=(10, 6))
for opt in df["Optimizer"].unique():
    for kernel in ["Categorical"]:
        sub = df[(df["Optimizer"] == opt) & (df["Kernel"] == kernel)]
        plt.plot(
            sub["Dim"], sub["MeanBest"],
            label=f"{opt} ({kernel})",
            color=colors[opt],
            linestyle=linestyles[kernel],
            marker='o'
        )

# Optimal line
plt.plot(
    dims, optimal, color="black", linestyle=":", marker="s", linewidth=2,
    label="Optimal"
)

# ---- Add percentage annotations (BCA vs Local Search) ----
bca = df[df["Optimizer"] == "BCA"].set_index("Dim")["MeanBest"]
ls = df[df["Optimizer"] == "LocalSearch"].set_index("Dim")["MeanBest"]
percent_worse = ((ls - bca) / ls) * 100

for dim in dims:
    if dim in percent_worse.index:
        x = dim
        y = ls[dim]
        pct = percent_worse[dim]

        plt.text(
            x, y + 0.15,
            f"{pct:.1f}% worse",
            fontsize=8,
            color="#444444"
        )

plt.title("Mean Best vs Dimension")
plt.xlabel("Dimension")
plt.ylabel("Mean Best Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/bo_mean_best_vs_dimension.png", dpi=300, bbox_inches='tight')
plt.close()

# ---------- Plot 2: Runtime vs Dimension ----------
plt.figure(figsize=(10, 6))
for opt in df["Optimizer"].unique():
    for kernel in ["Categorical"]:
        sub = df[(df["Optimizer"] == opt) & (df["Kernel"] == kernel)]
        plt.plot(
            sub["Dim"], sub["Runtime"],
            label=f"{opt} ({kernel})",
            color=colors[opt],
            linestyle=linestyles[kernel],
            marker='o'
        )
plt.title("Runtime vs Dimension ")
plt.xlabel("Dimension")
plt.ylabel("Runtime (s)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/bo_runtime_vs_dimension.png", dpi=300, bbox_inches='tight')
plt.close()
