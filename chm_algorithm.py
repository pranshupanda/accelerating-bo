# CHM (Contiguous Hypermutation) Implementation
# Based on: Fast Contiguous Somatic Hypermutations for Single-Objective Optimisation
# and Multi-Objective Optimisation Via Decomposition
# Authors: Dogan Corus, Pietro S. Oliveto, Donya Yazdani

import numpy as np
import matplotlib.pyplot as plt
import random
import os

os.makedirs("plots", exist_ok=True)


def chm(x, r):
    """
    Contiguous Hypermutation (CHM) operator.

    Parameters:
        x (np.ndarray): binary solution vector
        r (float): mutation probability per bit in selected segment

    Returns:
        np.ndarray: mutated solution
    """
    n = len(x)

    # select starting index a in {0, ..., n-1}
    a = np.random.randint(0, n)

    # select segment length l in {1, ..., n}
    l = np.random.randint(1, n + 1)

    x_mutated = x.copy()

    for i in range(l):
        if np.random.rand() < r:
            idx = (a + i) % n
            x_mutated[idx] = 1 - x_mutated[idx]

    return x_mutated


def test_chm():
    x = np.random.randint(0, 2, size=10)
    print("Original x:", x)

    x_mutated = chm(x, r=0.5)
    print("Mutated x: ", x_mutated)


def plot_chm(x, r, steps):
    """
    Visualize CHM evolution over multiple steps.

    Parameters:
        x (np.ndarray): initial solution
        r (float): mutation probability
        steps (int): number of iterations
    """
    history = [x.copy()]

    for _ in range(steps):
        x = chm(x, r)
        history.append(x.copy())

    history = np.array(history)

    plt.imshow(history, cmap='gray_r', aspect='auto')
    plt.xlabel('Index')
    plt.ylabel('Step')
    plt.title('CHM Evolution')
    plt.savefig("plots/chm_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test mutation
    test_chm()

    # Visualization example
    x = np.random.randint(0, 2, 10)
    plot_chm(x, r=0.5, steps=10)
