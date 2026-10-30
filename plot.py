"""Plots benchmark results from results.json into assets/."""

import json

import matplotlib.pyplot as plt
import numpy as np

data = json.load(open("results.json"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for device, res in data["results"].items():
    n = np.array([int(k) for k in res])
    ms = np.array([v["ms"] for v in res.values()])
    gf = np.array([v["gflops"] for v in res.values()])
    ax1.loglog(n, ms, "o-", label=device)
    ax2.plot(n, gf, "o-", label=device)
    # reference line ~n³ anchored at the largest size
    ref = ms[-1] * (n / n[-1]) ** 3
    ax1.loglog(n, ref, "--", color="gray", alpha=0.6, label="~n³ (referencia)")

ax1.set(title="Čas násobenia matíc n×n", xlabel="n", ylabel="čas [ms] (log)")
ax2.set(title="Priepustnosť", xlabel="n", ylabel="GFLOPS", ylim=(0, None))
for ax in (ax1, ax2):
    ax.grid(alpha=0.3, which="both")
    ax.legend()
fig.suptitle(f"PyTorch {data['torch']} · MPS dostupné: {data['mps']}", fontsize=10)
fig.tight_layout()
fig.savefig("assets/matmul_benchmark.png", dpi=150)
