import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1) Paths to coordinate_winners csv files
csv_noldp = ""  # No-LDP case
csv_ldp   = ""    #    LDP case

def load_and_mask_top2(csv_path):
    """
    Loads a per-epoch win‐counts CSV, computes percentages, masks out
    all but the top-2 clients each epoch, and returns the masked DataFrame
    plus the list of clients that ever appear in top-2.
    """
    df = pd.read_csv(csv_path)
    pct = df.div(df.sum(axis=1), axis=0) * 100
    in_top2 = pct.rank(axis=1, method="first", ascending=False) <= 2
    masked = pct.where(in_top2, other=np.nan)
    clients = [c for c in masked.columns if masked[c].notna().any()]
    return masked, clients

masked_noldp, clients_noldp = load_and_mask_top2(csv_noldp)
masked_ldp,   clients_ldp   = load_and_mask_top2(csv_ldp)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# -- No-LDP subplot (left) --
ax = axes[0]

for client in clients_noldp:
    ax.plot(
        masked_noldp.index,
        masked_noldp[client],
        marker="o",
        label=client
    )
ax.set_title("Without Local Differential Privacy (LDP)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Percentage of Contributed Coordinates")
ax.legend(title="Client", loc="best")
ax.grid(True, linestyle="--", alpha=0.5)

# -- LDP subplot (right) --
ax = axes[1]
for client in clients_ldp:
    ax.plot(
        masked_ldp.index,
        masked_ldp[client],
        marker="o",
        label=client
    )
ax.set_title("With Local Differential Privacy (LDP)")
ax.set_xlabel("Epoch")
ax.legend(title="Client", loc="best")
ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Top-2 Contributed-Coordinate Percentages per Epoch - DATASET - IID_TYPE - ATTACK_TYPE")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()