"""
Plot theoretical Pc vs lambda, fixing NSR, alpha, beta.

Model:
  q(NSR)  = min(alpha * NSR^beta, 1)
  Pa      = min(1, max(0.5, ((1+q)*exp(x) - (1-q)) / (2*(exp(x) - 1))))
  Pb      = 1 - Pa
  P1a     = Pa*exp(x) / (Pa*exp(x) + Pb)           <- P(A=a | Y=1)
  P2b     = Pb*exp(x) / (Pa + Pb*exp(x))           <- P(A=b | Y=2)
  Pc      = 0.5*(1-q)*P1a + 0.5*(1-q)*P2b + 0.5*q

  where  x = r/lambda = 1/lambda  (r=1 normalised, lambda = 1/x)

Fixed (estimated from text-noise bootstrap fit):
  alpha = 0.2946,  beta = 2.3181
  Three NSR curves: small / medium / large  (NSR on 0-100 scale)

Output:
  data/results/text_noise_set2/pc_vs_lambda.png
  data/results/text_noise_set2/pc_vs_lambda_table.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Fixed parameters ──────────────────────────────────────────────────────────
ALPHA = 0.2946
BETA  = 2.3181

# NSR values that give clearly different q with the text-noise alpha/beta.
# q saturates to 1 for NSR > ~1.7; if using audio alpha/beta (fitted on
# NSR 0-100 scale) replace these with values like [5, 30, 80].
NSR_VALUES   = [0.3, 1.0, 1.7]
NSR_LABELS   = ["Small  (NSR=0.3)", "Medium (NSR=1.0)", "Large  (NSR=1.7)"]
LAMBDA_RANGE = np.linspace(0.05, 2.0, 500)   # x-axis

OUT_DIR = "./data/results/text_noise_set2"
os.makedirs(OUT_DIR, exist_ok=True)

_EPS = 1e-10


# ── Core functions ────────────────────────────────────────────────────────────

def q_of_nsr(nsr, alpha, beta):
    if nsr <= 0:
        return 0.0
    return float(np.clip(alpha * (nsr ** beta), 0.0, 1.0))


def compute_Pa(x, q):
    """Pa = min(1, max(0.5, ((1+q)*exp(x) - (1-q)) / (2*(exp(x)-1))))"""
    ex = np.exp(np.clip(x, -500, 500))
    denom = 2 * (ex - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(
            np.abs(denom) < 1e-12,
            0.5,
            ((1 + q) * ex - (1 - q)) / denom
        )
    return np.clip(val, 0.5, 1.0 - _EPS)


def compute_P1a(x, Pa):
    """P(A=a | Y=1) = Pa*exp(x) / (Pa*exp(x) + Pb)"""
    ex = np.exp(np.clip(x, -500, 500))
    Pb = 1 - Pa
    return np.clip((Pa * ex) / (Pa * ex + Pb + _EPS), 0.0, 1.0)


def compute_P2b(x, Pa):
    """P(A=b | Y=2) = Pb*exp(x) / (Pa + Pb*exp(x))"""
    ex = np.exp(np.clip(x, -500, 500))
    Pb = 1 - Pa
    return np.clip((Pb * ex) / (Pa + Pb * ex + _EPS), 0.0, 1.0)


def compute_Pc(x, nsr, alpha, beta):
    """Full Pc given x=r/lambda, NSR, alpha, beta."""
    q   = q_of_nsr(nsr, alpha, beta)
    Pa  = compute_Pa(x, q)
    P1a = compute_P1a(x, Pa)
    P2b = compute_P2b(x, Pa)
    return 0.5 * (1 - q) * P1a + 0.5 * (1 - q) * P2b + 0.5 * q


# ── Build table of intermediate values ───────────────────────────────────────

rows = []
for nsr in NSR_VALUES:
    q = q_of_nsr(nsr, ALPHA, BETA)
    for lam in LAMBDA_RANGE:
        x   = 1.0 / lam
        Pa  = float(compute_Pa(x, q))
        Pb  = 1.0 - Pa
        P1a = float(compute_P1a(x, Pa))
        P2b = float(compute_P2b(x, Pa))
        Pc  = 0.5 * (1 - q) * P1a + 0.5 * (1 - q) * P2b + 0.5 * q
        rows.append({
            "NSR":    nsr,
            "lambda": round(lam, 6),
            "x":      round(x,   6),
            "q":      round(q,   6),
            "Pa":     round(Pa,  6),
            "Pb":     round(Pb,  6),
            "P1a":    round(P1a, 6),
            "P2b":    round(P2b, 6),
            "Pc":     round(Pc,  6),
        })

table_df = pd.DataFrame(rows)
table_path = os.path.join(OUT_DIR, "pc_vs_lambda_table.csv")
table_df.to_csv(table_path, index=False)
print(f"Intermediate values -> {table_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

colors = ["#2166ac", "#4dac26", "#d01c8b"]
for nsr, lbl, color in zip(NSR_VALUES, NSR_LABELS, colors):
    sub = table_df[table_df["NSR"] == nsr]
    q   = q_of_nsr(nsr, ALPHA, BETA)
    ax.plot(sub["lambda"], sub["Pc"], lw=2, color=color,
            label=f"NSR={nsr},  q={q:.3f}")

ax.set_xlabel(r"$\lambda$ (information cost)", fontsize=12)
ax.set_ylabel(r"$P^{(Y)}_{\mathrm{correct}}$", fontsize=12)
ax.set_title(
    rf"Theoretical $P^{{(Y)}}_{{\mathrm{{correct}}}}$ vs $\lambda$"
    f"\n(fixed α={ALPHA}, β={BETA:.4f})",
    fontsize=12
)
ax.set_xlim(0.05, 2.0)
ax.set_ylim(0.45, 1.0)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "pc_vs_lambda.png")
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Plot -> {plot_path}")

# ── Print q values for chosen NSR levels ──────────────────────────────────────
print(f"\nNSR -> q  (alpha={ALPHA}, beta={BETA}):")
for nsr, lbl in zip(NSR_VALUES, NSR_LABELS):
    print(f"  {lbl}  ->  q = {q_of_nsr(nsr, ALPHA, BETA):.4f}")
