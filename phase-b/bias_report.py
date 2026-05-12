"""B.4 — Quantify position bias + length bias from pairwise_results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PHASE_B_DIR


def main():
    df = pd.read_csv(PHASE_B_DIR / "pairwise_results.csv")
    n = len(df)

    # === Bias 1: Position bias ===
    a_first_wins = (df["run1_winner"] == "A").sum()
    pos_bias_pct = a_first_wins / n * 100

    # === Bias 2: Length bias ===
    df["len_a"] = df["answer_a"].astype(str).str.len()
    df["len_b"] = df["answer_b"].astype(str).str.len()
    df["len_diff"] = df["len_b"] - df["len_a"]
    b_longer = df[df["len_diff"] > 0]
    if len(b_longer) > 0:
        b_wins_when_longer = (b_longer["winner_after_swap"] == "B").sum()
        length_bias_pct = b_wins_when_longer / len(b_longer) * 100
    else:
        length_bias_pct = 0

    # === Chart ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["A first wins", "B first wins", "tie"],
                [(df["run1_winner"] == "A").sum(),
                 (df["run1_winner"] == "B").sum(),
                 (df["run1_winner"] == "tie").sum()])
    axes[0].set_title("Position Bias (run 1, A always first)")
    axes[0].set_ylabel("count")

    axes[1].scatter(df["len_diff"], df["winner_after_swap"].map({"A": 0, "tie": 1, "B": 2}), alpha=0.5)
    axes[1].set_xlabel("len(B) - len(A)")
    axes[1].set_ylabel("Winner (0=A, 1=tie, 2=B)")
    axes[1].set_title("Length vs Winner")
    plt.tight_layout()
    plt.savefig(PHASE_B_DIR / "bias_chart.png", dpi=80)
    plt.close()
    print(f"Chart saved")

    pos_verdict = "WARNING: position bias detected" if pos_bias_pct > 55 else "OK: no significant position bias"
    len_verdict = "WARNING: length bias detected" if length_bias_pct > 60 else "OK: no significant length bias"

    md = f"""# Judge Bias Observations Report

## Bias 1: Position bias

A wins as first position: **{a_first_wins}/{n} = {pos_bias_pct:.1f}%**

Expected ~50% if no bias. >55% suggests position bias.

**Verdict:** {pos_verdict}

## Bias 2: Length bias

When B is longer (n={len(b_longer)}), B wins: **{length_bias_pct:.1f}%**

Expected ~50% if no length preference. >60% suggests length bias.

**Verdict:** {len_verdict}

## Chart

![Bias chart](bias_chart.png)

## Mitigation strategy

- **Position bias:** swap-and-average is already applied in B.1 — agreements kept, disagreements default to tie. This mitigates the raw position bias measured above.
- **Length bias:** if detected, add prompt instruction: "Length is NOT a factor; rate purely on accuracy + relevance." Re-run B.1 with updated prompt.
- **Long-term:** rotate judges (cross-judge protocol, bonus +3) to detect persistent biases.
"""
    out = PHASE_B_DIR / "judge_bias_report.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"\nPosition bias: {pos_bias_pct:.1f}% ({pos_verdict})")
    print(f"Length bias:   {length_bias_pct:.1f}% ({len_verdict})")


if __name__ == "__main__":
    main()
