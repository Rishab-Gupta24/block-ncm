"""Figures for the Block-NCM paper. Every number is copied from RESULTS.md (section named next to it).

    python paper/scripts/make_figures.py   ->  paper/figures/*.pdf (+ .png previews)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "legend.frameon": False, "pdf.fonttype": 42})
# one colour per model, used in every figure
COL = {"B0": "#8c8c8c", "G3": "#1f77b4", "G5": "#2ca02c", "B2": "#d62728", "AR": "#9467bd"}
LAB = {"B0": "B0 (G=0)", "G3": "Ours G=3 + halo 4", "G5": "Ours G=5 + halo 4", "B2": "B2 (G=11)"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1: what attends to what
def fig_architecture():
    K, M, G = 11, 4, 3
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    titles = ["B0: independent chunks (G=0)", "Block-NCM: plan + local + halo", "B2: fully global (G=11)"]
    for ax, title, g, halo in zip(axes, titles, [0, G, K], [False, True, False]):
        for lvl in range(K):
            y = K - 1 - lvl
            if lvl < g:  # global level: one attention context across all chunks
                ax.add_patch(FancyBboxPatch((0.05, y + 0.12), M - 0.1, 0.76, boxstyle="round,pad=0.0,rounding_size=0.12",
                                            fc="#cfe2f3" if g < K else "#f4cccc", ec="#3d6d99" if g < K else "#a33",
                                            lw=0.8))
            else:  # local level: one context per chunk
                for m in range(M):
                    ax.add_patch(FancyBboxPatch((m + 0.08, y + 0.12), 0.84, 0.76,
                                                boxstyle="round,pad=0.0,rounding_size=0.12", fc="#e8e8e8", ec="#777",
                                                lw=0.6))
                    if halo:
                        for x0 in ([m + 0.08] if m > 0 else []) + ([m + 0.92 - 0.12] if m < M - 1 else []):
                            ax.add_patch(Rectangle((x0, y + 0.12), 0.12, 0.76, fc="#f6b26b", ec="none", alpha=0.9))
        ax.set_xlim(-0.05, M + 0.05)
        ax.set_ylim(-0.6, K + 0.2)
        ax.set_xticks([m + 0.5 for m in range(M)], [str(m + 1) for m in range(M)], fontsize=7)
        ax.set_xlabel("chunk", fontsize=7, labelpad=1)
        ax.set_yticks([K - 0.5, 0.5], ["level 1\n(coarse)", "level 11\n(fine)"], fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8)
    axes[1].annotate("plan: G=3 global levels", xy=(2, K - 1.5), xytext=(2, K + 0.05), ha="center", fontsize=7,
                     color="#3d6d99")
    handles = [Rectangle((0, 0), 1, 1, fc="#cfe2f3", ec="#3d6d99"), Rectangle((0, 0), 1, 1, fc="#e8e8e8", ec="#777"),
               Rectangle((0, 0), 1, 1, fc="#f6b26b"), Rectangle((0, 0), 1, 1, fc="#f4cccc", ec="#a33")]
    fig.legend(handles, ["global level (attends across chunks)", "local level (one chunk)",
                         "halo: 4 inputs of each neighbour", "all levels global"],
               loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, 0.955, "each level is one parallel pass; all variants take 11 sequential passes for any number of "
             "chunks", ha="center", fontsize=7.5, style="italic")
    fig.subplots_adjust(top=0.84, wspace=0.25)
    save(fig, "fig_architecture")


# ------------------------------------------------------------------ Fig. 2: speed
def fig_speed():
    # RESULTS.md 2026-11-21 (RTX 5050, median of 10, expandable_segments, no memory spill), unrounded:
    # (a) batch 1 vs story length: ~/ncm/runs/time_b1_v3.json; AR at 517 tokens from ~/ncm/runs/time_gen5.json
    toks = [256, 512, 1024]
    block1 = [0.38623, 0.35985, 0.37802]
    b1 = [0.79732, 1.55583, 3.55903]
    ar1 = 6.20140
    # (b) seconds per 512-token story vs batch: ~/ncm/runs/time_gen5.json
    bs = [1, 8, 16, 32]
    blk = [0.376373, 0.070453, 0.062550, 0.062750]
    ar = [6.201404, 0.760047, 0.373304, 0.202377]
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.6))
    a.plot(toks, b1, "-o", color=COL["B0"], ms=4, lw=1.6, label="B1: chunks in sequence (lower bound)")
    a.plot(toks, block1, "-o", color=COL["G5"], ms=4, lw=1.6, label="Block-NCM G=5 + halo 4")
    a.plot([512], [ar1], "*", color=COL["AR"], ms=9, label="TinyStories-33M (AR)")
    for x, v, w in zip(toks, b1, block1):
        a.text(x, v + 0.3, f"{v / w:.1f}×", ha="center", fontsize=7.5, color=COL["B0"])
        a.text(x, w - 0.55, f"{w:.2f} s", ha="center", fontsize=7, color=COL["G5"])
    a.text(560, ar1, f"{ar1:.2f} s ({ar1 / blk[0]:.1f}× Block-NCM, same run)", va="center", fontsize=7,
           color=COL["AR"])
    a.set_xticks(toks, [f"{t}\n({t // 128} chunks)" for t in toks], fontsize=7)
    a.set_xlim(150, 1250)
    a.set_ylim(-0.8, 7.4)
    a.set_ylabel("seconds per story", fontsize=8)
    a.set_title("(a) Batch 1: latency vs story length", fontsize=8, loc="left")
    a.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 0.8), frameon=False)
    a.tick_params(axis="y", labelsize=7)
    b.plot(bs, ar, "-*", color=COL["AR"], ms=7, lw=1.6, label="TinyStories-33M (AR)")
    b.plot(bs, blk, "-o", color=COL["G5"], ms=4, lw=1.6, label="Block-NCM G=5 + halo 4")
    for x, u, v in zip(bs, blk, ar):
        b.text(x, (u * v) ** 0.5, f"{v / u:.1f}×", ha="center", va="center", fontsize=7.5,
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    b.set_xscale("log", base=2)
    b.set_xlim(0.75, 42)
    b.set_yscale("log")
    b.set_xticks(bs, [str(x) for x in bs], fontsize=7)
    b.set_xlabel("batch size", fontsize=8)
    b.set_ylabel("seconds per story", fontsize=8)
    b.set_title("(b) 512-token stories: time per story vs batch", fontsize=8, loc="left")
    b.legend(fontsize=6.5, loc="upper right", frameon=False)
    b.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save(fig, "fig_speed")


# ------------------------------------------------------------------ Fig. 3: the trade-off
def fig_tradeoff():
    # RESULTS.md "Third seed of the main models": genPPL per seed (0 / 1 / 2)
    m4 = {"B0": [24.1, 24.0, 24.5], "G3": [21.0, 20.9, 21.5], "G5": [19.1, 19.0, 19.6], "B2": [19.0, 18.4, 18.7]}
    m6 = {"B0": [23.7, 23.7, 23.9], "G3": [21.5, 20.6, 21.6], "G5": [20.0, 21.0, 20.8], "B2": [33.5, 31.0, 29.4]}
    # RESULTS.md "Matched per-chunk test": chunk-1 genPPL, M=4 -> M=6, temperature 1.0 (one training seed)
    c1 = {"B0": (20.8, 20.6), "G3": (21.2, 22.5), "G5": (21.2, 28.6), "B2": (20.6, 35.3)}
    G = {"B0": 0, "G3": 3, "G5": 5, "B2": 11}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for k in m4:
        mu4, mu6 = sum(m4[k]) / 3, sum(m6[k]) / 3
        a.plot([0, 1], [mu4, mu6], "-o", color=COL[k], lw=1.6, ms=4, label=LAB[k], zorder=3)
        a.scatter([0] * 3, m4[k], color=COL[k], s=8, alpha=0.45, zorder=2)
        a.scatter([1] * 3, m6[k], color=COL[k], s=8, alpha=0.45, zorder=2)
        a.text(1.06, mu6, f"{mu6:.1f}", va="center", fontsize=7, color=COL[k])
        a.text(-0.06, mu4 + {"G5": 0.35, "B2": -0.35}.get(k, 0), f"{mu4:.1f}", va="center", ha="right", fontsize=7,
               color=COL[k])
    a.set_xticks([0, 1], ["M=4 (trained)\n512 tokens", "M=6 (unseen)\n768 tokens"])
    a.set_xlim(-0.35, 1.35)
    a.set_ylabel("genPPL (GPT-2 Large), lower is better")
    a.set_title("(a) Whole-story genPPL (3 seeds; dots = seeds)")
    a.legend(fontsize=7, loc="upper left")
    xs = range(len(c1))
    shift = [100 * (c1[k][1] / c1[k][0] - 1) for k in c1]
    bars = b.bar(xs, shift, color=[COL[k] for k in c1], width=0.6)
    for x, s in zip(xs, shift):
        b.text(x, max(s, 0) + 1.5, f"{s:+.0f}%", ha="center", fontsize=8)
    b.set_xticks(list(xs), [f"G={G[k]}" + ("\n+halo" if k in ("G3", "G5") else "") for k in c1], fontsize=8)
    b.axhline(0, color="k", lw=0.6)
    b.set_ylim(-5, 85)
    b.set_ylabel("change in chunk-1 genPPL (%)")
    b.set_title("(b) Chunk 1, M=4 → M=6 (1 seed)")
    fig.tight_layout()
    save(fig, "fig_tradeoff")


# ------------------------------------------------------------------ Fig. 4: per-chunk genPPL
def fig_chunks():
    # RESULTS.md "Matched per-chunk test", temperature 1.0: chunk 1..4 genPPL
    d = {"B0": ([20.8, 25.3, 24.2, 38.4], [20.6, 25.1, 23.5, 37.5]),
         "G3": ([21.2, 20.7, 19.9, 26.3], [22.5, 21.6, 20.1, 20.9]),
         "G5": ([21.2, 18.6, 18.5, 19.1], [28.6, 23.7, 22.4, 23.5]),
         "B2": ([20.6, 18.4, 17.7, 21.0], [35.3, 30.7, 28.8, 33.1])}
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0), sharey=True, sharex=True)
    axes = axes.ravel()
    for ax, (k, (a4, a6)) in zip(axes, d.items()):
        ax.plot(range(1, 5), a4, "-o", color=COL[k], ms=3, lw=1.3, label="4-chunk story")
        ax.plot(range(1, 5), a6, "--s", color=COL[k], ms=3, lw=1.3, alpha=0.7, label="6-chunk story")
        ax.set_title(LAB[k] + (" †" if k == "G5" else ""), fontsize=7.5, color=COL[k])
        ax.set_xticks(range(1, 5))
        ax.tick_params(labelsize=7)
    for ax in axes[2:]:
        ax.set_xlabel("chunk", fontsize=7.5)
    for ax in axes[::2]:
        ax.set_ylabel("chunk genPPL", fontsize=7.5)
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_chunks")


if __name__ == "__main__":
    fig_architecture()
    fig_speed()
    fig_tradeoff()
    fig_chunks()
    print("figures written to", os.path.abspath(OUT))
