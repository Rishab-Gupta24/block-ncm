"""Illustrations of the base NCM for the story-style log (paper/blockncm_story.docx).

Numbers: level lengths from HazyResearch/ncm (TinyStories config, see docs/long_form_design.md);
genPPL from the NCM paper (15.41, 8.89, 9.67) and our reproduction (16.38, RESULTS.md).

    python paper/scripts/make_story_figures.py   ->  paper/figures/story_ncm_*.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9.5, "legend.frameon": False})
LEVELS = [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]  # the 11 generated levels; level 0 is one start position
BLUE, PALE, GREY, ORANGE, PURPLE, GREEN = "#2f5d8a", "#d6e4f2", "#8c8c8c", "#f0a030", "#9467bd", "#2ca02c"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            wrap=True)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, color="#555", lw=1.1))


# ------------------------------------------------------------------ the NCM pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.1)
    ax.axis("off")
    # 1. text in
    box(ax, 0.1, 1.55, 1.6, 1.0, "128 tokens\nof text\n(one paragraph)", "#f3f3f3", GREY, fs=7.5)
    arrow(ax, 1.75, 2.2, 2.05)
    # 2. tokenizer encoder
    box(ax, 2.25, 1.55, 1.55, 1.0, "Tokenizer\nencoder\n(VQ-VAE)", PALE, BLUE)
    arrow(ax, 3.85, 4.3, 2.05)
    # 3. the code pyramid
    x0, top, h = 4.35, 3.75, 0.27
    cx = x0 + 1.25
    ax.add_patch(Rectangle((cx - 0.04, top - h + 0.04), 0.08, h - 0.08, fc=BLUE, ec="none"))
    for i, n in enumerate(LEVELS):
        w = 2.5 * n / 128
        y = top - (i + 2) * h
        ax.add_patch(Rectangle((cx - w / 2, y + 0.03), w, h - 0.06, fc=BLUE, ec="none", alpha=0.25 + 0.7 * (i + 1) / 11))
    ax.text(cx, top + 0.12, "codes, coarse → fine", ha="center", fontsize=7.5, weight="bold")
    ax.text(cx + 1.38, top - 2 * h + h / 2, "level 1: 8 codes\n(the gist)", fontsize=6.5, va="center", color=BLUE)
    ax.text(cx + 1.38, top - 12 * h + h / 2, "level 11: 128 codes\n(exact words)", fontsize=6.5, va="center",
            color=BLUE)
    ax.text(cx, 0.28, "1 + 8 + 16 + … + 128 = 641 codes", ha="center", fontsize=7)
    arrow(ax, 7.0, 7.4, 2.05)
    # 4. generator
    box(ax, 7.45, 1.55, 1.45, 1.0, "Generator\n(transformer)\none pass\nper level", "#e5f4e4", GREEN, fs=7.5)
    arrow(ax, 8.95, 9.4, 2.05)
    # 5. decoder
    box(ax, 9.45, 1.55, 1.1, 1.0, "Decoder\n→ text", PALE, BLUE)
    # training vs writing
    ax.text(3.0, 1.0, "training: real text\nis turned into codes", ha="center", fontsize=6.8, style="italic",
            color="#555")
    ax.text(8.2, 1.0, "writing: the generator invents\nthe codes, level by level", ha="center", fontsize=6.8,
            style="italic", color="#555")
    save(fig, "story_ncm_pipeline")


# ------------------------------------------------------------------ AR vs NCM: who needs how many passes
def fig_passes():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) AR: one new token per pass
    for p in range(128):
        a.add_patch(Rectangle((0, p), p, 1, fc=PALE, ec="none"))
        a.add_patch(Rectangle((p, p), 1, 1, fc=PURPLE, ec="none"))
    a.set_xlim(0, 128)
    a.set_ylim(128, 0)
    a.set_xlabel("token position")
    a.set_ylabel("sequential pass")
    a.set_xticks([0, 32, 64, 96, 128])
    a.set_yticks([1, 32, 64, 96, 128])
    a.set_title("Autoregressive: 1 new token per pass\n→ 128 passes", color=PURPLE)
    a.text(62, 44, "purple = the token\nwritten in that pass\n\nblue = everything it\nlooks back at", fontsize=7,
           color="#444")
    # (b) NCM: one whole level per pass
    for i, n in enumerate(LEVELS):
        b.add_patch(Rectangle((64 - n / 2, i + 0.12), n, 0.76, fc=GREEN, ec="none", alpha=0.35 + 0.6 * (i + 1) / 11))
        b.text(64 + n / 2 + 2, i + 0.5, f"{n} codes", va="center", fontsize=6.8, color="#333")
    b.set_xlim(0, 150)
    b.set_ylim(11, 0)
    b.set_yticks([i + 0.5 for i in range(11)], [str(i + 1) for i in range(11)], fontsize=7)
    b.set_ylabel("sequential pass")
    b.set_xticks([])
    b.spines["bottom"].set_visible(False)
    b.set_title("NCM: a whole level per pass, all at once\n→ 11 passes", color=GREEN)
    b.text(2, 1.6, "rough idea first,\nexact words last", fontsize=7, color="#444")
    fig.tight_layout(w_pad=2.5)
    save(fig, "story_ncm_passes")


# ------------------------------------------------------------------ the claims and the price
def fig_tradeoffs():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    names = ["Real\nTinyStories", "AR model\n(NCM paper)", "NCM\n(NCM paper)", "NCM\n(my re-run)"]
    vals = [9.67, 8.89, 15.41, 16.38]
    cols = [GREY, PURPLE, GREEN, GREEN]
    bars = a.bar(range(4), vals, color=cols, width=0.62)
    bars[3].set_hatch("///")
    bars[3].set_edgecolor("white")
    for i, v in enumerate(vals):
        a.text(i, v + 0.35, f"{v:.2f}", ha="center", fontsize=8)
    a.set_xticks(range(4), names, fontsize=7.5)
    a.set_ylabel("genPPL, 128 tokens (lower = better)")
    a.set_ylim(0, 19.5)
    a.set_title("Quality: NCM is about 1.7× behind AR")
    # (b) positions computed per token written
    bb = b.bar([0, 1], [1, 641 / 128], color=[PURPLE, GREEN], width=0.55)
    b.text(0, 1.15, "1", ha="center", fontsize=8)
    b.text(1, 641 / 128 + 0.15, "5.0\n(641 ÷ 128)", ha="center", fontsize=8)
    b.set_xticks([0, 1], ["AR", "NCM"])
    b.set_ylabel("positions computed per token")
    b.set_ylim(0, 6.6)
    b.set_title("Work: NCM does ~5× more,\nin far fewer passes")
    fig.tight_layout(w_pad=2.5)
    save(fig, "story_ncm_tradeoffs")


if __name__ == "__main__":
    fig_pipeline()
    fig_passes()
    fig_tradeoffs()
    print("wrote story_ncm_pipeline/passes/tradeoffs.png to", os.path.abspath(OUT))
