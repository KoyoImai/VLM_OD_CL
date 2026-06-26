"""survey_doc の図を生成（英語ラベル＝日本語フォント依存を回避、ベクタPDF出力）。
fig1: 既存手法の位置づけマップ（パラメータ拡張 × ゼロショット保持機構）
fig2: 事前学習VLM検出器の適応ギャップ（Grounding DINO のゼロショット mAP）
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8, "pdf.fonttype": 42})


# ---------- fig1: positioning map ----------
# x: parameter/memory growth with #domains (0=constant ... 1=grows linearly)
# y: explicit zero-shot(pretrained) retention mechanism (0=none ... 1=strong)
# category -> color
cat_color = {
    "baseline":   "#888888",
    "projection": "#1f77b4",
    "lora/peft":  "#2ca02c",
    "merging":    "#9467bd",
    "vlm/ovd-cl": "#d62728",
    "ours":       "#ff7f0e",
}
points = [
    # name, x, y, category, (dx,dy) label offset
    ("Naive seq. FT",      0.04, 0.04, "baseline",   (6, -2)),
    ("EWC / LwF",          0.05, 0.30, "baseline",   (6, 0)),
    ("GPM / OGD",          0.13, 0.42, "projection", (6, -10)),
    ("Low-rank Orth.",     0.16, 0.50, "projection", (6, 4)),
    ("CODA-Prompt",        0.42, 0.30, "lora/peft",  (6, 0)),
    ("O-LoRA",             0.62, 0.52, "lora/peft",  (6, 0)),
    ("SD-LoRA",            0.12, 0.55, "lora/peft",  (-4, 8)),
    ("InfLoRA",            0.07, 0.46, "lora/peft",  (8, -12)),
    ("WiSE-FT / Soups",    0.03, 0.64, "merging",    (8, -2)),
    ("ZSCL (distill)",     0.05, 0.76, "merging",    (8, 2)),
    ("Task Arithmetic",    0.05, 0.58, "merging",    (-2, -14)),
    ("MoE-Adapters",       0.72, 0.70, "merging",    (6, 0)),
    ("ZiRa",               0.07, 0.72, "vlm/ovd-cl", (10, -2)),
    ("DitHub",             0.90, 0.62, "vlm/ovd-cl", (-10, 8)),
    ("Textual-Inv. OVD",   0.10, 0.60, "vlm/ovd-cl", (8, 5)),
]
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.axhspan(0.84, 1.06, xmin=0, xmax=0.34/1.0, color="#ffe9d6", alpha=0.6, zorder=0)
for name, x, y, cat, (dx, dy) in points:
    ax.scatter(x, y, s=70, color=cat_color[cat], edgecolor="white", linewidth=0.6, zorder=3)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=8.5, zorder=4)
# Ours (star)
ax.scatter(0.07, 0.95, marker="*", s=440, color=cat_color["ours"],
           edgecolor="black", linewidth=0.8, zorder=5)
ax.annotate("Ours (InfLoRA-type + COCO-task0\n+ ZCOCO retention, VLM detection)",
            (0.07, 0.95), textcoords="offset points", xytext=(16, -16),
            fontsize=8.8, fontweight="bold", zorder=6)
ax.annotate("target region", (0.335, 1.045), fontsize=8.5, color="#b5651d",
            va="top", ha="right", fontweight="bold")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.10)
ax.set_xlabel("Parameter / memory growth with number of domains\n(left: constant  -  right: grows)")
ax.set_ylabel("Explicit zero-shot (pretrained) retention\n(bottom: none  -  top: strong)")
ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["constant", "moderate", "linear"])
ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["none", "implicit", "strong"])
ax.grid(True, linestyle=":", alpha=0.4)
legend_el = [Line2D([0], [0], marker="o", color="w", label=k,
                    markerfacecolor=v, markersize=8) for k, v in cat_color.items() if k != "ours"]
legend_el.append(Line2D([0], [0], marker="*", color="w", label="ours",
                        markerfacecolor=cat_color["ours"], markeredgecolor="black", markersize=14))
ax.legend(handles=legend_el, loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_positioning.pdf"))
plt.close(fig)


# ---------- fig2: adaptation gap ----------
labels = ["ODinW-13\n(zero-shot)", "RF100-VL\n(zero-shot avg)", "RF100-VL\nMedical"]
vals = [49.2, 16.0, 2.0]
colors = ["#4c9be8", "#e8884c", "#d64c4c"]
fig, ax = plt.subplots(figsize=(5.2, 4.0))
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Zero-shot detection mAP")
ax.set_title("Grounding DINO: domain adaptation gap")
ax.set_ylim(0, 56)
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
ax.annotate("", xy=(1, 18), xytext=(0, 49.2),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
ax.text(0.5, 36, "-33 mAP", color="gray", fontsize=9, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_adaptation_gap.pdf"))
plt.close(fig)

print("wrote:", os.listdir(OUT))
