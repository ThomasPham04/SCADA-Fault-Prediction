"""Simple system architecture diagram – Vietnamese."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#FFFFFF")

NAVY  = "#1A3A5C"
BLUE  = "#2176AE"
TEAL  = "#17A589"
PURP  = "#7D3C98"
ORAN  = "#D35400"
RED   = "#C0392B"
GREEN = "#1E8449"
WHITE = "#FFFFFF"


def box(ax, x, y, w, h, title, sub="", bg=BLUE, tfs=10, sfs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.18",
                 lw=2, edgecolor=bg, facecolor=WHITE, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y+h-0.40), w, 0.40,
                 boxstyle="round,pad=0,rounding_size=0.12",
                 lw=0, facecolor=bg, zorder=3))
    ax.text(x+w/2, y+h-0.20, title,
            va="center", ha="center", fontsize=tfs,
            fontweight="bold", color=WHITE, zorder=4)
    if sub:
        ax.text(x+w/2, y+(h-0.40)/2, sub,
                va="center", ha="center", fontsize=sfs,
                color="#222222", zorder=4, multialignment="center")


def arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=NAVY,
                                lw=2, mutation_scale=18), zorder=5)


# ── title ────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.2, 8.30), 12.6, 0.58,
             boxstyle="round,pad=0,rounding_size=0.18",
             lw=0, facecolor=NAVY, zorder=2))
ax.text(6.5, 8.59,
        "KIEN TRUC HE THONG  -  DU BAO LOI THIET BI DUA TREN DU LIEU SCADA",
        va="center", ha="center", fontsize=11, fontweight="bold",
        color=WHITE, zorder=3)

# ── row 1: 3 boxes ───────────────────────────────────────────────────────────
box(ax, 0.30, 6.35, 3.40, 1.70,
    "Du lieu SCADA tho",
    "CSV tung su kien\n+ event_info.csv\nWind Farm A / B / C",
    bg=BLUE)

box(ax, 4.80, 6.35, 3.40, 1.70,
    "Tien xu ly",
    "Ghep CSV tong hop\nKy thuat dac trung\nChuan hoa (MinMax)",
    bg=TEAL)

box(ax, 9.30, 6.35, 3.40, 1.70,
    "Xuat chuoi",
    "Cua so truot W = 24h\nTrain / Val / Test\nstride = 1 gio",
    bg=PURP)

arr(ax, 3.70, 7.225, 4.80, 7.225)
arr(ax, 8.20, 7.225, 9.30, 7.225)

# ── arrow down to row 2 ───────────────────────────────────────────────────────
arr(ax, 6.50, 6.35, 6.50, 5.45)

# ── row 2: 2 model boxes ─────────────────────────────────────────────────────
box(ax, 0.30, 3.55, 5.70, 1.70,
    "Bo Phan Loai (Global Classifier)",
    "LSTM  /  GRU  /  CNN-LSTM  /  CNN-GRU\nBinary Cross-Entropy + Class Weight\nNguong: quet F1 tren tap Val",
    bg=ORAN)

box(ax, 7.00, 3.55, 5.70, 1.70,
    "Autoencoder (Per-Asset)",
    "LSTM-AE  /  GRU-AE  /  Dense-AE\nHuan luyen tren cua so binh thuong\nNguong: sweep F1 / phan vi 99",
    bg=RED)

# branch arrows
ax.plot([6.50, 6.50], [6.35, 5.65], color=NAVY, lw=2, zorder=4)
ax.plot([3.15, 9.85], [5.65, 5.65], color=NAVY, lw=2, zorder=4)
arr(ax, 3.15, 5.65, 3.15, 5.25)
arr(ax, 9.85, 5.65, 9.85, 5.25)

# ── arrow down to row 3 ───────────────────────────────────────────────────────
arr(ax, 3.15, 3.55, 3.15, 2.65)
arr(ax, 9.85, 3.55, 9.85, 2.65)

# ── row 3: evaluation ────────────────────────────────────────────────────────
box(ax, 2.80, 1.10, 7.40, 1.35,
    "Danh gia ket qua",
    "Accuracy  /  Precision  /  Recall  /  F1  /  ROC-AUC  /  PR-AUC  /  Confusion Matrix",
    bg=GREEN, tfs=10.5, sfs=8.5)

ax.plot([3.15, 3.15], [3.55, 2.45], color=NAVY, lw=2, zorder=4)
ax.plot([9.85, 9.85], [3.55, 2.45], color=NAVY, lw=2, zorder=4)
ax.plot([3.15, 9.85], [2.45, 2.45], color=NAVY, lw=2, zorder=4)
arr(ax, 6.50, 2.45, 6.50, 2.45)

# central arrow from row2 join to row3
ax.annotate("", xy=(6.50, 2.45), xytext=(6.50, 3.00),
            arrowprops=dict(arrowstyle="-|>", color=NAVY,
                            lw=2, mutation_scale=18), zorder=5)

# ── footnote ─────────────────────────────────────────────────────────────────
ax.text(6.5, 0.55,
        "Du lieu: Wind Farm A  |  5 tuabin  |  86 dac trung cam bien  |  "
        "Lay mau: 10 phut  |  Nhan: last-timestep",
        va="center", ha="center", fontsize=8, color="#666666", style="italic")

out = "D:/Final Project/scada-fault-prediction/src/system_architecture_vi.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: " + out)
