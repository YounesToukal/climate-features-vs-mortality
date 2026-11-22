"""Shared figure style."""

import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GREY = "#8a8984"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "figure.constrained_layout.use": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.8,
    "axes.edgecolor": GREY,
    "axes.labelcolor": "#33322f",
    "axes.titlesize": 11,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "legend.frameon": False,
    "font.size": 10,
})
