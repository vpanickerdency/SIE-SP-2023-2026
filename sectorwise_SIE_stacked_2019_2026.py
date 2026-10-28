"""
Sector-wise Antarctic sea ice extent, 2019-2026, stacked in one figure.

Panels (top to bottom): (a) WSS, (b) ABS, (c) WPS, (d) RSS, (e) IOS.
All panels share the same x-axis (Jan-Dec, labelled only on the bottom panel)
and the same y-scale, with one common legend.

Reads the sheets wss, abs, wps, rss, ios of regional_SIE.xlsx.
Each sheet: column 'date' (a 2020 calendar, used only for the month axis)
and one column per year, in million km^2.

Requirements:  pip install pandas matplotlib openpyxl
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ------------------------------
# SETTINGS
# ------------------------------
file_path = r"C:\Users\vpani\Downloads\FINAL\regional_SIE.xlsx"
out_png = r"C:\Users\vpani\Downloads\FINAL\Sectorwise_SIE_stacked_2019_2026.png"

HIGHLIGHT_YEAR = 2023
Y_MAX = 8                         # same y-scale for every panel

panels = [
    ("wss", "(a) WSS"),
    ("abs", "(b) ABS"),
    ("wps", "(c) WPS"),
    ("rss", "(d) RSS"),
    ("ios", "(e) IOS"),
]

# fixed colours so each year looks the same in every panel
year_colors = {
    2019: "#1f77b4", 2020: "#ff7f0e", 2021: "#2ca02c", 2022: "#17becf",
    2023: "#d62728", 2024: "#9467bd", 2025: "#8c564b", 2026: "#e377c2",
}
LW, LW_HL = 1.6, 2.8              # line widths: other years / highlight year

FS_LABEL, FS_TICK, FS_AX, FS_LEG, FS_MAX = 19, 16, 19, 15, 15

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------
# PLOT
# ------------------------------
fig, axes = plt.subplots(len(panels), 1, figsize=(11, 15), sharex=True, sharey=True)

for ax, (sheet, label) in zip(axes, panels):
    df = pd.read_excel(file_path, sheet_name=sheet)
    df["date"] = pd.to_datetime(df["date"])
    year_cols = [c for c in df.columns[1:] if str(c).strip().isdigit()]

    for col in year_cols:
        s = df[["date", col]].dropna()          # skips the 29 Feb gap; 2026 ends at its last date
        hl = col == HIGHLIGHT_YEAR
        ax.plot(s["date"], s[col], color=year_colors.get(int(col)),
                linewidth=LW_HL if hl else LW, alpha=1 if hl else 0.8,
                zorder=5 if hl else 3)

    # maxima: highlight year and the whole 2019-2026 period
    max_hl = df[HIGHLIGHT_YEAR].max()
    date_hl = df.loc[df[HIGHLIGHT_YEAR].idxmax(), "date"]
    overall_max = df[year_cols].max().max()
    year_of_max = int(df[year_cols].max().idxmax())
    date_overall = df.loc[df[year_of_max].idxmax(), "date"]

    ax.scatter(date_overall, overall_max, color=year_colors[year_of_max], s=36,
               edgecolor="white", linewidth=0.8, zorder=10)
    ax.scatter(date_hl, max_hl, color=year_colors[HIGHLIGHT_YEAR], s=36,
               edgecolor="white", linewidth=0.8, zorder=11)

    # text inside the box, top-left
    ax.text(0.015, 0.93, label, transform=ax.transAxes, fontsize=FS_LABEL,
            fontweight="bold", va="top")
    ax.text(0.015, 0.72, f"Max {HIGHLIGHT_YEAR}: {max_hl:.2f}", transform=ax.transAxes,
            fontsize=FS_MAX, color=year_colors[HIGHLIGHT_YEAR], va="top")
    ax.text(0.015, 0.56, f"Max 2019–2026: {overall_max:.2f} ({year_of_max})",
            transform=ax.transAxes, fontsize=FS_MAX, color=year_colors[year_of_max], va="top")

    ax.set_ylim(0, Y_MAX)
    # only 0-6 are labelled: an "8" at the top of a panel would sit on the "0" of the panel above
    ax.yaxis.set_major_locator(mticker.FixedLocator([0, 2, 4, 6]))
    ax.grid(axis="y", color="0.88", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FS_TICK, direction="in", length=4, width=1,
                   top=True, right=True)
    for sp in ax.spines.values():
        sp.set_linewidth(1)

# x-axis: Jan-Dec on the bottom panel only
axes[-1].set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[-1].set_xlabel("Month", fontsize=FS_AX)

fig.supylabel(r"Sea ice extent (× 10$^6$ km$^2$)", fontsize=FS_AX)

# one common legend, inside the IOS panel (top-right, where that panel is empty)
handles = [Line2D([0], [0], color=year_colors[y], linewidth=LW_HL if y == HIGHLIGHT_YEAR else 2.2,
                  label=str(y)) for y in year_colors]
axes[-1].legend(handles=handles, loc="upper right", ncol=4, frameon=False,
                fontsize=FS_LEG, handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
                borderaxespad=0.6)

plt.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.savefig(out_png, dpi=1000, bbox_inches="tight")
print("Saved:", out_png)
plt.show()
