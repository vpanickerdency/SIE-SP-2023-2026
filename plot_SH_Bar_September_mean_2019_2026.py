"""
Antarctic September mean sea ice extent, 2019-2026.

Source: Sept_2019_2016.xlsx (Sheet1) - daily extents for 1-30 September, one column per year.
Each bar is the mean of the daily September extents. Bars are coloured by value:
    blue  : > 18 x 10^6 km^2
    green : 17-18 x 10^6 km^2
    red   : < 17 x 10^6 km^2
The legend lists the years in each colour class.

Requirements: pip install pandas matplotlib openpyxl
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

EXCEL_PATH = r"C:\Users\vpani\Downloads\FINAL\Sept_2019_2016.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_PNG = r"C:\Users\vpani\Downloads\FINAL\SH_September_mean_extent_2019_2026.png"
YEARS = list(range(2019, 2027))
MIN_DAYS = 15             # a September mean needs at least this many daily values

# ---------------------------------------------------------------------------
# 1. SEPTEMBER MEANS
# ---------------------------------------------------------------------------
raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)
header, data = raw.iloc[0], raw.iloc[1:].copy()

sept_mean = {}
for j, h in enumerate(header):
    try:
        y = int(float(h))
    except (TypeError, ValueError):
        continue
    s = pd.to_numeric(data[j], errors="coerce")
    if s.notna().sum() >= MIN_DAYS:
        sept_mean[y] = float(s.mean())

for y in YEARS:
    print(y, f"{sept_mean.get(y, float('nan')):.2f}")

# ---------------------------------------------------------------------------
# 2. STYLE  (three colours, assigned by value)
# ---------------------------------------------------------------------------
BLUE, GREEN, RED = "#4AA9C4", "#2E8B57", "#B33A2E"

def bar_colour(v):
    if v > 18:
        return BLUE
    if v >= 17:
        return GREEN
    return RED

def year_text(yrs):
    """Turn [2019, 2020, 2021, 2023] into '2019–2021, 2023'."""
    yrs, parts, start = sorted(yrs), [], None
    for i, y in enumerate(yrs):
        start = y if start is None else start
        if i == len(yrs) - 1 or yrs[i + 1] != y + 1:
            parts.append(str(y) if start == y else f"{start}–{y}")
            start = None
    return ", ".join(parts)

# years falling in each colour class, written into the legend
groups = {BLUE: [], GREEN: [], RED: []}
for y in YEARS:
    if y in sept_mean:
        groups[bar_colour(sept_mean[y])].append(y)

LEGEND = [
    (BLUE,  r"> 18 × 10$^6$ km$^2$"),
    (GREEN, r"> 17 × 10$^6$ km$^2$"),
    (RED,   r"< 17 × 10$^6$ km$^2$"),
]
LEGEND = [(c, f"{year_text(groups[c])} ({l})") for c, l in LEGEND if groups[c]]

FS_VALUE, FS_TICK, FS_LEG, FS_AX = 16, 17, 17, 18

# ---------------------------------------------------------------------------
# 3. PLOT
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.5))
bw = 0.6
for x, y in enumerate(YEARS):
    if y not in sept_mean:
        continue
    v = sept_mean[y]
    c = bar_colour(v)
    ax.add_patch(FancyBboxPatch((x - bw / 2, 0), bw, v,
                                boxstyle=f"round,pad=0,rounding_size={0.06 * bw}",
                                linewidth=0, facecolor=c, zorder=3))
    ax.text(x, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=FS_VALUE,
            fontweight="bold", color=c, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

ax.set_xticks(range(len(YEARS)))
ax.set_xticklabels([str(y) for y in YEARS], fontsize=FS_TICK)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.set_ylabel(r"September mean extent (× 10$^6$ km$^2$)", fontsize=FS_AX)
ax.set_xlabel("Year", fontsize=FS_AX)
ax.set_ylim(0, max(sept_mean[y] for y in YEARS if y in sept_mean) + 1.6)
ax.set_xlim(-0.7, len(YEARS) - 0.3)
for sp in ax.spines.values():
    sp.set_color("black")
    sp.set_linewidth(1.2)

plt.tight_layout()

# Legend: bold, spread across exactly the width of the plot box.
# If the text is too wide for the box, the font is reduced step by step until it fits.
handles = [Patch(facecolor=c, label=l) for c, l in LEGEND]
LEG_KW = dict(handles=handles, ncol=len(handles), frameon=False, borderaxespad=0, borderpad=0,
              handlelength=1.5, handleheight=1.5, handletextpad=0.5, columnspacing=1.0)

fig.canvas.draw()
ax_width = ax.get_window_extent().width
fs = FS_LEG
while fs > 8:
    leg = ax.legend(loc="upper center", prop={"size": fs, "weight": "bold"}, **LEG_KW)
    fig.canvas.draw()
    if leg.get_window_extent().width <= ax_width:
        break
    fs -= 0.5

# place it below the "Year" label, stretched to the plot's left and right edges
fig.canvas.draw()
inv = ax.transAxes.inverted()
label_bottom = inv.transform((0, ax.xaxis.label.get_window_extent().y0))[1]
ax.legend(loc="upper left", mode="expand", bbox_to_anchor=(0, label_bottom - 0.04, 1, 0),
          prop={"size": fs, "weight": "bold"}, **LEG_KW)
print(f"Legend font size used: {fs}")

plt.savefig(OUTPUT_PNG, dpi=1000, bbox_inches="tight")
print(f"Saved: {OUTPUT_PNG}")
plt.show()
