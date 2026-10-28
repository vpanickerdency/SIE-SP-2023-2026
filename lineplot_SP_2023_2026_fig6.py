"""
Figure 6 - Total daily Antarctic sea ice extent from EOS-06 SCAT-3, 2023 vs 2026
-------------------------------------------------------------------------------
Reads Sheet1 of an Excel file (row 1 = years, rows 2-367 = 1 Jan .. 31 Dec on a
366-day calendar; the 29 Feb row is blank in non-leap years; an optional text
"Date" column is ignored) and draws the daily extent of the chosen years.

Changes from the previous version
  * Plots EOS-06 SCAT-3 (the product named in the Figure 6 caption). NSIDC can be
    overlaid as thin dashed lines for comparison, or switched off.
  * The August dip is measured on the fixed event window used in the paper,
    7 -> 11 August, instead of the automatic "highest point before the low".
  * The product name appears in the legend and the console output, so the
    figure and the caption can be checked against each other.
  * The inset shows the event endpoints to 3 decimals. With 2 decimals they were
    rounded separately from the drop (e.g. 16.34 -> 15.96 next to -0.39), so a
    reader subtracting them got 0.38; with 3 decimals they match the drop.

Requirements:  pip install openpyxl matplotlib
"""

import datetime as dt

import openpyxl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.family"] = "DejaVu Sans"   # same font on every computer

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
FOLDER = r"C:\Users\vpani\Downloads\Sept_SIE_SP"
SHEET_NAME = "Sheet1"

# Main product (Figure 6) and optional comparison product
MAIN_FILE = FOLDER + r"\EED35920.xlsx"      # <-- EOS-06 SCAT-3 file
MAIN_LABEL = "EOS-06 SCAT-3"
COMPARE_FILE = None    # NSIDC overlay for checking: FOLDER + r"\SP_September_new.xlsx"
COMPARE_LABEL = "NSIDC"

OUT_PNG = FOLDER + r"\Fig6_SP_SIE_SCAT3_2023_vs_2026.png"

YEARS = {                      # year -> (colour, line width, max-label offset in points (x, y))
    2023: ("#C0392B", 3.5, (15, -150)),    # label well below-right of the 2023 maximum, clear of the curve
    2026: ("#1F5F8B", 3.0, (-40, 30)),     # label above-left of the 2026 maximum
}
MARK_MAXIMUM = True            # dot + label at each year's maximum (main product only)

# Zoom-in inset on the August 2026 event
SHOW_ZOOM = True
ZOOM_YEAR = 2026
ZOOM_START, ZOOM_END = (7, 25), (8, 31)           # (month, day): 25 Jul - 31 Aug
EVENT_START, EVENT_END = (8, 7), (8, 11)          # fixed event window: 7 -> 11 August
INSET_POS = [0.06, 0.44, 0.31, 0.50]              # inset position in axes fraction [x, y, w, h]

FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_NOTE = 22, 24, 24, 20, 20
FS_INSET = 17
FS_INSET_TICK = 18             # inset x/y tick labels


# ---------------------------------------------------------------
# READ DATA
# ---------------------------------------------------------------
def read_series(path, years):
    """Return {year: [(axis_date, value), ...]} from a year-column sheet."""
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    rows = list(wb[SHEET_NAME].iter_rows(values_only=True))
    header = rows[0]
    # common leap-year calendar so every year lines up by day of year
    axis_dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(len(rows) - 1)]
    out = {}
    for year in years:
        if year not in header:
            raise ValueError(f"{year} is not a column in {path}")
        col = header.index(year)
        out[year] = [(axis_dates[i], float(r[col])) for i, r in enumerate(rows[1:])
                     if col < len(r) and isinstance(r[col], (int, float))]
    return out


def value_on(pts, month, day):
    d = dt.date(2020, month, day)
    for dd, v in pts:
        if dd == d:
            return v
    raise ValueError(f"No value on {d:%d %b}")


main = read_series(MAIN_FILE, YEARS)
comp = read_series(COMPARE_FILE, YEARS) if COMPARE_FILE else None

print(f"--- {MAIN_LABEL} ({MAIN_FILE}) ---")
for year, pts in main.items():
    vmax = max(pts, key=lambda p: p[1])
    print(f"{year}: {len(pts)} days, last {pts[-1][0]:%d %b}, "
          f"max {vmax[1]:.3f} on {vmax[0]:%d %b}")
if comp:
    print(f"--- {COMPARE_LABEL} ({COMPARE_FILE}) ---")
    for year, pts in comp.items():
        vmax = max(pts, key=lambda p: p[1])
        print(f"{year}: max {vmax[1]:.3f} on {vmax[0]:%d %b}")

# ---------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

for year, (color, lw, offset) in YEARS.items():
    d = [p[0] for p in main[year]]
    v = [p[1] for p in main[year]]
    ax.plot(d, v, color=color, linewidth=lw, label=f"{year} ({MAIN_LABEL})", zorder=3)

    if comp:
        ax.plot([p[0] for p in comp[year]], [p[1] for p in comp[year]], color=color,
                linewidth=1.3, linestyle="--", alpha=0.8,
                label=f"{year} ({COMPARE_LABEL})", zorder=2)

    if MARK_MAXIMUM:
        md, mv = max(main[year], key=lambda p: p[1])
        ax.scatter([md], [mv], s=140, color=color, edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate(f"{year} max: {mv:.2f}\n({md:%d %b})", xy=(md, mv),
                    xytext=offset, textcoords="offset points",
                    ha="left" if offset[0] > 0 else "right",
                    va="bottom" if offset[1] > 0 else "top",
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.2),
                    fontsize=FS_NOTE, fontweight="bold", color=color, zorder=6)

ax.set_xlabel("Month", fontsize=FS_LABEL)
ax.set_ylabel(r"Sea Ice Extent ($\times\,10^{6}$ km$^{2}$)", fontsize=FS_LABEL)

ax.set_xlim(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))   # labels centred in each month
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())                # faint line at each month start
ax.grid(axis="x", which="minor", color="#DDDDDD", linewidth=0.8)
ax.grid(axis="y", which="major", linestyle=":", alpha=0.6)
ax.tick_params(axis="both", which="major", labelsize=FS_TICK, length=8, width=2)
ax.tick_params(axis="x", which="minor", length=0)

all_vals = [v for pts in main.values() for _, v in pts]
ax.set_ylim(0, max(all_vals) + 3.6)          # room for the maximum labels

ax.legend(fontsize=FS_LEGEND - 2 if comp else FS_LEGEND,
          loc="lower right" if SHOW_ZOOM else "upper left", frameon=False)

# ---------------------------------------------------------------
# ZOOM-IN INSET: August 2026 event, fixed 7 -> 11 August window
# ---------------------------------------------------------------
if SHOW_ZOOM:
    z0, z1 = dt.date(2020, *ZOOM_START), dt.date(2020, *ZOOM_END)
    color = YEARS[ZOOM_YEAR][0]
    zpts = [(d, v) for d, v in main[ZOOM_YEAR] if z0 <= d <= z1]

    e0, e1 = dt.date(2020, *EVENT_START), dt.date(2020, *EVENT_END)
    v0 = value_on(main[ZOOM_YEAR], *EVENT_START)
    v1 = value_on(main[ZOOM_YEAR], *EVENT_END)
    drop = v0 - v1
    ndays = (e1 - e0).days
    rec = next(((d, v) for d, v in zpts if d > e1 and v >= v0), None)
    print(f"\n{MAIN_LABEL} {ZOOM_YEAR}: {v0:.3f} on {e0:%d %b} -> {v1:.3f} on {e1:%d %b} "
          f"(-{drop:.3f} in {ndays} days)"
          + (f"; back above {v0:.3f} on {rec[0]:%d %b}" if rec else ""))
    if comp:
        c0 = value_on(comp[ZOOM_YEAR], *EVENT_START)
        c1 = value_on(comp[ZOOM_YEAR], *EVENT_END)
        print(f"{COMPARE_LABEL} {ZOOM_YEAR}: -{c0 - c1:.3f} over the same window")

    axin = ax.inset_axes(INSET_POS)
    all_zoom_vals = []
    for yr, (yc, ylw, _) in YEARS.items():
        ypts = [(d, v) for d, v in main[yr] if z0 <= d <= z1]
        all_zoom_vals += [v for _, v in ypts]
        axin.plot([d for d, _ in ypts], [v for _, v in ypts], color=yc, linewidth=2.6,
                  marker="o", markersize=4.5, label=str(yr), zorder=3)
    axin.axvspan(e0, e1, color=color, alpha=0.10, zorder=0)      # shade 7-11 August
    axin.scatter([e0, e1], [v0, v1], s=90, color=color, edgecolor="white",
                 linewidth=1.0, zorder=5)
    mid = e0 + (e1 - e0) / 2
    axin.annotate(f"−{drop:.2f} in {ndays} days\n"
                  f"{v0:.3f} → {v1:.3f}\n"      # 3 decimals so the endpoints subtract to the drop
                  f"({e0:%d} → {e1:%d %b})",
                  xy=(mid, max(v0, v1) + 0.30), ha="center", va="bottom", linespacing=1.15,
                  fontsize=FS_INSET, fontweight="bold", color=color, zorder=6)

    axin.set_xlim(z0, z1)
    axin.set_ylim(min(all_zoom_vals) - 0.2, max(all_zoom_vals) + 0.95)
    axin.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 11, 21]))
    axin.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axin.tick_params(labelsize=FS_INSET_TICK)
    axin.grid(linestyle=":", alpha=0.6)
    axin.set_title(f"Zoom {z0:%d %b} – {z1:%d %b} ({MAIN_LABEL})", fontsize=FS_INSET)
    axin.legend(fontsize=FS_INSET - 1, loc="lower right", frameon=False)
    for sp in axin.spines.values():
        sp.set_edgecolor("#555555")
    ax.indicate_inset_zoom(axin, edgecolor="#555555", alpha=0.9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=1000, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
plt.show()
