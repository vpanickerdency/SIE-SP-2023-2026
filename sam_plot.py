"""SAM (AAO) index at three time scales: daily, monthly and annual.
(a) NOAA CPC daily AAO index (sheet SAM_daily).
(b) NOAA CPC official monthly AAO index (sheet SAM_monthly_CPC), so values match CPC's table.
(c) Annual means of the official monthly index.

Run from Spyder (Excel file in the same folder as this script), or:
    python sam_plot.py SAM_daily_CPC.xlsx [out.png]
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "DejaVu Sans"   # same font on every computer

# Default: the Excel file sits in the same folder as this script.
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
SRC = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "SAM_daily_CPC.xlsx")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "SAM_2014_2026.png")

START, END = 2014, 2026
HIGHLIGHT, HIGHLIGHT2 = 2023, 2026

GREY, BLUE = "black", "#3a8ade"
HI, HI_DARK = "#d95a32", "#9c3a1f"      # 2023
HI2, HI2_DARK = "#7b4fc9", "#55309a"    # 2026

# ------------------------------------------------------------------ data
full = pd.read_excel(SRC, sheet_name="SAM_daily", parse_dates=["date"]).dropna(subset=["sam"])
REC0 = int(full.year.min())
d = full[(full.year >= START) & (full.year <= END)]

# Official CPC monthly index (computed by CPC from monthly-mean fields, so it is
# slightly different from the mean of the daily values)
monfull = pd.read_excel(SRC, sheet_name="SAM_monthly_CPC", parse_dates=["date"]).dropna(subset=["sam"])
mon = monfull[(monfull.year >= START) & (monfull.year <= END)].copy()

ann = mon.groupby("year").sam.agg(["mean", "size"])      # size = number of months
mu, sd = ann["mean"].mean(), ann["mean"].std()
last_m = d.date.max().strftime("%b")

# ------------------------------------------------------------------ ranks (for the footnotes)
def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"

def month_note(year, month):
    """Rank of a month against the same calendar month in every other year."""
    same = monfull[monfull.month == month].set_index("year").sam
    rank = int((same > same.loc[year]).sum()) + 1
    name = pd.Timestamp(2000, month, 1).strftime("%B")
    plural = name[:-1] + "ies" if name.endswith("y") else name + "s"
    span = f"all {len(same)} {plural} ({int(same.index.min())}–{int(same.index.max())})"
    if rank == 1:
        nxt = same.drop(year).idxmax()
        return f"the highest of {span}; next is {name} {nxt} ({same.loc[nxt]:+.2f})"
    top = same.idxmax()
    return (f"the {ordinal(rank)}-highest of {span}; "
            f"highest is {name} {top} ({same.loc[top]:+.2f})")

def fmt_day(t):
    return f"{t.day} {t:%b %Y}"   # portable (no %-d, which fails on Windows)

day_top = full.loc[full.sam.idxmax()]
def day_note(row):
    rank = int((full.sam > row.sam).sum()) + 1
    if rank == 1:
        return f"the highest of all {len(full):,} days since {REC0}"
    return (f"the {ordinal(rank)}-highest of all {len(full):,} days since {REC0}; "
            f"highest is {fmt_day(day_top.date)} ({day_top.sam:+.3f})")

def month_short(year, month):
    """Two-line box text: rank against the same calendar month in all years."""
    same = monfull[monfull.month == month].set_index("year").sam
    rank = int((same > same.loc[year]).sum()) + 1
    name = pd.Timestamp(2000, month, 1).strftime("%B")
    span = f"{len(same)} {name}s ({int(same.index.min())}–{int(same.index.max())})".replace("ys (", "ies (")
    if rank == 1:
        return f"highest of all {span}"
    top = same.idxmax()
    return f"{ordinal(rank)}-highest of {span}\nhighest: {name[:3]} {top} ({same.loc[top]:+.2f})"

def day_short(row):
    rank = int((full.sam > row.sam).sum()) + 1
    what = "highest" if rank == 1 else f"{ordinal(rank)}-highest"
    return f"{what} of {len(full):,} days since {REC0}"

jan_note = month_note(HIGHLIGHT, 1)
pk_m = int(mon[mon.year == HIGHLIGHT2].set_index("month").sam.idxmax())
pk_name = pd.Timestamp(2000, pk_m, 1).strftime("%b")
pk_note = month_note(HIGHLIGHT2, pk_m)
dpk = full[full.year == HIGHLIGHT2].nlargest(1, "sam").iloc[0]
dpk_note = day_note(dpk)

# ------------------------------------------------------------------ figure
# One shared time axis for all three panels: year labels appear only once, at the bottom
fig, (ax0, ax1, ax2) = plt.subplots(
    3, 1, figsize=(14, 16), sharex=True,
    gridspec_kw=dict(height_ratios=[1.1, 1.1, 1], hspace=0.06))
BOX = dict(boxstyle="round,pad=0.3", fc="white", lw=1)

def year_span(ax):
    ax.axvspan(pd.Timestamp(f"{HIGHLIGHT}-01-01"), pd.Timestamp(f"{HIGHLIGHT+1}-01-01"),
               color=HI, alpha=0.07, lw=0, zorder=0)
    ax.axvspan(pd.Timestamp(f"{HIGHLIGHT2}-01-01"), d.date.max() + pd.Timedelta(days=15),
               color=HI2, alpha=0.07, lw=0, zorder=0)
    ax.axhline(0, color="#999", lw=0.8, zorder=1)

def date_axis(ax):
    ax.set_xlim(pd.Timestamp(f"{START-1}-10-01"), pd.Timestamp(f"{END+1}-03-01"))
    # Ticks at 1 January mark the year boundaries; the year label sits in the
    # middle of each year (1 July), which is also where the annual bars are drawn
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7, bymonthday=1))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="y", ls=":", color="#ccc")
    ax.grid(axis="x", which="major", ls=":", color="#e2e2e2")
    ax.tick_params(labelsize=13)
    ax.tick_params(axis="x", which="minor", labelsize=13)

def panel_label(ax, tag):
    """Panel letter inside the top-left corner of the axes."""
    ax.text(0.012, 0.975, tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=17, fontweight="bold", zorder=20,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))

# ---------- (a) daily ----------
year_span(ax0)
ax0.plot(d.date, d.sam, color=GREY, lw=0.45, zorder=2, label="Daily value")
for yr, col, lab in [(HIGHLIGHT, HI, f"{HIGHLIGHT}"),
                     (HIGHLIGHT2, HI2, f"{HIGHLIGHT2} (Jan–{last_m})")]:
    s = d[d.year == yr]
    ax0.plot(s.date, s.sam, color=col, lw=0.9, zorder=3, label=lab)

ax0.annotate(f"{fmt_day(dpk.date)}: {dpk.sam:+.3f}\n{day_short(dpk)}",
             xy=(dpk.date, dpk.sam), xytext=(pd.Timestamp("2023-10-01"), 6.9),
             ha="center", va="center", fontsize=12, color=HI2_DARK,
             bbox={**BOX, "ec": HI2_DARK},
             arrowprops=dict(arrowstyle="->", color=HI2_DARK, lw=1.4,
                             connectionstyle="arc3,rad=-0.2"))
if START <= day_top.year <= END and day_top.year != HIGHLIGHT2:
    ax0.plot(day_top.date, day_top.sam, "o", ms=9, mfc="none", mec="#333", mew=1.3, zorder=4)
    ax0.annotate(f"{fmt_day(day_top.date)}: {day_top.sam:+.3f}\nhighest of {len(full):,} days since {REC0} (record)",
                 xy=(day_top.date, day_top.sam), xytext=(pd.Timestamp("2018-07-01"), 6.9),
                 ha="center", va="center", fontsize=12, color="#333",
                 bbox={**BOX, "ec": "#777"},
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1.2,
                                 connectionstyle="arc3,rad=0.2"))
ax0.set_ylim(-6.6, 8.2)   # room for boxes (top) and legend (bottom)
ax0.set_yticks(range(-6, 7, 2))
date_axis(ax0)
panel_label(ax0, "(a)")
ax0.set_ylabel("SAM index (daily)", fontsize=15)
leg = ax0.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#ccc",
                 fontsize=12, ncol=3, borderaxespad=0.5)
for line in leg.get_lines():
    line.set_linewidth(2.2)

# ---------- (b) monthly ----------
year_span(ax1)
ax1.plot(mon.date, mon.sam, "-o", color=GREY, lw=1.0, ms=2.6, zorder=2,
         label="Monthly index (CPC)")
h = mon[mon.year == HIGHLIGHT]
ax1.plot(h.date, h.sam, "-o", color=HI, lw=2.6, ms=6, zorder=3, label=f"{HIGHLIGHT}")
h2 = mon[mon.year == HIGHLIGHT2]
ax1.plot(h2.date, h2.sam, "-o", color=HI2, lw=2.6, ms=6, zorder=3,
         label=f"{HIGHLIGHT2} (Jan–{last_m})")

jan_pt = h[h.month == 1].iloc[0]
ax1.annotate(f"Jan {HIGHLIGHT}: {jan_pt.sam:+.2f}\n{month_short(HIGHLIGHT, 1)}",
             xy=(jan_pt.date, jan_pt.sam), xytext=(pd.Timestamp("2019-01-01"), 4.0),
             ha="center", va="center", fontsize=12, color=HI_DARK,
             bbox={**BOX, "ec": HI_DARK},
             arrowprops=dict(arrowstyle="->", color=HI_DARK, lw=1.4,
                             connectionstyle="arc3,rad=0.15"))
pk_pt = h2[h2.month == pk_m].iloc[0]
ax1.annotate(f"{pk_name} {HIGHLIGHT2}: {pk_pt.sam:+.2f}\n{month_short(HIGHLIGHT2, pk_m)}",
             xy=(pk_pt.date, pk_pt.sam), xytext=(pd.Timestamp("2023-09-01"), 4.25),
             ha="center", va="center", fontsize=12, color=HI2_DARK,
             bbox={**BOX, "ec": HI2_DARK},
             arrowprops=dict(arrowstyle="->", color=HI2_DARK, lw=1.4,
                             connectionstyle="arc3,rad=-0.2"))
ax1.set_ylim(-3.4, 5.3)   # room for boxes (top) and legend (bottom)
ax1.set_yticks(range(-3, 5, 1))
date_axis(ax1)
panel_label(ax1, "(b)")
ax1.set_ylabel("SAM index (monthly)", fontsize=15)
ax1.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#ccc",
           fontsize=12, ncol=3, borderaxespad=0.5)

# ---------- (c) annual ----------
years = ann.index.values
xmid = pd.to_datetime([f"{y}-07-01" for y in years])   # bar centres = middle of each year
colors = [HI if y == HIGHLIGHT else HI2 if y == HIGHLIGHT2 else BLUE for y in years]
ax2.axhspan(mu - sd, mu + sd, color="#e6e6e3", alpha=0.6, lw=0, zorder=0,
            label=f"±1 SD of annual means ({sd:.2f})")
bars = ax2.bar(xmid, ann["mean"], color=colors, width=220, zorder=2)   # width in days
for b, n in zip(bars, ann["size"]):
    if n < 12:   # partial year: hatch it
        b.set_hatch("///"); b.set_edgecolor("white"); b.set_linewidth(0)
ax2.axhline(mu, ls="--", color="#444", lw=1, zorder=3,
            label=f"{START}-{END} mean = {mu:.2f}")
pad = 0.035
for x, y, v, n in zip(xmid, years, ann["mean"], ann["size"]):
    txt = f"{v:.2f}"
    if n < 12:
        txt += f"\n(Jan–{mon[mon.year == y].date.max():%b})"
    ax2.text(x, v + (pad if v >= 0 else -pad), txt, ha="center",
             va="bottom" if v >= 0 else "top", fontsize=12,
             color=HI_DARK if y == HIGHLIGHT else HI2_DARK if y == HIGHLIGHT2 else "#444")
ax2.axhline(0, color="#999", lw=0.8, zorder=1)
ymin, ymax = ann["mean"].min(), ann["mean"].max()
ax2.set_ylim(min(ymin - 0.15, -0.3), ymax + 0.42)
ax2.yaxis.set_major_locator(MultipleLocator(0.2))
panel_label(ax2, "(c)")
ax2.set_ylabel("SAM index (annual)", fontsize=15)
date_axis(ax2)
ax2.set_xlabel("Year", fontsize=15)
ax2.legend(loc="upper left", bbox_to_anchor=(0.045, 1.0), frameon=False, fontsize=12, ncol=2)

for ax in (ax0, ax1):
    ax.tick_params(axis="x", which="both", labelbottom=False)
fig.align_ylabels((ax0, ax1, ax2))

# Data source and ranking method go in the figure caption:
# NOAA CPC AAO index (700 hPa); monthly ranks compare each month with the same calendar month.
fig.savefig(OUT, dpi=1000, bbox_inches="tight")
print("saved", OUT)
