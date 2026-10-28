"""
Figure 10 (a, b, c) from the final ABS August daily data (1991-2026).
Panel titles removed (only (a), (b), (c) tags kept); all label, tick, legend and annotation fonts enlarged.
Calculations are unchanged, so every value printed on the figure is identical to the previous version.

Input : ABS_August_daily_1991_2026.xlsx (sheet "August_daily")
Output: Fig10_ABS_wind_response.png (600 dpi), Fig10_numbers.txt
Keep this script and the Excel file in the same folder, then press Run (F5).
Requirements: pip install pandas numpy matplotlib openpyxl
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import builtins
_log = []
def print(*a, **k):
    builtins.print(*a, **k)
    _log.append(" ".join(str(x) for x in a))
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
FOLDER = os.path.dirname(os.path.abspath(__file__))   # the folder this script is in
IN_XLSX = os.path.join(FOLDER, "ABS_August_daily_1991_2026.xlsx")
OUT_FIG10 = os.path.join(FOLDER, "Fig10_ABS_wind_response.png")
OUT_TXT = os.path.join(FOLDER, "Fig10_numbers.txt")

FIT_YEARS = range(1991, 2026)      # climatological fit, 2026 excluded
ALL_YEARS = range(1991, 2027)
EVENT = (7, 11)                    # 7-11 August wind window
CHANGE = (6, 11)                   # 6-11 August extent change
EPISODE_MIN_WIND = 2.0             # northerly episode: >= 2 consecutive days with 10 m index >= 2 m/s
DPI = 600

# Font sizes
FS_BASE = 15        # default text
FS_LABEL = 16       # axis labels
FS_TICK = 16        # tick labels
FS_LEGEND = 14      # legends
FS_ANNOT = 14       # annotations / values on the plot
FS_PANEL = 18       # (a), (b), (c) tags

C26, CBAND = "#6A51A3", "#e3e3e3"
CPAST, CPAST_TXT = "#1E7B34", "#1B5E2B"     # 1991-2025 dots and their year labels (green)
CLOSS, CGAIN, CEPI, CEPI_TXT = "#D95F0E", "#6A51A3", "#F4CCCC", "#C0392B"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS_BASE,
                     "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK,
                     "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND})

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
d = pd.read_excel(IN_XLSX, sheet_name="August_daily", parse_dates=["Date"]).set_index("Date")
w10 = d["Northerly_10m_ms"]
sie = d["SIE_ABS_1e6km2"] * 1000.0          # 10^3 km^2 (five-day trailing mean)


def day(y, dd):
    return pd.Timestamp(year=y, month=8, day=dd)


def five_day_change(y, dd):
    """Exact change in daily extent from day dd-5 to day dd: 5 x day-to-day change of the trailing mean."""
    return 5.0 * (sie[day(y, dd)] - sie[day(y, dd - 1)])


def window_mean(series, y, d0, d1):
    return series[day(y, d0):day(y, d1)].mean()


def fit_line(x, y):
    b, a = np.polyfit(x, y, 1)                  # y = a + b x
    res = y - (a + b * x)
    sd = res.std(ddof=2)
    r = np.corrcoef(x, y)[0, 1]
    return a, b, sd, r

# ---------------------------------------------------------------------------
# 2. EVENT TEST (Figure 10a)
# ---------------------------------------------------------------------------
x_ev = pd.Series({y: window_mean(w10, y, *EVENT) for y in ALL_YEARS})
y_ev = pd.Series({y: five_day_change(y, CHANGE[1]) for y in ALL_YEARS})
a, b, sd, r = fit_line(x_ev[list(FIT_YEARS)].values, y_ev[list(FIT_YEARS)].values)
pred = a + b * x_ev[2026]
ev10 = dict(x=x_ev, y=y_ev, a=a, b=b, sd=sd, r=r, pred=pred, resid=y_ev[2026] - pred)
print(f"(a) r = {r:.2f}; residual {ev10['resid']:.0f} x10^3 km^2 ({ev10['resid'] / sd:.1f} SD)")

# ---------------------------------------------------------------------------
# 3. FIVE-DAY RELATION (Figure 10b, 10c)
# ---------------------------------------------------------------------------
END_DAYS = range(5, 32)                       # five-day windows ending 5-31 August
X5, Y5 = [], []
for y in FIT_YEARS:
    for dd in END_DAYS:
        X5.append(window_mean(w10, y, dd - 4, dd))
        Y5.append(five_day_change(y, dd))
a5, b5, sd5, r5 = fit_line(np.array(X5), np.array(Y5))

var_ice = pd.Series({y: np.std([five_day_change(y, dd) for dd in END_DAYS], ddof=1) for y in ALL_YEARS})
var_wind = pd.Series({y: np.std([window_mean(w10, y, dd - 4, dd) for dd in END_DAYS], ddof=1) for y in ALL_YEARS})
rank_ice = int(var_ice.rank(ascending=False)[2026])
rank_wind = int(var_wind.rank(ascending=False)[2026])
print(f"(b) ice-edge variability rank {rank_ice}; wind variability rank {rank_wind}")

obs26 = pd.Series({dd: five_day_change(2026, dd) for dd in END_DAYS})
x26 = pd.Series({dd: window_mean(w10, 2026, dd - 4, dd) for dd in END_DAYS})
exp26 = a5 + b5 * x26
z26 = (obs26 - exp26) / sd5

# northerly episodes from the daily 10 m index
strong = (w10["2026-08-01":"2026-08-31"] >= EPISODE_MIN_WIND).values
episodes, i = [], 0
while i < len(strong):
    if strong[i]:
        j = i
        while j + 1 < len(strong) and strong[j + 1]:
            j += 1
        if j - i + 1 >= 2:
            episodes.append((i + 1, j + 1))
        i = j + 1
    else:
        i += 1


def episode_windows(e0, e1):
    """Windows (by last day) whose centre day (dd-2) lies in the episode e0-e1."""
    return [dd for dd in END_DAYS if e0 + 2 <= dd <= e1 + 2]


ep_z, ep_win = [], []
for e0, e1 in episodes:
    sel = episode_windows(e0, e1)
    ep_win.append(sel)
    ep_z.append(z26[sel].min() if sel else np.nan)
gain1 = z26[[dd for dd in END_DAYS if 14 <= dd <= 20]]
gain2 = z26[[dd for dd in END_DAYS if 24 <= dd <= 31]]
print("(c) episodes: " + ", ".join(f"{e0}-{e1} Aug ({z:+.1f} SD)" for (e0, e1), z in zip(episodes, ep_z)))
print(f"(c) regrowth: {gain1.max():+.1f} SD, {gain2.max():+.1f} SD")

# ---------------------------------------------------------------------------
# 4. FIGURE 10
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 12.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.34, wspace=0.36)
axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])


def panel_tag(ax, tag):
    ax.text(0.025, 0.975, tag, transform=ax.transAxes, ha="left", va="top", zorder=10,
            fontsize=FS_PANEL, fontweight="bold")


def tidy(ax):
    ax.grid(color="0.9", ls=":", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# (a) ice loss vs wind
x, y = ev10["x"], ev10["y"]
xs = np.linspace(-4, 6, 50)
axa.fill_between(xs, ev10["a"] + ev10["b"] * xs - ev10["sd"], ev10["a"] + ev10["b"] * xs + ev10["sd"],
                 color=CBAND, lw=0, zorder=0)
axa.plot(xs, ev10["a"] + ev10["b"] * xs, "--", color="#444444", lw=1.5, zorder=1)
fy = list(FIT_YEARS)
axa.scatter(x[fy], y[fy], s=40, color=CPAST, edgecolor="white", lw=0.6, zorder=2)
axa.scatter(x[2026], y[2026], s=140, color=C26, edgecolor="black", lw=0.8, zorder=4)
axa.annotate("", xy=(x[2026], y[2026] + 8), xytext=(x[2026], ev10["pred"]),
             arrowprops=dict(arrowstyle="<->", color=C26, lw=1.5), zorder=3)
axa.text(x[2026], y[2026] - 25, f"{ev10['resid']:.0f}×10³ km²\n({ev10['resid'] / ev10['sd']:.1f} SD)",
         ha="center", va="top", color=C26, fontweight="bold", fontsize=FS_ANNOT)
LABEL_POS = {2017: ((-8, 0), "right", "center"), 2018: ((0, 9), "center", "bottom"),
             2000: ((0, 9), "center", "bottom")}
for yr in y[fy].sort_values().head(3).index:          # three largest losses before 2026
    off, ha, va = LABEL_POS.get(yr, ((7, 2), "left", "center"))
    axa.annotate(str(yr), (x[yr], y[yr]), xytext=off, textcoords="offset points",
                 ha=ha, va=va, color=CPAST_TXT, fontsize=FS_ANNOT)
axa.axhline(0, color="k", lw=0.7)
axa.set_ylim(y.min() - 130, y.max() + 90)
axa.set_xlim(xs[0], xs[-1])
axa.set_xlabel("Mean northerly 10 m wind, 7-11 Aug (m/s)")
axa.set_ylabel("ABS extent change,\n6-11 Aug (10³ km²)")
axa.yaxis.set_major_locator(MultipleLocator(100))
tidy(axa)
axa.legend(handles=[Patch(color=CBAND, label="Fit ± 1 residual SD"),
                    Line2D([], [], ls="--", color="#444444", label=f"Fit 1991-2025 (r = {ev10['r']:.2f})"),
                    Line2D([], [], ls="", marker="o", color=CPAST, label="1991-2025"),
                    Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor=C26,
                           markeredgecolor="black", label="2026")],
           loc="lower left", frameon=False, handlelength=1.4, borderaxespad=0.3, labelspacing=0.35)
panel_tag(axa, "(a)")

# (b) variability
axb.scatter(var_wind[fy], var_ice[fy], s=40, color=CPAST, edgecolor="white", lw=0.6)
axb.scatter(var_wind[2026], var_ice[2026], s=140, color=C26, edgecolor="black", lw=0.8, zorder=3)
for yr in var_ice[fy].sort_values(ascending=False).head(2).index:
    axb.annotate(str(yr), (var_wind[yr], var_ice[yr]), xytext=(7, 0), textcoords="offset points",
                 va="center", color=CPAST_TXT, fontsize=FS_ANNOT)
axb.annotate(f"2026: ice-edge variability rank {rank_ice} of 36\nwind variability rank {rank_wind} of 36",
             xy=(var_wind[2026], var_ice[2026]), xytext=(0.5, 0.88), textcoords="axes fraction",
             ha="center", va="top", color=C26, fontweight="bold", fontsize=FS_ANNOT,
             arrowprops=dict(arrowstyle="-|>", color=C26, lw=1.3))
axb.set_ylim(min(var_ice) * 0.82, max(var_ice) * 1.48)
axb.set_xlabel("Wind variability, August\n(SD of 5-day wind index, m/s)")
axb.set_ylabel("Ice-edge variability, August\n(SD of 5-day change, 10³ km²)")
tidy(axb)
axb.legend(handles=[Line2D([], [], ls="", marker="o", color=CPAST, label="1991-2025"),
                    Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor=C26,
                           markeredgecolor="black", label="2026")],
           loc="center left", bbox_to_anchor=(0.0, 0.55), frameon=False)
panel_tag(axb, "(b)")

# (c) 2026 five-day changes
dd = np.array(list(END_DAYS))
for sel in [w for w in ep_win if w]:     # shade the windows centred within each episode
    axc.axvspan(sel[0] - 0.45, sel[-1] + 0.45, color=CEPI, alpha=0.6, lw=0, zorder=0)
axc.fill_between(dd, exp26 - sd5, exp26 + sd5, color="#d9d9d9", alpha=0.8, lw=0, zorder=1)
axc.bar(dd, obs26.values, width=0.72, color=[CLOSS if v < 0 else CGAIN for v in obs26.values],
        alpha=0.9, zorder=2)
axc.plot(dd, exp26.values, "-o", color="#404040", ms=4, lw=1.8, zorder=3)
axc.axhline(0, color="k", lw=0.7)
ymin, ymax = min(obs26.min(), (exp26 - sd5).min()), max(obs26.max(), (exp26 + sd5).max())
axc.set_ylim(ymin - 260, ymax + 100)
main_k = int(np.nanargmin(ep_z)) + 1
for k, (sel, zk) in enumerate(zip(ep_win, ep_z), 1):
    if not sel:
        continue
    main = k == main_k
    axc.text((sel[0] + sel[-1]) / 2, ymin - 245, f"Episode {k}\n{zk:.1f} SD",
             ha="center", va="bottom", fontsize=FS_ANNOT,
             color=CEPI_TXT if main else "#555555", fontweight="bold" if main else "normal")
for g in (gain1, gain2):
    run = [g.idxmax()]                       # consecutive gain windows around the largest departure
    while run[0] - 1 in obs26.index and obs26[run[0] - 1] > 0:
        run.insert(0, run[0] - 1)
    while run[-1] + 1 in obs26.index and obs26[run[-1] + 1] > 0:
        run.append(run[-1] + 1)
    top = max(obs26[run].max(), (exp26[run] + sd5).max())
    axc.text(np.mean(run), top + 18, f"{g.max():+.1f} SD", ha="center", va="bottom",
             color=CGAIN, fontweight="bold", fontsize=FS_ANNOT)
axc.set_xlim(dd[0] - 0.6, dd[-1] + 0.6)
axc.set_xticks([5, 10, 15, 20, 25, 30])
axc.set_xticklabels([f"{t} Aug" for t in [5, 10, 15, 20, 25, 30]])
axc.set_xlabel("Last day of each 5-day window, August 2026")
axc.set_ylabel("ABS 5-day extent\nchange (10³ km²)")
tidy(axc)
axc.legend(handles=[Patch(color=CLOSS, label="Observed loss"), Patch(color=CGAIN, label="Observed gain"),
                    Line2D([], [], marker="o", ms=4, color="#404040",
                           label="Expected from wind (1991-2025 relationship)"),
                    Patch(color="#d9d9d9", label="Expected ± 1 SD"), Patch(color=CEPI, label="Northerly episodes")],
           loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=16)
panel_tag(axc, "(c)")

fig.savefig(OUT_FIG10, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_FIG10}")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(_log) + "\n")
builtins.print(f"Saved: {OUT_TXT}")
