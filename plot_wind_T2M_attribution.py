"""
Daily ABS extent change in August 2026 explained by wind and T2M (daily regression, 1991-2025).

(a) Observed daily change in ABS extent vs the change expected from the 10 m northerly wind and T2M,
    with the expected value +/- 1 residual SD.
(b) Contributions of wind and T2M to the expected change (stacked bars) and the residual
    (observed - expected). The wind and T2M parts are departures from average conditions; the dashed
    line is the regression constant (mean daily change), so constant + wind part + T2M part = expected.
The 7-11 August wind window is shaded.

Input : ABS_wind_T2M_regression_DAILY.xlsx (sheets "2026_attribution" and "Model_stats")
Output: Fig_wind_T2M_attribution.png (600 dpi)
Keep this script and the Excel file in the same folder, then press Run (F5).
Requirements: pip install pandas numpy matplotlib openpyxl
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
FOLDER = os.path.dirname(os.path.abspath(__file__))       # the folder this script is in
IN_XLSX = os.path.join(FOLDER, "ABS_wind_T2M_regression_DAILY.xlsx")
OUT_PNG = os.path.join(FOLDER, "Fig_wind_T2M_attribution.png")
EVENT = ("2026-08-07", "2026-08-11")       # shaded wind window
DPI = 600

FS_BASE, FS_LABEL, FS_TICK, FS_LEGEND, FS_PANEL = 15, 16, 16, 14, 18
C_OBS, C_EXP, C_BAND = "black", "#2C5D8F", "#C9D9EA"
C_WIND, C_T2M, C_RES, C_EVENT = "#2C5D8F", "#E8922E", "#C0392B", "#F4CCCC"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS_BASE,
                     "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK,
                     "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND})

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
d = pd.read_excel(IN_XLSX, sheet_name="2026_attribution", parse_dates=["Date"]).set_index("Date")
stats = pd.read_excel(IN_XLSX, sheet_name="Model_stats").set_index("Statistic")["Value"]
sd = float(stats[[s for s in stats.index if s.startswith("Residual SD")][0]])
obs, exp_ = d["dSIE"], d["Expected"]
wind, t2m, res = d["Wind part"], d["T2M part"], d["Residual"]
t = d.index
const = (exp_ - wind - t2m).mean()          # regression constant: expected = const + wind part + T2M part
assert (exp_ - wind - t2m - const).abs().max() < 0.05, "parts do not add up to the expected change"

print(f"Residual SD = {sd:.1f} x10^3 km^2 per day; R2 = {stats['R2']:.2f}")
print(f"Constant (mean daily change) = {const:+.1f} x10^3 km^2 per day")
print(f"Largest negative residual: {res.min():.1f} on {res.idxmin():%d %b} ({res.min() / sd:.1f} SD)")
print(f"Largest positive residual: {res.max():.1f} on {res.idxmax():%d %b} ({res.max() / sd:.1f} SD)")

# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
fig, (axa, axb) = plt.subplots(2, 1, figsize=(13, 11), sharex=True,
                               gridspec_kw=dict(hspace=0.12, height_ratios=[1, 1]))
e0 = pd.Timestamp(EVENT[0]) - pd.Timedelta(hours=12)
e1 = pd.Timestamp(EVENT[1]) + pd.Timedelta(hours=12)


def tidy(ax, tag):
    ax.axvspan(e0, e1, color=C_EVENT, alpha=0.6, lw=0, zorder=0)
    ax.axhline(0, color="#777777", lw=0.9, zorder=1)
    ax.grid(axis="y", color="0.9", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.012, 0.975, tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_PANEL, fontweight="bold", zorder=10)


# (a) observed vs expected
axa.fill_between(t, exp_ - sd, exp_ + sd, color=C_BAND, lw=0, alpha=0.9, zorder=2)
axa.plot(t, exp_, color=C_EXP, lw=2.2, zorder=3)
axa.plot(t, obs, "-o", color=C_OBS, lw=2.2, ms=5, zorder=4)
axa.set_ylabel("Daily change in ABS extent\n(10³ km² per day)")
tidy(axa, "(a)")
lo, hi = min(obs.min(), (exp_ - sd).min()), max(obs.max(), (exp_ + sd).max())
axa.set_ylim(lo - 0.08 * (hi - lo), hi + 0.10 * (hi - lo))

# (b) contributions: negative parts stack downward, positive parts upward
w = pd.Timedelta(hours=18)
pos_w, neg_w = wind.clip(lower=0), wind.clip(upper=0)
pos_t, neg_t = t2m.clip(lower=0), t2m.clip(upper=0)
axb.bar(t, pos_w, width=w, color=C_WIND, zorder=2)
axb.bar(t, pos_t, width=w, bottom=pos_w, color=C_T2M, zorder=2)
axb.bar(t, neg_w, width=w, color=C_WIND, zorder=2)
axb.bar(t, neg_t, width=w, bottom=neg_w, color=C_T2M, zorder=2)
axb.axhline(const, color="#555555", lw=1.4, ls="--", zorder=3)
axb.scatter(t, res, marker="D", s=55, color=C_RES, edgecolor="white", lw=0.5, zorder=4)
axb.set_ylabel("Contribution or residual\n(10³ km² per day)")
tidy(axb, "(b)")
axb.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[5, 10, 15, 20, 25, 30]))
axb.xaxis.set_minor_locator(mdates.DayLocator())
axb.xaxis.set_major_formatter(mdates.DateFormatter("%-d Aug" if os.name != "nt" else "%#d Aug"))
axb.set_xlim(t[0] - pd.Timedelta(days=0.7), t[-1] + pd.Timedelta(days=0.7))
axb.set_xlabel("Day, August 2026")
axb.legend(handles=[Line2D([], [], color=C_OBS, lw=2.2, marker="o", ms=5, label="Observed"),
                    Line2D([], [], color=C_EXP, lw=2.2, label="Expected from wind and T2M"),
                    Patch(color=C_BAND, label="Expected ± 1 residual SD"),
                    Patch(color=C_EVENT, alpha=0.6, label="7-11 Aug wind window"),
                    Line2D([], [], ls="", marker="D", ms=8, color=C_RES, label="Residual (observed - expected)"),
                    Patch(color=C_WIND, label="Wind part"), Patch(color=C_T2M, label="T2M part"),
                    Line2D([], [], color="#555555", lw=1.4, ls="--",
                           label=f"Mean daily change ({const:+.1f})")],
           loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=16,
           columnspacing=1.2, handlelength=1.8)

fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved:", OUT_PNG)
