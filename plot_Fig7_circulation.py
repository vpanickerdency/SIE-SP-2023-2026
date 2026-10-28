"""
How unusual was the August 2026 ABS pressure dipole and northerly flow?  (ERA5, Augusts 1979-2026)
---------------------------------------------------------------------------------------------------
Baseline: 1979-2008, same calendar days, 00 UTC, and records over the satellite era (1979 onward),
as in Jena et al. (2024). Falls back to 1991-2020 if the 1979-1990 file is not present.

Figure panels (the daily-index and longest-spell panels of earlier versions are no longer
plotted; their statistics are still printed and saved in the CSV):
a) MSLP anomaly, mean of 7-11 Aug 2026 minus the baseline mean of 7-11 Aug; stippled where
   |anomaly| >= 2 standard deviations of the baseline 7-11 Aug means. Thin lines: 2026 MSLP.
b) 7-11 Aug mean northerly index for each year from PLOT_START, with the 2026 rank in that period,
   and the strongest 2026 day in 7-11 Aug compared with every August day of the earlier years.
c) Amundsen Sea Low (Hosking et al. 2016 style): longitude and relative central pressure of the
   minimum in 60-75 S, 170 E-62 W (ocean), 7-11 Aug mean of each year; 2026 highlighted.

The geostrophic index is the same as in the earlier figures (validated against ERA5 10 m wind:
r = 0.94, 10 m wind = 0.74 x geostrophic). Only MSLP is needed, so it can be computed for every year.

Console: anomaly and z-score at the high and low centres, ranks and percentiles, ASL statistics.
CSV   : one row per year with all indices.

Requirements:  pip install xarray netCDF4 numpy pandas scipy matplotlib basemap

Figure 7 revision:
  * baseline fixed at 1991-2020 (as in the caption); it no longer switches to 1979-2008 when the
    1979-1990 file happens to be present
  * panel c shows the same years as panel b (PLOT_START-2026); the 2026 ASL rank is in its legend
  * panel b y-label names the geostrophic index; shorter 2026 annotation
  * map legend: "MSLP 2026 (every 8 hPa)"; output at 600 dpi
  * layout: (a) map and (c) ASL side by side on top, (b) full width below; no panel titles
    (description in the caption), only (a), (b), (c) tags
  * sector lines start at the coast (drawn under the land) and sector names sit over the ocean
  * other years in (b) and (c) in dark green
  * panel b (and the index statistics) use the FINAL geostrophic index from
    ABS_August_daily_1991_2026.xlsx (column Northerly_geostrophic_ms), the same values as Section 3.4,
    Figure S9d and the earlier Figure 7. The index computed in this script (Gaussian smoothing,
    Basemap coastline mask) is about 11 % weaker and gives rank 6 / 7.1 m/s instead of rank 5 / 8.2 m/s.
    Set USE_FINAL_INDEX = False to go back to the index computed here.
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap, addcyclic
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
BASE = r"C:\Users\vpani\Downloads\Sept_SIE_SP"
# all August MSLP files (any number, any year split); years found in several files are read once
MSLP_FILES = [
    os.path.join(BASE, "1979_1990_Aug_MSLP", "data_stream-oper_stepType-instant.nc"),
    os.path.join(BASE, "1991_2026_Aug_MSLP", "data_stream-oper_stepType-instant.nc"),
]
SIC_FOLDER = os.path.join(BASE, "2026_Jul_Aug_SIC")          # ice edges on map a (optional)
OUT_FIG = os.path.join(BASE, "Fig7_ABS_circulation_August2026.png")
OUT_CSV = os.path.join(BASE, "ABS_northerly_index_August_all_years.csv")
# final geostrophic index (same values as Section 3.4, Figure S9d and Figure 10's data file)
INDEX_XLSX = r"C:\Users\vpani\Downloads\FINAL\Fig.10\ABS_August_daily_1991_2026.xlsx"
USE_FINAL_INDEX = True

TARGET, COMPARE = 2026, 2023
# Baseline as in Jena et al. (2024): anomalies relative to 1979-2008, records over the satellite
# era (1979 onward). If the 1979-1990 file is missing, the script falls back to 1991-2020.
CLIM = (1991, 2020)                  # baseline used in the Figure 7 caption
CLIM_FALLBACK = (1991, 2020)
DIP = (7, 11)                        # days of August
LAT_READ = (-90.0, -35.0)            # only this latitude band is read

BOX_LAT = (-72.0, -60.0)             # ABS northerly index box (same as before)
BOX_LON = (-130.0, -60.0)
SMOOTH_SIGMA_DEG = 1.0
RHO, OMEGA, R_EARTH = 1.25, 7.292e-5, 6.371e6

WINDOW = 5                           # days, running window for the "all windows" percentile
RUN_THRESHOLD = 2.7                  # m/s geostrophic (~2 m/s at 10 m, using the 0.74 ratio)

ASL_LAT = (-75.0, -60.0)             # Amundsen Sea Low search box (Hosking et al. 2016)
ASL_LONX = (-190.0, -62.0)           # 170 E .. 62 W, on the shifted longitude axis

HIGH_SEARCH = dict(lat=(-66, -40), lonx=(-135, -55))   # positive anomaly centre near the ABS
LOW_SEARCH = dict(lat=(-76, -50), lonx=(-190, -120))   # negative anomaly centre to its west

MAP_BOUNDINGLAT = -40
ANOM_LEVELS = np.concatenate([np.arange(-24, -2, 3), np.arange(3, 25, 3)])   # white = -3..+3 hPa
Z_STIPPLE = 2.0
ISOBARS = np.arange(920, 1061, 8)

LAND_COLOUR = "#9E9E9E"
COL_T, COL_C, COL_OTHER = "#6A3D9A", "#D9541E", "#1E7B34"      # 2026, 2023, other years (green)
COL_BAND, COL_MED = "#D9D9D9", "#595959"
EDGE_START = dict(colors="#00C853", linewidths=2.0)
EDGE_END = dict(colors="black", linewidths=1.7)
BOX_STYLE = dict(color="#F5B700", linewidth=2.4, linestyle="--")
SECTOR_LONS = [-130, -60, 20, 90, 160]
# sector names on map a, over the ocean in the middle of each sector (lon, lat)
# ABS sits poleward of the +30 hPa label; IOS and WPS sit north of the (narrow) ice edge
SECTOR_LABELS = {"WSS": (-20, -64), "IOS": (55, -54), "WPS": (125, -54),
                 "RSS": (-165, -68), "ABS": (-100, -69.5)}

FS_TITLE, FS_PANEL, FS_LABEL, FS_TICK, FS_LEGEND = 21, 20, 18, 16, 16
PLOT_START = 1991                    # first year shown in panel b (1991 = the 8.2 m/s figure version)
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "hatch.color": "#1A1A1A", "hatch.linewidth": 0.6})
HALO = [pe.withStroke(linewidth=3.5, foreground="white")]


def shift_lon(lo):
    lo = np.asarray(lo, dtype="float64")
    return np.where(lo > 145, lo - 360.0, lo)


def lon_label(x, _=None):
    lo = x + 360 if x < -180 else x
    if abs(abs(lo) - 180) < 1e-6:
        return "180°"
    if abs(lo) < 1e-6:
        return "0°"
    return f"{abs(lo):.0f}°{'W' if lo < 0 else 'E'}"


def land_at(m, x, y):
    p = np.column_stack([np.ravel(x), np.ravel(y)])
    out = np.zeros(p.shape[0], bool)
    for poly, ptype in zip(m.coastpolygons, m.coastpolygontypes):
        if ptype == 1:
            out |= Path(np.column_stack(poly)).contains_points(p)
    return out.reshape(np.shape(x))


# ---------------------------------------------------------------
# 1. FILE AND GRID
# ---------------------------------------------------------------
sources = {}                          # year -> (dataset, names, time indices in that file)
grid = None
for path in MSLP_FILES:
    if not os.path.exists(path):
        print(f"Not found (skipped): {path}")
        continue
    ds_ = xr.open_dataset(path)
    names = dict(var="msl" if "msl" in ds_.data_vars else list(ds_.data_vars)[0],
                 tn="valid_time" if "valid_time" in ds_.coords else "time",
                 la="latitude" if "latitude" in ds_.coords else "lat",
                 lo="longitude" if "longitude" in ds_.coords else "lon")
    t_ = pd.DatetimeIndex(np.atleast_1d(ds_[names["tn"]].values))
    la_ = np.asarray(ds_[names["la"]].values, dtype="float64")
    lo_ = np.asarray(ds_[names["lo"]].values, dtype="float64")
    if grid is None:
        grid = (la_, lo_)
    elif not (np.allclose(grid[0], la_) and np.allclose(grid[1], lo_)):
        raise ValueError(f"{path} is on a different grid from the first file")
    for yr in sorted(set(t_.year)):
        if yr not in sources:
            sources[yr] = (ds_, names, np.where(t_.year == yr)[0], t_)
    print(f"Read index of {path}: {t_.min():%Y-%m-%d} .. {t_.max():%Y-%m-%d}, {len(t_)} steps")
if not sources:
    raise FileNotFoundError("None of the MSLP files was found")

years = sorted(sources)
if not all(y in sources for y in range(CLIM[0], CLIM[1] + 1)):
    print(f"\nWARNING: baseline {CLIM[0]}-{CLIM[1]} not fully available (data start {years[0]}); "
          f"using {CLIM_FALLBACK[0]}-{CLIM_FALLBACK[1]} instead. Download the missing Augusts to "
          f"match Jena et al. (2024).\n")
    CLIM = CLIM_FALLBACK

lat_all, lon_all = grid
lat_idx = np.where((lat_all >= LAT_READ[0]) & (lat_all <= LAT_READ[1]))[0]
lat_order = np.argsort(lat_all[lat_idx])
lon_order = np.argsort(lon_all)
lat = lat_all[lat_idx][lat_order]
lon = lon_all[lon_order]
dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
print(f"Years {years[0]}..{years[-1]} ({len(years)}), baseline {CLIM[0]}-{CLIM[1]}, "
      f"grid {dlat}° x {dlon}°, latitudes {lat[0]:.1f}..{lat[-1]:.1f}")

LAT2, LON2 = np.meshgrid(lat, lon, indexing="ij")
LONX2 = shift_lon(LON2)
W_LAT = np.cos(np.radians(LAT2))

m_mask = Basemap(projection="spstere", boundinglat=LAT_READ[1] + 1, lon_0=180, resolution="l")
OCEAN = ~land_at(m_mask, *m_mask(LON2, LAT2))

in_box = ((LAT2 >= BOX_LAT[0]) & (LAT2 <= BOX_LAT[1]) &
          (LON2 >= BOX_LON[0]) & (LON2 <= BOX_LON[1]) & OCEAN)
in_asl = ((LAT2 >= ASL_LAT[0]) & (LAT2 <= ASL_LAT[1]) &
          (LONX2 >= ASL_LONX[0]) & (LONX2 <= ASL_LONX[1]) & OCEAN)

phi = np.radians(lat)[:, None]
f_cor = 2 * OMEGA * np.sin(phi)
sig = (SMOOTH_SIGMA_DEG / dlat, SMOOTH_SIGMA_DEG / dlon)


def northerly_index(p_hpa):
    """Area-weighted geostrophic northerly wind (m/s) over the ABS box."""
    p = gaussian_filter(p_hpa * 100.0, sigma=sig, mode=["nearest", "wrap"])
    pe_ = np.concatenate([p[:, -1:], p, p[:, :1]], axis=1)
    dpdx = (pe_[:, 2:] - pe_[:, :-2]) / (2 * np.radians(dlon)) / (R_EARTH * np.cos(phi))
    north = -dpdx / (RHO * f_cor)
    return np.sum(north[in_box] * W_LAT[in_box]) / np.sum(W_LAT[in_box])


def asl(p_hpa):
    """ASL longitude (shifted axis), actual and relative central pressure (hPa)."""
    vals = np.where(in_asl, p_hpa, np.inf)
    i, j = np.unravel_index(np.argmin(vals), vals.shape)
    box_mean = np.sum(p_hpa[in_asl] * W_LAT[in_asl]) / np.sum(W_LAT[in_asl])
    return LONX2[i, j], lat[i], p_hpa[i, j], p_hpa[i, j] - box_mean


# ---------------------------------------------------------------
# 2. ONE PASS OVER THE YEARS
# ---------------------------------------------------------------
daily = {}                            # year -> Series(day of month -> index)
dip_mean = {}                         # year -> 7-11 Aug mean MSLP (hPa)
for yr in years:
    ds_, nm, tid, t_ = sources[yr]
    P = np.asarray(ds_[nm["var"]].isel({nm["tn"]: tid, nm["la"]: lat_idx}).values, dtype="float32")
    P = P[:, lat_order][:, :, lon_order]
    if np.nanmean(P[0]) > 2000:
        P = P / 100.0                                            # Pa -> hPa
    d = t_[tid].day
    daily[yr] = pd.Series([northerly_index(P[k].astype("float64")) for k in range(len(tid))], index=d)
    sel = (d >= DIP[0]) & (d <= DIP[1])
    dip_mean[yr] = P[sel].mean(axis=0)
    print(f"  {yr}: {len(tid)} days, 7-11 Aug index {daily[yr].loc[DIP[0]:DIP[1]].mean():+.2f} m/s")

D = pd.DataFrame(daily)                                          # rows = day of month, cols = years
if USE_FINAL_INDEX:
    if not os.path.exists(INDEX_XLSX):
        raise FileNotFoundError(f"Final index file not found: {INDEX_XLSX}\n"
                                "Set INDEX_XLSX to its location, or USE_FINAL_INDEX = False.")
    fx = pd.read_excel(INDEX_XLSX, sheet_name="August_daily", parse_dates=["Date"])
    fx = fx[fx["Date"].dt.month == 8]
    D_final = fx.pivot_table(index=fx["Date"].dt.day, columns=fx["Date"].dt.year,
                             values="Northerly_geostrophic_ms")
    chk = [y for y in D_final.columns if y in D.columns]
    diff = (D[chk].loc[DIP[0]:DIP[1]].mean() - D_final[chk].loc[DIP[0]:DIP[1]].mean()).abs()
    print(f"\nUsing the final index from {INDEX_XLSX} ({D_final.columns.min()}-{D_final.columns.max()}); "
          f"7-11 Aug {TARGET}: final {D_final[TARGET].loc[DIP[0]:DIP[1]].mean():+.2f} vs computed here "
          f"{D[TARGET].loc[DIP[0]:DIP[1]].mean():+.2f} m/s (mean |difference| over years {diff.mean():.2f})")
    D = D_final
clim_years = [y for y in years if CLIM[0] <= y <= CLIM[1]]
past_years = [y for y in years if y != TARGET]
iyears = [y for y in years if y in D.columns]                    # years with an index value
ipast = [y for y in iyears if y != TARGET]


# ---------------------------------------------------------------
# 3. STATISTICS
# ---------------------------------------------------------------
def longest_run(s, thr):
    best = cur = 0
    for v in s.values:
        cur = cur + 1 if v >= thr else 0
        best = max(best, cur)
    return best


dip_index = D.loc[DIP[0]:DIP[1]].mean()
run_len = pd.Series({y: longest_run(D[y].dropna(), RUN_THRESHOLD) for y in iyears})
roll = D.rolling(WINDOW).mean()
pool = roll[ipast].values.ravel()
pool = pool[np.isfinite(pool)]

t_val = dip_index[TARGET]
rank = int((dip_index > t_val).sum()) + 1
pct = 100.0 * np.mean(pool < t_val)
clim_mu, clim_sd = dip_index[clim_years].mean(), dip_index[clim_years].std(ddof=1)
t_max_win = roll[TARGET].max()
t_max_day = int(roll[TARGET].idxmax())

print("\n=== ABS geostrophic northerly index ===")
print(f"7-11 Aug {TARGET}: {t_val:+.2f} m/s  ({CLIM[0]}-{CLIM[1]} mean {clim_mu:+.2f}, SD {clim_sd:.2f}, "
      f"z = {(t_val - clim_mu) / clim_sd:+.1f})")
print(f"Rank among the {len(iyears)} Augusts (same dates): {rank}")
top5 = dip_index.sort_values(ascending=False).head(5)
print("Top 5 years for 7-11 Aug: " + ", ".join(f"{y} ({v:+.1f})" for y, v in top5.items()))
print(f"Higher than {pct:.1f}% of all {WINDOW}-day August windows {ipast[0]}-{ipast[-1]} "
      f"(n = {pool.size}); record window {pool.max():+.2f} m/s")
print(f"Strongest {WINDOW}-day window in {TARGET}: {t_max_win:+.2f} m/s, ending {t_max_day} Aug "
      f"(higher than {100 * np.mean(pool < t_max_win):.1f}% of past windows)")
print(f"Longest northerly run (index >= {RUN_THRESHOLD} m/s): {TARGET} = {run_len[TARGET]} days; "
      f"{CLIM[0]}-{CLIM[1]} mean {run_len[clim_years].mean():.1f}, max {run_len[ipast].max()} "
      f"({', '.join(str(y) for y in list(run_len[ipast][run_len[ipast] == run_len[ipast].max()].index)[:5])}"
      f"{', ...' if (run_len[ipast] == run_len[ipast].max()).sum() > 5 else ''})")

# anomaly field
stack = np.stack([dip_mean[y] for y in clim_years]).astype("float64")
clim_field = stack.mean(axis=0)
sd_field = stack.std(axis=0, ddof=1)
anom = dip_mean[TARGET] - clim_field
z = anom / np.where(sd_field > 0, sd_field, np.nan)


def centre(field, box, find_max):
    sel = ((LAT2 >= box["lat"][0]) & (LAT2 <= box["lat"][1]) &
           (LONX2 >= box["lonx"][0]) & (LONX2 <= box["lonx"][1]) & OCEAN)
    vals = np.where(sel, field, -np.inf if find_max else np.inf)
    i, j = np.unravel_index(np.argmax(vals) if find_max else np.argmin(vals), field.shape)
    return i, j


hi_i, hi_j = centre(anom, HIGH_SEARCH, True)
lo_i, lo_j = centre(anom, LOW_SEARCH, False)
print("\n=== MSLP anomaly, 7-11 Aug %d vs %d-%d ===" % (TARGET, *CLIM))
for name, (i, j) in (("Positive centre", (hi_i, hi_j)), ("Negative centre", (lo_i, lo_j))):
    print(f"{name}: {anom[i, j]:+.1f} hPa ({z[i, j]:+.1f} SD) at {abs(lat[i]):.1f}°S, "
          f"{lon_label(LONX2[i, j])}; 2026 MSLP {dip_mean[TARGET][i, j]:.0f} hPa")
dipole = dip_mean[TARGET][hi_i, hi_j] - dip_mean[TARGET][lo_i, lo_j]
dipole_clim = [dip_mean[y][hi_i, hi_j] - dip_mean[y][lo_i, lo_j] for y in clim_years]
print(f"Pressure difference between the two centres: {dipole:.0f} hPa "
      f"({CLIM[0]}-{CLIM[1]} mean {np.mean(dipole_clim):.0f} ± {np.std(dipole_clim, ddof=1):.0f} hPa, same points)")

# Amundsen Sea Low
A = pd.DataFrame({y: asl(dip_mean[y].astype("float64")) for y in years},
                 index=["lonx", "lat", "p_central", "p_relative"]).T
print("\n=== Amundsen Sea Low, 7-11 Aug mean ===")
for y in (TARGET, COMPARE):
    print(f"{y}: {lon_label(A.loc[y, 'lonx'])}, {abs(A.loc[y, 'lat']):.1f}°S, "
          f"central {A.loc[y, 'p_central']:.0f} hPa, relative {A.loc[y, 'p_relative']:+.1f} hPa")
print(f"{CLIM[0]}-{CLIM[1]}: longitude {lon_label(A.loc[clim_years, 'lonx'].mean())} ± "
      f"{A.loc[clim_years, 'lonx'].std(ddof=1):.0f}°, relative pressure "
      f"{A.loc[clim_years, 'p_relative'].mean():+.1f} ± {A.loc[clim_years, 'p_relative'].std(ddof=1):.1f} hPa")

out = pd.DataFrame({"index_7_11_Aug": dip_index, f"max_{WINDOW}day_window": roll.max(),
                    "longest_northerly_run_days": run_len,
                    "ASL_lon": [lon_label(v) for v in A["lonx"]], "ASL_lat": A["lat"],
                    "ASL_central_hPa": A["p_central"], "ASL_relative_hPa": A["p_relative"]})
out.index.name = "year"
out.round(3).to_csv(OUT_CSV)
print(f"\nSaved table: {OUT_CSV}")


# ---------------------------------------------------------------
# 4. FIGURE  (one row: a) anomaly map | b) 7-11 Aug flow by year | c) Amundsen Sea Low)
# ---------------------------------------------------------------
# rank and percentile within the period shown in panel b
plot_years = [y for y in iyears if y >= PLOT_START]
plot_past = [y for y in plot_years if y != TARGET]
rank_p = int((dip_index[plot_years] > t_val).sum()) + 1
# daily comparison: strongest 2026 day within 7-11 Aug against every August day of the earlier years
pool_d = D[plot_past].values.ravel()
pool_d = pool_d[np.isfinite(pool_d)]
dip_days = D[TARGET].loc[DIP[0]:DIP[1]]
pk_day, pk_val = int(dip_days.idxmax()), float(dip_days.max())
pct_d = 100.0 * np.mean(pool_d < pk_val)
print(f"\nPanel b period {plot_years[0]}-{plot_years[-1]}: {TARGET} 7-11 Aug mean rank {rank_p} of {len(plot_years)}")
print(f"Strongest day {pk_day} Aug {TARGET}: {pk_val:+.2f} m/s, higher than {pct_d:.1f}% of all "
      f"{pool_d.size} August days {plot_past[0]}-{plot_past[-1]}")

# layout: (a) map | (c) ASL on top, (b) full width below (its top is set after the map legend is placed)
fig = plt.figure(figsize=(16, 13.5))
ax_map = fig.add_axes([0.005, 0.50, 0.45, 0.485])       # becomes square (map aspect); (c) is fitted to it below
ax_asl = fig.add_axes([0.55, 0.56, 0.43, 0.42])
ax_rank = fig.add_axes([0.075, 0.055, 0.91, 0.25])


def panel_tag(ax, tag, x=0.012, y=0.975):
    ax.text(x, y, tag, transform=ax.transAxes, ha="left", va="top", fontsize=FS_PANEL,
            fontweight="bold", zorder=20)


def ycol(y):
    return COL_T if y == TARGET else COL_OTHER


# ---------------- a) anomaly map ----------------
plt.sca(ax_map)
m = Basemap(projection="spstere", boundinglat=MAP_BOUNDINGLAT, lon_0=180, resolution="l", ax=ax_map)
ac, lonc = addcyclic(anom, lon)
zc, _ = addcyclic(np.abs(z), lon)
pc, _ = addcyclic(dip_mean[TARGET].astype("float64"), lon)
LONC2, LATC2 = np.meshgrid(lonc, lat)
XC, YC = m(LONC2, LATC2)
land_c = land_at(m, XC, YC)
hide = land_c | (LATC2 > MAP_BOUNDINGLAT)

_base = plt.get_cmap("RdBu_r")
_cols = [_base(v) for v in np.linspace(0.05, 0.95, len(ANOM_LEVELS) - 1)]
_cols[len(_cols) // 2] = (1, 1, 1, 1)
ANOM_CMAP = ListedColormap(_cols)
ANOM_CMAP.set_under(_base(0.0))
ANOM_CMAP.set_over(_base(1.0))
ANOM_NORM = BoundaryNorm(ANOM_LEVELS, ANOM_CMAP.N)

cs = m.contourf(XC, YC, np.ma.masked_where(hide, ac), levels=ANOM_LEVELS, cmap=ANOM_CMAP,
                norm=ANOM_NORM, extend="both", zorder=1)
m.contourf(XC, YC, np.ma.masked_where(hide, zc), levels=[Z_STIPPLE, 1e9], colors="none",
           hatches=[".."], zorder=2)
m.contour(XC, YC, np.ma.masked_where(hide, pc), levels=ISOBARS, colors="#4D4D4D",
          linewidths=0.6, zorder=3)
m.fillcontinents(color=LAND_COLOUR, lake_color=LAND_COLOUR, zorder=4)
m.drawcoastlines(linewidth=0.8, zorder=5)
m.drawparallels(np.arange(-80, MAP_BOUNDINGLAT + 0.1, 10), labels=[0, 0, 0, 0], linewidth=0.4)
m.drawmeridians(np.arange(-180, 181, 30), labels=[0, 0, 0, 0], linewidth=0.4)

for day, style in ((DIP[0], EDGE_START), (DIP[1], EDGE_END)):
    found = glob.glob(os.path.join(SIC_FOLDER, "**", f"ice_conc_sh_*_{TARGET}08{day:02d}*.nc"),
                      recursive=True)
    if found:
        with xr.open_dataset(found[0]) as s:
            ic = np.asarray(s["ice_conc"].isel(time=0).values, dtype="float32")
            sx, sy = m(np.asarray(s["lon"].values), np.asarray(s["lat"].values))
        m.contour(sx, sy, np.ma.masked_invalid(ic), levels=[15], zorder=6, **style)

for L in SECTOR_LONS:
    ll = np.linspace(-90, MAP_BOUNDINGLAT, 200)
    m.plot(*m(np.full_like(ll, L), ll), color="black", linewidth=0.9, zorder=3.5)   # under the land (zorder 4)
bl = np.concatenate([np.full(50, BOX_LAT[0]), np.linspace(*BOX_LAT, 50),
                     np.full(50, BOX_LAT[1]), np.linspace(*BOX_LAT[::-1], 50)])
bo = np.concatenate([np.linspace(*BOX_LON, 50), np.full(50, BOX_LON[1]),
                     np.linspace(*BOX_LON[::-1], 50), np.full(50, BOX_LON[0])])
m.plot(*m(bo, bl), zorder=8, **BOX_STYLE)
for sname, (lo_pt, la_pt) in SECTOR_LABELS.items():
    ax_map.text(*m(lo_pt, la_pt), sname, fontsize=15, fontweight="bold", ha="center", va="center",
                color="black", zorder=10, path_effects=HALO)

for (i, j), col in (((hi_i, hi_j), "#B2182B"), ((lo_i, lo_j), "#2166AC")):
    px, py = m(LON2[i, j], lat[i])
    ax_map.text(px, py, f"{anom[i, j]:+.0f} hPa\n({z[i, j]:+.1f} SD)", color=col, fontsize=16,
                fontweight="bold", ha="center", va="center", zorder=10, path_effects=HALO)

clip = Circle((0.5, 0.5), 0.5, transform=ax_map.transAxes)
for art in ax_map.collections + ax_map.patches + ax_map.lines + ax_map.images:
    art.set_clip_path(clip)
ax_map.add_patch(Circle((0.5, 0.5), 0.5, transform=ax_map.transAxes, fill=False, lw=1.6, zorder=12))
ax_map.axis("off")

# ---------------- b) 7-11 Aug index by year, from PLOT_START ----------------
yrs = np.array(plot_years)
ax_rank.axhspan(clim_mu - clim_sd, clim_mu + clim_sd, color=COL_BAND, alpha=0.7, lw=0)
ax_rank.axhline(clim_mu, color=COL_MED, lw=1.4, ls="--")
ax_rank.bar(yrs, dip_index[plot_years].values, color=[ycol(y) for y in plot_years], width=0.72, zorder=3)
ax_rank.axhline(0, color="#555555", lw=0.9)
day_text = ("highest of all August days" if pct_d >= 99.95
            else f"above {pct_d:.0f}% of all August days")
ax_rank.annotate(f"{TARGET}: rank {rank_p} of {len(plot_years)}\n"
                 f"strongest day: {pk_day} Aug, {pk_val:.1f} m/s\n({day_text}, {plot_past[0]}–{plot_past[-1]})",
                 xy=(TARGET, t_val), xytext=(0.985, 0.97), textcoords="axes fraction", ha="right", va="top",
                 multialignment="right", fontsize=FS_TICK, fontweight="bold", color=COL_T,
                 arrowprops=dict(arrowstyle="-|>", color=COL_T, lw=1.6, shrinkA=4, shrinkB=4,
                                 relpos=(0.97, 0.0), connectionstyle="arc3,rad=0"))
ax_rank.set_xlim(yrs[0] - 0.8, yrs[-1] + 0.8)
lo_v, hi_v = dip_index[plot_years].min(), dip_index[plot_years].max()
ax_rank.set_ylim(min(lo_v, clim_mu - clim_sd) - 0.8, hi_v + 0.28 * (hi_v - min(lo_v, 0)))
YEAR_TICKS = [y for y in plot_years if y % 5 == 0 and abs(y - TARGET) > 2] + [TARGET]
ax_rank.set_xticks(YEAR_TICKS)
ax_rank.tick_params(labelsize=FS_TICK)
ax_rank.set_ylabel(f"Mean northerly geostrophic\nwind, {DIP[0]}–{DIP[1]} Aug (m/s)", fontsize=FS_LABEL)
ax_rank.grid(axis="y", linestyle=":", alpha=0.6)
ax_rank.legend(handles=[Patch(color=COL_T, label=str(TARGET)),
                        Patch(color=COL_OTHER, label="Other years"),
                        Patch(color=COL_BAND, label=f"{CLIM[0]}–{CLIM[1]} mean ± 1 SD")],
               loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=FS_LEGEND,
               frameon=False, handlelength=1.3, columnspacing=1.2)

# ---------------- c) Amundsen Sea Low (2026 highlighted) ----------------
oth = [y for y in plot_years if y != TARGET]                     # same years as panel b
rank_asl = int(A.loc[plot_years, "p_relative"].rank()[TARGET])     # 1 = deepest
ordinal = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
print(f"ASL {TARGET}: {ordinal(rank_asl)} deepest relative central pressure of {len(plot_years)} "
      f"({plot_years[0]}-{plot_years[-1]})")
ax_asl.scatter(A.loc[oth, "lonx"], A.loc[oth, "p_relative"], s=55, color=COL_OTHER,
               edgecolor="white", zorder=3, label="Other years")
mx, my = A.loc[clim_years, "lonx"].mean(), A.loc[clim_years, "p_relative"].mean()
ax_asl.errorbar(mx, my, xerr=A.loc[clim_years, "lonx"].std(ddof=1),
                yerr=A.loc[clim_years, "p_relative"].std(ddof=1), fmt="X", ms=13, color="black",
                ecolor="black", elinewidth=1.4, capsize=5, zorder=4, label=f"{CLIM[0]}–{CLIM[1]} mean ± 1 SD")
ax_asl.scatter(A.loc[TARGET, "lonx"], A.loc[TARGET, "p_relative"], s=170, color=COL_T, edgecolor="black",
               zorder=5, label=f"{TARGET} ({ordinal(rank_asl)} deepest of {len(plot_years)})")
ax_asl.annotate(str(TARGET), (A.loc[TARGET, "lonx"], A.loc[TARGET, "p_relative"]), xytext=(10, 6),
                textcoords="offset points", fontsize=FS_TICK, fontweight="bold", color=COL_T,
                path_effects=HALO, zorder=6)
ax_asl.invert_yaxis()                                            # deeper low = higher on the plot
ax_asl.xaxis.set_major_locator(plt.FixedLocator([-180, -150, -120, -90, -60]))
ax_asl.xaxis.set_major_formatter(plt.FuncFormatter(lon_label))
ax_asl.set_xlim(ASL_LONX[0] - 3, ASL_LONX[1] + 3)
ax_asl.tick_params(labelsize=FS_TICK)
ax_asl.set_xlabel("ASL longitude", fontsize=FS_LABEL)
ax_asl.set_ylabel("ASL relative central\npressure (hPa, deeper ↑)", fontsize=FS_LABEL)
ax_asl.grid(linestyle=":", alpha=0.6)
ax_asl.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=FS_LEGEND,
              frameon=False, handletextpad=0.3, columnspacing=1.0)

# ---------------- colour bar and map legend under the map ----------------
fig.canvas.draw()
inv = fig.transFigure.inverted()
pm_ = ax_map.get_position()
cax = fig.add_axes([pm_.x0 + 0.04 * pm_.width, pm_.y0 - 0.045, 0.92 * pm_.width, 0.022])
cb = fig.colorbar(cs, cax=cax, orientation="horizontal", extend="both")
cb.set_ticks([-24, -18, -12, -6, -3, 3, 6, 12, 18, 24])
cb.ax.tick_params(labelsize=FS_TICK - 1)
cb.set_label(f"MSLP anomaly vs {CLIM[0]}–{CLIM[1]} (hPa)", fontsize=FS_LABEL - 1)
fig.canvas.draw()
cb_bottom = inv.transform(cb.ax.xaxis.label.get_window_extent())[0, 1]
fig.legend(handles=[Patch(facecolor="white", edgecolor="#333333", hatch="..",
                          label=f"|anomaly| ≥ {Z_STIPPLE:g} SD"),
                    Line2D([], [], color="#4D4D4D", lw=1.0, label=f"MSLP {TARGET} (every 8 hPa)"),
                    Line2D([], [], label="ABS index box", **BOX_STYLE),
                    Line2D([], [], color=EDGE_START["colors"], lw=2.2, label=f"Ice edge, {DIP[0]} Aug"),
                    Line2D([], [], color=EDGE_END["colors"], lw=1.9, label=f"Ice edge, {DIP[1]} Aug")],
           loc="upper center", bbox_to_anchor=(pm_.x0 + pm_.width / 2, cb_bottom - 0.006), ncol=2,
           fontsize=FS_LEGEND - 1, frameon=False, columnspacing=1.2, handlelength=1.8)

# ---------------- final layout: (b) under the map legend, panel tags ----------------
fig.canvas.draw()
inv = fig.transFigure.inverted()
# (c): from just right of the map circle to the right edge, same top as the map; its legend sits
# under its x-axis label, beside the map colour bar
pm_ = ax_map.get_position()
c_x0 = pm_.x1 + 0.075                                     # room for the (c) y-axis label
ax_asl.set_position([c_x0, pm_.y0 + 0.01, 0.985 - c_x0, pm_.y1 - 0.004 - (pm_.y0 + 0.01)])
fig.canvas.draw()
c_lab_bottom = inv.transform(ax_asl.xaxis.label.get_window_extent())[0, 1]
pc_ = ax_asl.get_position()
ax_asl.get_legend().set_bbox_to_anchor((pc_.x0 + pc_.width / 2, c_lab_bottom - 0.008), transform=fig.transFigure)
fig.canvas.draw()
map_leg_bottom = inv.transform(fig.legends[-1].get_window_extent())[0, 1]
c_leg_bottom = inv.transform(ax_asl.get_legend().get_window_extent())[0, 1]
b_top = min(map_leg_bottom, c_leg_bottom) - 0.03
pb_ = ax_rank.get_position()
ax_rank.set_position([pb_.x0, pb_.y0, pb_.width, b_top - pb_.y0])
panel_tag(ax_map, "(a)", x=0.0, y=1.0)
panel_tag(ax_rank, "(b)")
panel_tag(ax_asl, "(c)")

# no figure title: the description goes in the caption
plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_FIG}")
plt.show()
