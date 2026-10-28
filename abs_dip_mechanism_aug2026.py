"""
Northerly wind episodes and ABS sea ice loss, August 2026  (the "mechanism" figure)
-----------------------------------------------------------------------------------
a) 7-11 Aug: mean northerly 10 m wind (-v10), mean MSLP, and the sea ice LOST between
   7 and 11 Aug (hatched), regional map of the Amundsen-Bellingshausen region.
b) Hovmoller (longitude x day): northerly wind averaged over the ice-edge band (60-72 S).
c) Hovmoller (longitude x day): daily sea ice extent change in each 2-degree longitude strip.
d) Daily ABS extent change (bars) and ABS northerly wind index (line), August; northerly
   episodes found automatically, shaded and numbered, with the net ice change of each.
e) Scatter: wind index (00 UTC, day t) vs extent change ending on day t, with r; dip episode highlighted.

Sign: ERA5 v10 > 0 = wind toward the north. Northerly wind (pushes ice south) = -v10 > 0.

Console: a table of ice change and wind by 10-degree longitude for the dip (to check the
index box), the episodes, and the correlation.

Requirements:  pip install xarray netCDF4 numpy pandas matplotlib basemap
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
BASE = r"C:\Users\vpani\Downloads\Sept_SIE_SP"
V10_FILE = os.path.join(BASE, "Aug_V10_2026_SP", "data_stream-oper_stepType-instant.nc")
MSLP_FILE = os.path.join(BASE, "Aug_MSLP_2026_SP", "data_stream-oper_stepType-instant.nc")
# optional: ERA5 10 m u-wind for the arrows on map a. If this file is missing, the arrows show the
# geostrophic wind computed from the MSLP above (same method as the pressure-pattern figure).
U10_FILE = os.path.join(BASE, "Aug_U10_2026_SP", "data_stream-oper_stepType-instant.nc")
SIC_FOLDER = os.path.join(BASE, "2026_Jul_Aug_SIC")
# saved in the FINAL folder under a new name (the old PNG may be open in a viewer, which blocks overwriting)
OUT_FIG = r"C:\Users\vpani\Downloads\FINAL\ABS_dip_mechanism_Aug2026_v2.png"

YEAR, MONTH = 2026, 8
DIP = (7, 11)                        # map a: wind/MSLP mean and ice lost between these days

ABS_LON = (-130.0, -60.0)            # ABS sector, used for the extent (as in the sector table)
BOX_LAT = (-72.0, -60.0)             # wind index box (same as the earlier figures)
BOX_LON = (-130.0, -60.0)
BAND_LAT = BOX_LAT                   # latitude band averaged in the wind Hovmoller

EPISODE_MIN_WIND = 2.0               # m/s: northerly episode = index at or above this ...
EPISODE_MIN_DAYS = 2                 # ... for at least this many consecutive days

HOV_LON = (-200.0, -40.0)            # Hovmoller longitude range (-200 = 160 E)
HOV_BIN = 2.0                        # degrees
CELL_AREA_KM2 = 625.0                # OSI SAF EASE2 25 km grid
ICE_THRESHOLD = 15.0

# regional map (oblique stereographic, north up at its centre)
MAP_CENTRE = (-66.0, -110.0)         # lat, lon
MAP_SIZE_M = (5.6e6, 3.9e6)          # width, height

WIND_LEVELS = np.concatenate([np.arange(-16, -1, 2), np.arange(2, 17, 2)])   # white = -2..+2
ICE_LEVELS = np.array([-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8])             # thousand km2 / day / 2 deg
ISOBARS = np.arange(920, 1061, 8)

LAND_COLOUR = "#9E9E9E"
COL_WIND, COL_LOSS, COL_GAIN = "#1F3A93", "#D9711C", "#6A51A3"
COL_EPISODE, COL_DIP = "#F4C7C3", "#C0392B"
EDGE_START = dict(colors="#00C853", linewidths=2.2)
EDGE_END = dict(colors="black", linewidths=1.9)
BOX_STYLE = dict(color="#F5B700", linewidth=2.4, linestyle="--")
SECTORS = {"RSS": (160, -130), "ABS": (-130, -60), "WSS": (-60, 20)}   # for labels on b/c

HOV_SMOOTH_BINS = 3                  # running mean over 3 strips (6 deg) in longitude, b and c only
WIND_CONTOUR = 6.0                   # m/s: northerly contour from b drawn on c
SHOW_TITLE = False                   # False for the journal version (title goes in the caption)

# wind arrows on map a
VEC_STEP_DEG = 2.2                   # spacing of arrows (degrees of latitude; longitude spacing widened
                                     # towards the pole so the arrows are evenly spread on the map)
VEC_MIN = 3.0                        # m/s, weaker arrows not drawn
VEC_SCALE = 260                      # larger = shorter arrows
VEC_KEY = 10                         # m/s, reference arrow
RHO, OMEGA, R_EARTH = 1.25, 7.292e-5, 6.371e6

FS_TITLE, FS_PANEL, FS_LABEL, FS_TICK, FS_LEGEND = 26, 26, 21, 19, 19
FS_CBTICK = 20                       # colour-bar tick labels
plt.rcParams.update({"hatch.color": "black", "hatch.linewidth": 0.9,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]})


def shift_lon(lo):
    """Longitudes east of 145 E become negative (e.g. 160 E -> -200) so the Pacific is continuous."""
    lo = np.asarray(lo, dtype="float64")
    return np.where(lo > 145, lo - 360.0, lo)


# ---------------------------------------------------------------
# 1. SEA ICE: daily concentration 31 Jul - 31 Aug
# ---------------------------------------------------------------
sic_days = pd.date_range(f"{YEAR}-{MONTH - 1:02d}-31", f"{YEAR}-{MONTH:02d}-31", freq="D")
SIC, SLAT, SLON = {}, None, None
for d in sic_days:
    found = glob.glob(os.path.join(SIC_FOLDER, "**", f"ice_conc_sh_*_{d:%Y%m%d}*.nc"), recursive=True)
    if not found:
        continue
    with xr.open_dataset(found[0]) as s:
        SIC[d] = s["ice_conc"].isel(time=0).values.astype("float32")
        if SLAT is None:
            SLAT = s["lat"].values.astype("float64")
            SLON = s["lon"].values.astype("float64")
missing = [f"{d:%d %b}" for d in sic_days if d not in SIC]
print(f"Sea ice: {len(SIC)} daily files" + (f"; missing {missing}" if missing else ""))
if pd.Timestamp(YEAR, MONTH, DIP[0]) not in SIC or pd.Timestamp(YEAR, MONTH, DIP[1]) not in SIC:
    raise FileNotFoundError("Sea ice files for the dip start / end days are needed")

SLONX = shift_lon(SLON)
abs_mask = (SLON >= ABS_LON[0]) & (SLON < ABS_LON[1])
x_edges = np.arange(HOV_LON[0], HOV_LON[1] + 0.01, HOV_BIN)
x_mid = 0.5 * (x_edges[1:] + x_edges[:-1])


def covered(ic):
    return np.nan_to_num(ic, nan=0.0) >= ICE_THRESHOLD


abs_extent = pd.Series({d: covered(ic)[abs_mask].sum() * CELL_AREA_KM2 / 1e6 for d, ic in SIC.items()})
abs_extent = abs_extent.reindex(sic_days)
abs_change = (abs_extent - abs_extent.shift(1)) * 1e3          # thousand km2 per day, ending on day t

strip_extent = pd.DataFrame(
    {d: np.histogram(SLONX[covered(SIC[d])], bins=x_edges)[0] * CELL_AREA_KM2 / 1e3
     if d in SIC else np.full(x_mid.size, np.nan) for d in sic_days}).T   # thousand km2 per strip
strip_change = strip_extent.diff()                                      # per day


# ---------------------------------------------------------------
# 2. ERA5: v10 and MSLP (region south of 45 S)
# ---------------------------------------------------------------
def read_era5(path, prefer):
    ds = xr.open_dataset(path)
    var = prefer if prefer in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var]
    tn = "valid_time" if "valid_time" in da.dims else "time"
    la = "latitude" if "latitude" in da.dims else "lat"
    lo = "longitude" if "longitude" in da.dims else "lon"
    for d in [d for d in da.dims if d not in (tn, la, lo)]:
        da = da.isel({d: 0})
    da = da.sortby(la).sel({la: slice(-85, -45)})
    return (da.values.astype("float64"), da[la].values, da[lo].values,
            pd.to_datetime(da[tn].values).normalize())


V10, lat, lon, days = read_era5(V10_FILE, "v10")
P, lat_p, lon_p, days_p = read_era5(MSLP_FILE, "msl")
if not (np.allclose(lat, lat_p) and np.allclose(lon, lon_p) and days.equals(days_p)):
    raise ValueError("v10 and MSLP files differ in grid or days")
if np.nanmean(P[0]) > 2000:
    P = P / 100.0                                                  # Pa -> hPa
N10 = -V10                                                         # northerly wind
U10 = None
if os.path.exists(U10_FILE):
    U10, lat_u, lon_u, days_u = read_era5(U10_FILE, "u10")
    if not (np.allclose(lat, lat_u) and np.allclose(lon, lon_u) and days.equals(days_u)):
        raise ValueError("u10 file differs from v10 in grid or days")
    print("Arrows on map a: ERA5 10 m wind (u10, v10)")
else:
    print(f"u10 file not found ({U10_FILE}); arrows on map a: geostrophic wind from MSLP")

# reorder longitudes so the Pacific is continuous, keep the plotted region only
lonx = shift_lon(lon)
order = np.argsort(lonx)
lonx, N10, P = lonx[order], N10[:, :, order], P[:, :, order]
if U10 is not None:
    U10 = U10[:, :, order]
keep = (lonx >= HOV_LON[0] - 15) & (lonx <= HOV_LON[1] + 20)
lonx, N10, P = lonx[keep], N10[:, :, keep], P[:, :, keep]
if U10 is not None:
    U10 = U10[:, :, keep]
LAT2, LONX2 = np.meshgrid(lat, lonx, indexing="ij")


# ---------------------------------------------------------------
# 3. LAND MASK on the ERA5 grid (from a polar Basemap, independent of the regional map)
# ---------------------------------------------------------------
def land_at(m, x, y):
    p = np.column_stack([np.ravel(x), np.ravel(y)])
    out = np.zeros(p.shape[0], bool)
    for poly, ptype in zip(m.coastpolygons, m.coastpolygontypes):
        if ptype == 1:
            out |= Path(np.column_stack(poly)).contains_points(p)
    return out.reshape(np.shape(x))


m_polar = Basemap(projection="spstere", boundinglat=-44, lon_0=180, resolution="l")
OCEAN = ~land_at(m_polar, *m_polar(LONX2, LAT2))


# ---------------------------------------------------------------
# 4. WIND INDEX, HOVMOLLER, EPISODES, DIAGNOSTICS
# ---------------------------------------------------------------
w_lat = np.cos(np.radians(LAT2))
in_box = ((LAT2 >= BOX_LAT[0]) & (LAT2 <= BOX_LAT[1]) &
          (LONX2 >= BOX_LON[0]) & (LONX2 <= BOX_LON[1]) & OCEAN)
wind_index = pd.Series([np.sum(N10[k][in_box] * w_lat[in_box]) / np.sum(w_lat[in_box])
                        for k in range(len(days))], index=days)

in_band = (LAT2 >= BAND_LAT[0]) & (LAT2 <= BAND_LAT[1]) & OCEAN
band_w = np.where(in_band, w_lat, 0.0)
band_wind = (np.nan_to_num(N10) * band_w).sum(axis=1) / np.maximum(band_w.sum(axis=0), 1e-9)  # day x lon
band_wind[:, band_w.sum(axis=0) == 0] = np.nan
bin_id = np.digitize(lonx, x_edges) - 1
hov_wind = np.full((len(days), x_mid.size), np.nan)
for b in range(x_mid.size):
    cols = bin_id == b
    if cols.any():
        hov_wind[:, b] = np.nanmean(band_wind[:, cols], axis=1)

# northerly episodes (consecutive days with index >= threshold)
on = (wind_index >= EPISODE_MIN_WIND).values
episodes, start = [], None
for k, flag in enumerate(np.append(on, False)):
    if flag and start is None:
        start = k
    elif not flag and start is not None:
        if k - start >= EPISODE_MIN_DAYS:
            episodes.append((days[start], days[k - 1]))
        start = None
dip_day = pd.Timestamp(YEAR, MONTH, DIP[0] + 2)

print("\nNortherly episodes (index >= %.1f m/s):" % EPISODE_MIN_WIND)
ep_info = []
for n, (s, e) in enumerate(episodes, 1):
    before = abs_extent.get(s - pd.Timedelta(days=1), np.nan)
    low = abs_extent[s:e + pd.Timedelta(days=1)].min()
    is_dip = s <= dip_day <= e
    ep_info.append((n, s, e, low - before, is_dip))
    print(f"  {n}: {s:%d %b} - {e:%d %b}, peak {wind_index[s:e].max():.1f} m/s, "
          f"ABS extent {before:.3f} -> lowest {low:.3f} ({low - before:+.3f} million km2)"
          + ("   <- the dip" if is_dip else ""))

# where along the coast was ice lost during the dip?
d0, d1 = pd.Timestamp(YEAR, MONTH, DIP[0]), pd.Timestamp(YEAR, MONTH, DIP[1])
lost_by_strip = strip_extent.loc[d1] - strip_extent.loc[d0]
dip_sel = (days >= d0) & (days <= d1)
print(f"\nDip {DIP[0]}-{DIP[1]} Aug by 10-degree longitude (use this to check the index box):")
print(f"{'longitude':>14s} {'ice change (10^3 km2)':>22s} {'mean northerly wind (m/s)':>27s}")
def lon_name(x):
    lo = x + 360 if x < -180 else x
    if abs(abs(lo) - 180) < 1e-6:
        return "180°"
    return f"{abs(lo):.0f}°{'W' if lo < 0 else 'E'}"


for lo0 in np.arange(HOV_LON[0], HOV_LON[1], 10):
    sb = (x_mid >= lo0) & (x_mid < lo0 + 10)
    lab = f"{lon_name(lo0)}–{lon_name(lo0 + 10)}"
    print(f"{lab:>14s} {lost_by_strip[sb].sum():22.1f} {np.nanmean(hov_wind[dip_sel][:, sb]):27.1f}")

pair = pd.concat([wind_index, abs_change], axis=1, keys=["wind", "dext"], sort=True).dropna()
r = pair["wind"].corr(pair["dext"])
print(f"\nWind index vs same-day ABS extent change: r = {r:+.2f} (n = {len(pair)})")


# ---------------------------------------------------------------
# 5. COLOURS AND SMALL HELPERS
# ---------------------------------------------------------------
def white_centre_cmap(name, levels, lo=0.05, hi=0.95):
    base = plt.get_cmap(name)
    cols = [base(v) for v in np.linspace(lo, hi, len(levels) - 1)]
    cols[len(cols) // 2] = (1, 1, 1, 1)
    cm = ListedColormap(cols)
    cm.set_under(base(0.0))
    cm.set_over(base(1.0))
    return cm, BoundaryNorm(levels, cm.N)


WIND_CMAP, WIND_NORM = white_centre_cmap("RdBu_r", WIND_LEVELS)
ICE_CMAP, ICE_NORM = white_centre_cmap("PuOr", ICE_LEVELS, 0.04, 0.96)   # orange = loss, purple = gain
HALO = [pe.withStroke(linewidth=3.5, foreground="white")]


def lon_label(x, _=None):
    lo = x + 360 if x < -180 else x
    if abs(lo) < 1e-6 or abs(abs(lo) - 180) < 1e-6:
        return "180°" if abs(lo) > 1 else "0°"
    return f"{abs(lo):.0f}°{'W' if lo < 0 else 'E'}"


def label_in_box(ax, m, la, lo, value, sym, col):
    """L/H label kept inside the rectangular map; a centre outside is marked at the frame with an arrow."""
    px, py = m(lo, la)
    fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((px, py)))
    gx, gy = np.clip(fx, 0.08, 0.92), np.clip(fy, 0.14, 0.84)
    if (gx, gy) != (fx, fy):
        dx, dy = fx - gx, fy - gy
        ux, uy = dx / np.hypot(dx, dy), dy / np.hypot(dx, dy)
        # arrow starts beyond the letter and points towards the centre outside the frame
        ax.annotate("", xy=(gx + 0.105 * ux, gy + 0.105 * uy), xytext=(gx + 0.045 * ux, gy + 0.045 * uy),
                    xycoords="axes fraction", zorder=11,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=16))
    ax.annotate(sym, xy=(gx, gy), xycoords="axes fraction", ha="center", va="center",
                fontsize=28, fontweight="bold", color=col, zorder=11, path_effects=HALO)
    ax.annotate(f"{value:.0f}", xy=(gx, gy), xycoords="axes fraction", xytext=(0, -24),
                textcoords="offset points", ha="center", va="center", fontsize=17,
                fontweight="bold", color=col, zorder=11, path_effects=HALO)


# ---------------------------------------------------------------
# 6. FIGURE LAYOUT
# ---------------------------------------------------------------
fig = plt.figure(figsize=(20, 17.5))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.60], left=0.05, right=0.97,
                         top=0.91, bottom=0.07, hspace=0.40)
top = outer[0].subgridspec(1, 3, width_ratios=[1.35, 0.72, 0.72], wspace=0.27)
bot = outer[1].subgridspec(1, 2, width_ratios=[0.72, 0.28], wspace=0.45)
ax_map = fig.add_subplot(top[0])
ax_hw = fig.add_subplot(top[1])
ax_hi = fig.add_subplot(top[2], sharey=ax_hw)
ax_ts = fig.add_subplot(bot[0])
ax_sc = fig.add_subplot(bot[1])

# ---------------- a) regional map ----------------
plt.sca(ax_map)
m = Basemap(projection="stere", lat_0=MAP_CENTRE[0], lon_0=MAP_CENTRE[1], lat_ts=MAP_CENTRE[0],
            width=MAP_SIZE_M[0], height=MAP_SIZE_M[1], resolution="l", ax=ax_map)
X, Y = m(LONX2, LAT2)
wind_dip = N10[dip_sel].mean(axis=0)
p_dip = P[dip_sel].mean(axis=0)
cs_w = m.contourf(X, Y, np.ma.masked_where(~OCEAN, wind_dip), levels=WIND_LEVELS,
                  cmap=WIND_CMAP, norm=WIND_NORM, extend="both", zorder=1)
m.contour(X, Y, np.ma.masked_where(~OCEAN, p_dip), levels=ISOBARS, colors="#4D4D4D",
          linewidths=0.8, zorder=2)

# ---- wind arrows (7-11 Aug mean): 10 m wind if u10 is available, otherwise geostrophic from MSLP
if U10 is not None:
    u_vec, v_vec = U10[dip_sel].mean(axis=0), -wind_dip
    VEC_LABEL = "10 m wind (arrows)"
else:
    from scipy.ndimage import gaussian_filter
    dlat_, dlon_ = abs(lat[1] - lat[0]), abs(lonx[1] - lonx[0])
    p_pa = gaussian_filter(p_dip * 100.0, sigma=(1.0 / dlat_, 1.0 / dlon_), mode="nearest")
    f_cor = 2 * OMEGA * np.sin(np.radians(lat))[:, None]
    dpdy = np.gradient(p_pa, np.radians(lat), axis=0) / R_EARTH
    dpdx = np.gradient(p_pa, np.radians(lonx), axis=1) / (R_EARTH * np.cos(np.radians(lat))[:, None])
    u_vec, v_vec = -dpdy / (RHO * f_cor), dpdx / (RHO * f_cor)
    VEC_LABEL = "Geostrophic wind (arrows)"
dlat_, dlon_ = abs(lat[1] - lat[0]), abs(lonx[1] - lonx[0])
pts_i, pts_j = [], []
for i in range(0, len(lat), max(1, int(round(VEC_STEP_DEG / dlat_)))):
    if lat[i] < -80:
        continue
    step_j = max(1, int(round(VEC_STEP_DEG / (dlon_ * np.cos(np.radians(lat[i]))))))
    for j in range(0, len(lonx), step_j):
        pts_i.append(i)
        pts_j.append(j)
pts_i, pts_j = np.array(pts_i), np.array(pts_j)
uu, vv = u_vec[pts_i, pts_j], v_vec[pts_i, pts_j]
ok = OCEAN[pts_i, pts_j] & np.isfinite(uu) & np.isfinite(vv) & (np.hypot(uu, vv) >= VEC_MIN)
ur, vr = m.rotate_vector(uu[ok], vv[ok], lonx[pts_j][ok], lat[pts_i][ok])
xq, yq = m(lonx[pts_j][ok], lat[pts_i][ok])
q_vec = m.quiver(xq, yq, ur, vr, scale=VEC_SCALE, width=0.0024, headwidth=4.2, headlength=4.5,
                 color="#1A1A1A", zorder=3.5)
# reference arrow in a small white box, lower right of the map
ax_map.add_patch(FancyBboxPatch((0.80, 0.025), 0.175, 0.10, transform=ax_map.transAxes,
                                boxstyle="round,pad=0.006", fc="white", ec="#888888", lw=0.8, zorder=12))
qk = ax_map.quiverkey(q_vec, 0.8875, 0.055, VEC_KEY, f"{VEC_KEY} m/s", labelpos="N", coordinates="axes",
                      fontproperties={"size": FS_TICK - 1, "weight": "bold"}, labelsep=0.06)
qk.set_zorder(13)

sx, sy = m(SLON, SLAT)
ic0, ic1 = SIC[d0], SIC[d1]
lost = (covered(ic0) & ~covered(ic1)).astype(float)
m.contourf(sx, sy, lost, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=3)
m.contour(sx, sy, np.ma.masked_invalid(ic0), levels=[ICE_THRESHOLD], zorder=6, **EDGE_START)
m.contour(sx, sy, np.ma.masked_invalid(ic1), levels=[ICE_THRESHOLD], zorder=7, **EDGE_END)

m.fillcontinents(color=LAND_COLOUR, lake_color=LAND_COLOUR, zorder=4)
m.drawcoastlines(linewidth=0.9, zorder=5)
# lat/lon labels only outside the map frame: latitudes on the left, longitudes along the top
# (where the meridians are far apart, so the labels don't crowd)
m.drawparallels(np.arange(-80, -44, 10), labels=[1, 0, 0, 0], fontsize=FS_TICK, linewidth=0.4,
                zorder=3.9)
m.drawmeridians(np.arange(-180, 181, 20), labels=[0, 0, 1, 0], fontsize=FS_TICK, linewidth=0.4,
                zorder=3.9)


def inside(ax, px, py, margin=0.04):
    fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((px, py)))
    return margin < fx < 1 - margin and margin < fy < 1 - margin


# sector lines drawn below the land fill (zorder 4), so they appear only over the ocean
for L in (-130, -60, 160):
    ll = np.linspace(-90, -45, 200)
    xx, yy = m(np.full_like(ll, L), ll)
    m.plot(xx, yy, color="black", linewidth=1.1, zorder=3.8)
bl = np.concatenate([np.full(50, BOX_LAT[0]), np.linspace(*BOX_LAT, 50),
                     np.full(50, BOX_LAT[1]), np.linspace(*BOX_LAT[::-1], 50)])
bo = np.concatenate([np.linspace(*BOX_LON, 50), np.full(50, BOX_LON[1]),
                     np.linspace(*BOX_LON[::-1], 50), np.full(50, BOX_LON[0])])
m.plot(*m(bo, bl), zorder=9, **BOX_STYLE)

for box, sym, col, fmin in ((dict(lat=(-75, -50), lon=(-185, -115)), "L", "#B2182B", True),
                            (dict(lat=(-68, -45), lon=(-120, -78)), "H", "#2166AC", False)):
    sel = ((LAT2 >= box["lat"][0]) & (LAT2 <= box["lat"][1]) &
           (LONX2 >= box["lon"][0]) & (LONX2 <= box["lon"][1]) & OCEAN)
    vals = np.where(sel, p_dip, np.inf if fmin else -np.inf)
    i, j = np.unravel_index(np.argmin(vals) if fmin else np.argmax(vals), p_dip.shape)
    label_in_box(ax_map, m, lat[i], lonx[j], p_dip[i, j], sym, col)

# sector names in open water next to the coast, only where they fall inside the frame
for name, lo_, la_ in (("RSS", -150, -71), ("ABS", -97, -68)):
    xx, yy = m(lo_, la_)
    if inside(ax_map, xx, yy, 0.06):
        ax_map.text(xx, yy, name, fontsize=FS_LABEL, fontweight="bold", ha="center", va="center",
                    zorder=10, path_effects=HALO)

# ---------------- b) and c) Hovmollers ----------------
y_edges = mdates.date2num(pd.date_range(days[0] - pd.Timedelta(hours=12), periods=len(days) + 1,
                                        freq="D").to_pydatetime())
def lon_smooth(a, n):
    """Running mean over n longitude strips (NaN-aware); time is not smoothed."""
    if n <= 1:
        return a
    k = np.ones(n)
    num = np.apply_along_axis(lambda r: np.convolve(np.nan_to_num(r), k, "same"), 1, a)
    den = np.apply_along_axis(lambda r: np.convolve(np.isfinite(r).astype(float), k, "same"), 1, a)
    out = num / np.where(den > 0, den, np.nan)
    out[~np.isfinite(a)] = np.nan
    return out


hov_wind_s = lon_smooth(hov_wind, HOV_SMOOTH_BINS)
ice_rows = lon_smooth(strip_change.reindex(days).values, HOV_SMOOTH_BINS)   # change ending on day t
pm_w = ax_hw.pcolormesh(x_edges, y_edges, hov_wind_s, cmap=WIND_CMAP, norm=WIND_NORM, shading="flat")
pm_i = ax_hi.pcolormesh(x_edges, y_edges, ice_rows, cmap=ICE_CMAP, norm=ICE_NORM, shading="flat")
# the northerly wind bands of panel b, outlined on panel c
y_mid = 0.5 * (y_edges[1:] + y_edges[:-1])
ax_hi.contour(x_mid, y_mid, np.nan_to_num(hov_wind_s), levels=[WIND_CONTOUR], colors="#7F0000",
              linewidths=1.8)

for ax in (ax_hw, ax_hi):
    ax.set_xlim(*HOV_LON)
    ax.xaxis.set_major_locator(plt.FixedLocator([-200, -160, -120, -80, -40]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lon_label))
    ax.yaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(labelsize=FS_TICK)
    for L in (-130, -60):
        ax.axvline(L, color="black", lw=1.2)
    for dd in DIP:
        ax.axhline(mdates.date2num(pd.Timestamp(YEAR, MONTH, dd)), color="black", lw=1.0, ls=":")
    for name, (a, b) in SECTORS.items():
        a_, b_ = shift_lon(a), shift_lon(b)
        lo_, hi_ = max(min(a_, b_), HOV_LON[0]), min(max(a_, b_), HOV_LON[1])
        if hi_ > lo_:
            ax.text((lo_ + hi_) / 2, 1.01, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=FS_TICK, fontweight="bold")
ax_hw.invert_yaxis()            # time runs downward (once: the y axis is shared with panel c)
plt.setp(ax_hi.get_yticklabels(), visible=False)
ax_hw.annotate("", xy=(BOX_LON[0], 1.0), xytext=(BOX_LON[1], 1.0), xycoords=("data", "axes fraction"),
               arrowprops=dict(arrowstyle="-", lw=4, color=BOX_STYLE["color"]))

# ---------------- d) time series ----------------
x = wind_index.index
EP_TEXTS = []
ax_ts.axhline(0, color="#555555", lw=1)
ax_b = ax_ts.twinx()
for n, s, e, dchg, is_dip in ep_info:
    # shading on the bar axis, behind the bars, so the bars keep their colour
    ax_b.axvspan(s - pd.Timedelta(hours=12), e + pd.Timedelta(hours=12),
                 color=COL_EPISODE, alpha=0.60 if is_dip else 0.38, lw=0, zorder=0)
    # episode label inside the panel, at the top of its shaded band
    # short label (the unit is given once, in the legend) so it fits inside its own band
    EP_TEXTS.append(ax_ts.text(s + (e - s) / 2, 0.965, f"Ep. {n}" + (" (dip)" if is_dip else "") +
                               f"\n{dchg * 1e3:+.0f}".replace("-", "−"),
                               transform=ax_ts.get_xaxis_transform(),
                               ha="center", va="top", fontsize=FS_TICK - 1, fontweight="bold",
                               color=COL_DIP if is_dip else "#444444", zorder=8,
                               path_effects=[pe.withStroke(linewidth=3, foreground="white")]))
chg = abs_change.reindex(x)
ax_b.bar(x, chg.values, width=0.7, color=[COL_LOSS if v < 0 else COL_GAIN for v in chg.fillna(0)],
         alpha=0.85, zorder=2)
ax_ts.plot(x, wind_index.values, color=COL_WIND, lw=3, marker="o", ms=5, zorder=5)
ax_ts.set_zorder(ax_b.get_zorder() + 1)
ax_ts.patch.set_visible(False)

# align the two zero lines
wmin = min(wind_index.min(), -0.5) * 1.15
# headroom at the top, so the episode labels inside the panel sit above the wind line
# (the highest wind value is placed at 75% of the panel height)
wmax = wmin + (wind_index.max() - wmin) / 0.75
bmin, bmax = np.nanmin(chg.values), np.nanmax(chg.values)
s_ = max(bmax / wmax, bmin / wmin) * 1.05
ax_ts.set_ylim(wmin, wmax)
ax_b.set_ylim(wmin * s_, wmax * s_)

ax_ts.set_ylabel("Northerly wind,\nABS index (m/s)", fontsize=FS_LABEL, color=COL_WIND)
ax_b.set_ylabel("ABS extent change\n(10³ km² per day)", fontsize=FS_LABEL)
ax_ts.tick_params(axis="y", labelcolor=COL_WIND, labelsize=FS_TICK)
ax_ts.yaxis.set_major_locator(plt.MultipleLocator(2))       # whole numbers: -2, 0, 2, 4, 6
ax_b.tick_params(axis="y", labelsize=FS_TICK)
ax_ts.tick_params(axis="x", labelsize=FS_TICK)
ax_ts.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]))
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(ax_ts.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")   # dates no longer touch
ax_ts.set_xlim(x.min() - pd.Timedelta(hours=18), x.max() + pd.Timedelta(hours=18))
ax_ts.grid(axis="y", linestyle=":", alpha=0.6)
ax_ts.legend(handles=[Line2D([], [], color=COL_WIND, lw=3, marker="o",
                             label="Northerly 10 m wind"),
                      Patch(color=COL_LOSS, label="ABS ice extent loss"),
                      Patch(color=COL_GAIN, label="ABS ice extent gain"),
                      Patch(color=COL_EPISODE, label=f"Northerly episode (≥ {EPISODE_MIN_WIND:g} m/s);\n"
                                                     "number = net ABS ice change (10³ km²)")],
             loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, fontsize=FS_LEGEND, frameon=False,
             columnspacing=3.0, handlelength=1.8, handletextpad=0.5)   # 2 x 2: stays within (d)
# (title of d is placed after the final layout, level with the title of e)

# ---------------- e) scatter ----------------
dip_days = [n for n in pair.index if any(s <= n <= e and d for _, s, e, _, d in ep_info)]
ep_days = [n for n in pair.index if any(s <= n <= e for _, s, e, _, _ in ep_info)]
other = pair.drop(index=dip_days)                                  # every day outside the dip episode
ax_sc.axhline(0, color="#888888", lw=0.8)
ax_sc.axvline(0, color="#888888", lw=0.8)
ax_sc.scatter(other["wind"], other["dext"], s=45, color="#9E9E9E", edgecolor="white", zorder=3,
              label="Other days")
dip_ep = [(s, e) for _, s, e, _, d in ep_info if d][0]
ax_sc.scatter(pair.loc[dip_days, "wind"], pair.loc[dip_days, "dext"], s=80, color=COL_DIP,
              edgecolor="black", zorder=5,
              label=f"Dip episode ({dip_ep[0].day}–{dip_ep[1].day} Aug)")
x_hi = pair["wind"].max() - 0.12 * (pair["wind"].max() - pair["wind"].min())
for d in dip_days:                                                 # day numbers on the dip points
    right_edge = pair.loc[d, "wind"] > x_hi                        # near the right edge: label on the left
    ax_sc.annotate(f"{d.day}", xy=(pair.loc[d, "wind"], pair.loc[d, "dext"]),
                   xytext=(-8, 5) if right_edge else (7, 4), ha="right" if right_edge else "left",
                   textcoords="offset points", fontsize=FS_TICK - 2, fontweight="bold",
                   color=COL_DIP, zorder=7, path_effects=HALO)
ax_sc.margins(x=0.06, y=0.08)                                      # keep points and labels off the frame
k_, c_ = np.polyfit(pair["wind"], pair["dext"], 1)
xx = np.linspace(pair["wind"].min(), pair["wind"].max(), 50)
ax_sc.plot(xx, k_ * xx + c_, color="black", lw=1.6, zorder=6)
ax_sc.text(0.97, 0.96, f"r = {r:+.2f}\nn = {len(pair)}", transform=ax_sc.transAxes, ha="right",
           va="top", fontsize=FS_LABEL, fontweight="bold")
ax_sc.set_xlabel("Northerly 10 m wind (m/s)", fontsize=FS_LABEL)
ax_sc.set_ylabel("Daily ABS sea ice extent\nchange (10³ km² per day)", fontsize=FS_LABEL)
ax_sc.tick_params(labelsize=FS_TICK)
ax_sc.grid(linestyle=":", alpha=0.6)
ax_sc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1, fontsize=FS_LEGEND,
             frameon=False, handletextpad=0.3)                       # stacked: e is narrow
# (title of e is placed after the final layout)

# ---------------- colour bars and legend for the top row ----------------
fig.canvas.draw()
inv = fig.transFigure.inverted()
pw, pi_, pm_ = ax_hw.get_position(), ax_hi.get_position(), ax_map.get_position()
# colour bars just below the lowest tick labels of a/b/c (measured, so bigger fonts never overlap)
tick_bottom = min(inv.transform(t.get_window_extent())[0, 1]
                  for a in (ax_hw, ax_hi) for t in a.get_xticklabels() if t.get_text())
cb_y = min(tick_bottom, pm_.y0) - 0.022
CB_H = 0.026                                           # colour-bar thickness (figure fraction)
cax_w = fig.add_axes([pm_.x0, cb_y - CB_H + 0.012, pw.x1 - pm_.x0, CB_H])   # one wind colour bar under a and b
cb = fig.colorbar(pm_w, cax=cax_w, orientation="horizontal", extend="both", extendfrac=0.025)
cb.set_ticks(WIND_LEVELS)
cb.ax.tick_params(labelsize=FS_CBTICK, length=6, width=1.2)
cb.set_label("Northerly 10 m wind (m/s)", fontsize=FS_LABEL)   # = −v10: positive = blowing towards the pole
cax_i = fig.add_axes([pi_.x0, cb_y - CB_H + 0.012, pi_.width, CB_H])
cb2 = fig.colorbar(pm_i, cax=cax_i, orientation="horizontal", extend="both", extendfrac=0.06)
cb2.set_ticks([-6, -3, -1, 1, 3, 6])
cb2.ax.tick_params(labelsize=FS_CBTICK, length=6, width=1.2)
cb2.set_label(f"Ice extent change\n(10³ km² per day per {HOV_BIN:g}°)", fontsize=FS_LABEL)

fig.canvas.draw()
cb_label_bottom = inv.transform(cb.ax.xaxis.label.get_window_extent())[0, 1]
cb2_label_bottom = inv.transform(cb2.ax.xaxis.label.get_window_extent())[0, 1]
# note under the ice colour bar, placed below its label (measured, never overlapping)
fig.text(pi_.x0 + pi_.width / 2, cb2_label_bottom - 0.006,
         f"Dark red line: northerly wind ≥ {WIND_CONTOUR:g} m/s (from b)",
         ha="center", va="top", fontsize=FS_LEGEND - 1, color="#7F0000")

# map legend under the wind colour bar, spread across the full width of (a) + (b)
leg_map = fig.legend(handles=[Line2D([], [], color=EDGE_START["colors"], lw=2.6, label=f"Ice edge, {DIP[0]} Aug"),
                       Line2D([], [], color=EDGE_END["colors"], lw=2.2, label=f"Ice edge, {DIP[1]} Aug"),
                       Patch(facecolor="white", edgecolor="black", hatch="////",
                             label=f"Ice lost {DIP[0]}–{DIP[1]} Aug"),
                       Line2D([], [], color="#4D4D4D", lw=1.2, label="MSLP (8 hPa interval)"),
                       Line2D([], [], label="Wind index box", **BOX_STYLE),
                       Line2D([], [], color="#1A1A1A", marker=r"$\longrightarrow$", markersize=30,
                              linestyle="none", label=VEC_LABEL)],
           loc="upper left", mode="expand", ncol=3, frameon=False, fontsize=FS_LEGEND,
           bbox_to_anchor=(pm_.x0, cb_label_bottom - 0.008, pw.x1 - pm_.x0, 0.0),
           bbox_transform=fig.transFigure, handlelength=2.6, handleheight=1.2,
           labelspacing=0.7, borderaxespad=0)

# ---------------- final layout: measured, so nothing overlaps whatever the map shape ----------------
fig.canvas.draw()
ext_fig = lambda artist: inv.transform(artist.get_window_extent())          # [[x0, y0], [x1, y1]]

# 1) bottom row moved down if it would touch the colour-bar labels / map legend
row_bottom = min(ext_fig(cb.ax.xaxis.label)[0, 1], ext_fig(cb2.ax.xaxis.label)[0, 1],
                 ext_fig(leg_map)[0, 1], ext_fig(fig.texts[-1])[0, 1])
# top of the bottom row = top of the episode labels of d
row2_top = max([ax_ts.get_position().y1] + [ext_fig(t)[1, 1] for t in ax_ts.texts])
shift = row2_top - (row_bottom - 0.025)                 # small gap; there are no panel titles now
for ax in (ax_ts, ax_sc):            # moves the row up or down to close the gap (ax_b follows ax_ts as its twin)
    p = ax.get_position()
    ax.set_position([p.x0, p.y0 - shift, p.width, p.height])
pts, psc = ax_ts.get_position(), ax_sc.get_position()   # e: same height as d
ax_sc.set_position([psc.x0, pts.y0, psc.width, pts.height])

# 1b) line (d) up with (a): the left edge of d's y-axis label goes where a's latitude labels
#     start; d keeps its right edge, so it simply becomes a little narrower
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
a_left = inv.transform((ax_map.get_tightbbox(renderer).x0, 0))[0]
d_left = inv.transform((ax_ts.yaxis.get_tightbbox(renderer).x0, 0))[0]
delta = a_left - d_left
if delta > 0:
    for ax in (ax_ts, ax_b):                      # the twin axis must move with it
        p = ax.get_position()
        ax.set_position([p.x0 + delta, p.y0, p.width - delta, p.height])

# 2) panel letters inside each box, top-left (no titles: the caption describes the panels)
LETTER_HALO = [pe.withStroke(linewidth=5, foreground="white")]
for ax, letter in ((ax_map, "a"), (ax_hw, "b"), (ax_hi, "c"), (ax_ts, "d"), (ax_sc, "e")):
    t_letter = ax.text(0.02, 0.975, f"({letter})", transform=ax.transAxes, ha="left", va="top",
                       fontsize=FS_PANEL, fontweight="bold", zorder=20, path_effects=LETTER_HALO)
    if ax is ax_ts:
        d_letter = t_letter

# episode labels in (d): if one would run into the "(d)" letter, slide it to the right
fig.canvas.draw()
letter_right = d_letter.get_window_extent().x1
for t in EP_TEXTS:
    bb = t.get_window_extent()
    overlap = letter_right + 8 - bb.x0            # pixels, with a small gap
    if overlap > 0:
        x_disp = ax_ts.transData.transform((mdates.date2num(t.get_position()[0]), 0))[0] + overlap
        t.set_x(float(ax_ts.transData.inverted().transform((x_disp, 0))[0]))   # date number

if SHOW_TITLE:
    fig.canvas.draw()
    y_titles = max([ax_map.get_position().y1, ax_hw.get_position().y1] +
                   [ext_fig(t)[1, 1] for t in ax_hw.texts]) + 0.008
    fig.text(0.5, y_titles + 0.035,
             "Northerly wind episodes and Amundsen–Bellingshausen sea ice loss, August 2026",
             ha="center", va="bottom", fontsize=FS_TITLE, fontweight="bold")

plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT_FIG}")
plt.show()
