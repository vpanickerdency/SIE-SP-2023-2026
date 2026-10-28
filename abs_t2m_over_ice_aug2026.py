"""
Air temperature over the ABS sea ice, August 2026: cause or effect of the ice loss?
----------------------------------------------------------------------------------
ERA5 2 m temperature (t2m) is partly set by the sea ice underneath: where the ice goes, open
water keeps the air near -1.8 degC. So "warm air over the box" can be a RESULT of ice loss.
This figure only looks at air over ice that was present that day (OSI SAF concentration), and
asks whether the air over ice that disappeared was already warm the day BEFORE it disappeared.

a-c) Mean t2m before (3-6 Aug), during (7-11 Aug, Episode 2) and after (12-15 Aug) the dip,
     coloured so that blue = colder and red = warmer than the seawater freezing point (-1.8 degC),
     with the 15 % ice edge on the first (green) and last (black) day of each period;
     panel b also hatches the ice lost between 7 and 11 Aug.
d)   Daily t2m over ice in the ABS index box: edge ice (15-80 %) and pack ice (>= 80 %),
     and over open water for comparison.
e)   Share of the ice-covered area in the box where the air was warmer than -1.8 degC.
Supplementary figure (former panel f), the day-before test: t2m on day t-1 at ice cells that were gone by day t, compared with
     edge-ice cells (15-80 %) that were still there on day t (same part of the pack, so a fair comparison).

Console: daily table (also saved as CSV) and a summary for each episode.

Requirements: xarray netCDF4 numpy pandas scipy matplotlib basemap   (same as the mechanism figure)
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import cKDTree
from mpl_toolkits.basemap import Basemap

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
BASE = r"C:\Users\vpani\Downloads\Sept_SIE_SP"
T2M_FILE = os.path.join(BASE, "2026_Aug_T2M", "data_stream-oper_stepType-instant.nc")
SIC_FOLDER = os.path.join(BASE, "2026_Jul_Aug_SIC")
OUT_FIG = os.path.join(BASE, "ABS_t2m_over_ice_Aug2026.png")
# supplementary: the day-before test (former panel f); set MAKE_SUPPLEMENT = False to skip it
MAKE_SUPPLEMENT = True
OUT_FIG_S = os.path.join(BASE, "ABS_t2m_daybefore_test_Aug2026_supplement.png")
OUT_CSV = os.path.join(BASE, "ABS_t2m_over_ice_Aug2026_daily.csv")

YEAR, MONTH = 2026, 8
PERIODS = [("a", "3–6 Aug, before", 3, 6),
           ("b", "7–11 Aug, Episode 2 (the dip)", 7, 11),
           ("c", "12–15 Aug, after", 12, 15)]
DIP = (7, 11)                                        # panel b: ice lost between these days
EPISODES = [(1, 1, 5), (2, 8, 13), (3, 20, 23)]      # report Table 3 (number, first day, last day)

BOX_LAT = (-72.0, -60.0)                             # ABS index box
BOX_LON = (-130.0, -60.0)
FREEZE = -1.8                                        # degC, seawater freezing point
ICE_THRESHOLD = 15.0                                 # % (ice edge)
PACK_THRESHOLD = 80.0                                # % (edge ice 15-80, pack ice >= 80)
MAX_MATCH_KM = 30.0                                  # ERA5 point -> nearest OSI SAF cell
MIN_LOST_CELLS = 15                                  # panel f: only days with at least this many lost cells

MAP_CENTRE = (-66.0, -105.0)                         # lat, lon (oblique stereographic, as the mechanism figure)
MAP_SIZE_M = (4.6e6, 3.3e6)                          # width, height
LABEL_MERIDIANS = True                               # longitude labels along the top edge of the maps

T_LEVELS = np.array([-24, -20, -16, -12, -9, -6, -4, FREEZE, 0, 2, 4, 6])
CB_FRAC = 0.66                                       # colour bar width as a fraction of the three maps' span

LAND_COLOUR = "#9E9E9E"
COL_EDGE_ICE, COL_PACK_ICE, COL_OPEN = "#2166AC", "#08306B", "#B2182B"
COL_LOST, COL_STAY = "#C0392B", "#2166AC"
COL_EPISODE, COL_DIP = "#F4C7C3", "#C0392B"
EDGE_START = dict(colors="#00C853", linewidths=2.0)
EDGE_END = dict(colors="black", linewidths=1.7)
BOX_STYLE = dict(color="#F5B700", linewidth=2.2, linestyle="--")
SHOW_TITLE = False                                   # title goes in the caption

FS_TITLE, FS_PANEL, FS_LABEL, FS_TICK, FS_LEGEND = 22, 20, 18, 16, 16
plt.rcParams.update({"hatch.color": "black", "hatch.linewidth": 0.8,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]})
HALO = [pe.withStroke(linewidth=3.0, foreground="white")]
# white box behind the panel letters (a)-(e)
LABEL_BOX = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#333333", linewidth=0.8, alpha=0.9)


def minus(v):
    """Number as text with a true minus sign (−), matching the tick labels."""
    return f"{v:g}".replace("-", "−")


FREEZE_TXT = minus(FREEZE)                           # "−1.8"


def panel_label(ax, letter, right=False):
    """Panel letter inside the top-left corner (top-right for the time series, so it
    does not collide with the Episode 1 label)."""
    x, ha = (0.985, "right") if right else (0.015, "left")
    ax.text(x, 0.975, f"({letter})", transform=ax.transAxes, ha=ha, va="top",
            fontsize=FS_PANEL, fontweight="bold", zorder=20, bbox=LABEL_BOX)


def shade_episodes(ax, fontsize, two_line=False):
    """Shade the episodes and write each label inside its own shaded box, at the top."""
    for n, a, b in EPISODES:
        s, e = pd.Timestamp(YEAR, MONTH, a), pd.Timestamp(YEAR, MONTH, b)
        ax.axvspan(s - pd.Timedelta(hours=12), e + pd.Timedelta(hours=12),
                   color=COL_EPISODE, alpha=0.65 if n == 2 else 0.4, lw=0, zorder=0)
        label = f"Episode\n{n}" if two_line else f"Episode {n}"
        ax.text(s + (e - s) / 2, 0.975, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=fontsize, fontweight="bold", linespacing=1.05,
                color=COL_DIP if n == 2 else "#444444", zorder=3)


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(pd.Timestamp(YEAR, MONTH, 1) - pd.Timedelta(hours=18),
                pd.Timestamp(YEAR, MONTH, 31) + pd.Timedelta(hours=18))
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(linestyle=":", alpha=0.6)


def shift_lon(lo):
    """Longitudes east of 145 E become negative (e.g. 160 E -> -200) so the Pacific is continuous."""
    lo = np.asarray(lo, dtype="float64")
    lo = np.where(lo > 180, lo - 360.0, lo)
    return np.where(lo > 145, lo - 360.0, lo)


def lon_label(x):
    lo = x + 360 if x < -180 else x
    if abs(abs(lo) - 180) < 1e-6:
        return "180°"
    return f"{abs(lo):.0f}°{'W' if lo < 0 else 'E'}"


# ---------------------------------------------------------------
# 1. SEA ICE: daily concentration 31 Jul - 31 Aug (same reading as the mechanism figure)
# ---------------------------------------------------------------
def read_sic(folder):
    days = pd.date_range(f"{YEAR}-{MONTH - 1:02d}-31", f"{YEAR}-{MONTH:02d}-31", freq="D")
    sic, slat, slon = {}, None, None
    for d in days:
        found = glob.glob(os.path.join(folder, "**", f"ice_conc_sh_*_{d:%Y%m%d}*.nc"), recursive=True)
        if not found:
            continue
        with xr.open_dataset(found[0]) as s:
            sic[d] = s["ice_conc"].isel(time=0).values.astype("float32")
            if slat is None:
                slat = s["lat"].values.astype("float64")
                slon = s["lon"].values.astype("float64")
    missing = [f"{d:%d %b}" for d in days if d not in sic]
    print(f"Sea ice: {len(sic)} daily files" + (f"; missing {missing}" if missing else ""))
    return sic, slat, slon


# ---------------------------------------------------------------
# 2. ERA5 t2m -> daily mean, degC, region south of 45 S, Pacific continuous
# ---------------------------------------------------------------
def read_t2m(path):
    ds = xr.open_dataset(path)
    var = next((v for v in ("t2m", "2t", "VAR_2T") if v in ds.data_vars), list(ds.data_vars)[0])
    da = ds[var]
    tn = "valid_time" if "valid_time" in da.dims else "time"
    la = "latitude" if "latitude" in da.dims else "lat"
    lo = "longitude" if "longitude" in da.dims else "lon"
    for d in [d for d in da.dims if d not in (tn, la, lo)]:
        da = da.isel({d: 0})
    da = da.sortby(la).sel({la: slice(-85, -45)})
    t = da.values.astype("float64")
    if np.nanmean(t) > 100:
        t = t - 273.15                                             # K -> degC
    stamps = pd.to_datetime(da[tn].values)
    days = pd.DatetimeIndex(sorted(set(stamps.normalize())))
    if len(days) < len(stamps):                                    # sub-daily -> daily mean
        t = np.stack([t[stamps.normalize() == d].mean(axis=0) for d in days])
        print(f"t2m: {len(stamps)} time steps averaged to {len(days)} daily means")
    lat, lon = da[la].values, da[lo].values
    lonx = shift_lon(lon)
    order = np.argsort(lonx)
    lonx, t = lonx[order], t[:, :, order]
    keep = (lonx >= -200) & (lonx <= -20)
    return t[:, :, keep], lat, lonx[keep], days


# ---------------------------------------------------------------
# 3. HELPERS (land mask as in the mechanism figure; nearest OSI SAF cell for each ERA5 point)
# ---------------------------------------------------------------
def land_at(m, x, y):
    p = np.column_stack([np.ravel(x), np.ravel(y)])
    out = np.zeros(p.shape[0], bool)
    for poly, ptype in zip(m.coastpolygons, m.coastpolygontypes):
        if ptype == 1:
            out |= Path(np.column_stack(poly)).contains_points(p)
    return out.reshape(np.shape(x))


def unit_xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def match_cells(slat, slon, lat2, lon2, max_km):
    """Index of the nearest OSI SAF cell for every ERA5 point (-1 where none within max_km)."""
    tree = cKDTree(unit_xyz(slat.ravel(), slon.ravel()))
    dist, idx = tree.query(unit_xyz(lat2.ravel(), lon2.ravel()))
    idx[dist * 6371.0 > max_km] = -1
    return idx.reshape(lat2.shape)


def sic_on_era5(ic, idx):
    flat = ic.ravel()
    out = np.where(idx >= 0, flat[np.clip(idx, 0, None)], np.nan)
    return out


def wmean(values, weights, mask):
    w = np.where(mask & np.isfinite(values), weights, 0.0)
    return np.nan if w.sum() == 0 else float((np.nan_to_num(values) * w).sum() / w.sum())


def freeze_cmap():
    """Blues below the freezing point, reds above it, so the colour break IS the freezing point."""
    n_cold = int(np.sum(T_LEVELS < FREEZE))
    n_warm = len(T_LEVELS) - 1 - n_cold
    cold = plt.get_cmap("Blues_r")(np.linspace(0.05, 0.85, n_cold))
    warm = plt.get_cmap("Reds")(np.linspace(0.25, 0.9, n_warm))
    cm = ListedColormap(np.vstack([cold, warm]))
    cm.set_under(plt.get_cmap("Blues_r")(0.0))
    cm.set_over(plt.get_cmap("Reds")(1.0))
    return cm, BoundaryNorm(T_LEVELS, cm.N)


def daily_table(T, days, lat2, lonx2, ocean, sic_e):
    """All daily statistics over the ABS index box."""
    w = np.cos(np.radians(lat2))
    box = ((lat2 >= BOX_LAT[0]) & (lat2 <= BOX_LAT[1]) &
           (lonx2 >= BOX_LON[0]) & (lonx2 <= BOX_LON[1]) & ocean)
    rows = []
    for k, d in enumerate(days):
        if d not in sic_e:
            continue
        s, t = sic_e[d], T[k]
        ice = box & (s >= ICE_THRESHOLD)
        edge = box & (s >= ICE_THRESHOLD) & (s < PACK_THRESHOLD)
        pack = box & (s >= PACK_THRESHOLD)
        open_ = box & np.isfinite(s) & (s < ICE_THRESHOLD)
        warm_ice = ice & (t > FREEZE)
        rec = dict(date=d,
                   t2m_ice=wmean(t, w, ice), t2m_edge_ice=wmean(t, w, edge),
                   t2m_pack_ice=wmean(t, w, pack), t2m_open_water=wmean(t, w, open_),
                   pct_ice_area_above_freeze=100 * np.sum(w[warm_ice]) / max(np.sum(w[ice]), 1e-9),
                   ice_cells=int(ice.sum()))
        # day-before test: air on day t-1 over ice that is gone on day t, and over ice that stays
        prev = d - pd.Timedelta(days=1)
        kp = days.get_loc(prev) if prev in days else None
        if kp is not None and prev in sic_e:
            was = box & (sic_e[prev] >= ICE_THRESHOLD)
            lost = was & np.isfinite(s) & (s < ICE_THRESHOLD)
            stay = was & (sic_e[prev] < PACK_THRESHOLD) & (s >= ICE_THRESHOLD)   # edge ice that stayed
            rec.update(lost_cells=int(lost.sum()),
                       t2m_daybefore_ice_lost=wmean(T[kp], w, lost),
                       t2m_daybefore_edge_ice_kept=wmean(T[kp], w, stay))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------
# 4. FIGURE
# ---------------------------------------------------------------
def make_figure(T, days, lat, lonx, ocean, sic, slat, slon, tab, out_fig):
    lat2, lonx2 = np.meshgrid(lat, lonx, indexing="ij")
    cmap, norm = freeze_cmap()

    fig = plt.figure(figsize=(20, 15))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.82], left=0.05, right=0.97,
                             top=0.90, bottom=0.08, hspace=0.42)
    top = outer[0].subgridspec(1, 3, wspace=0.06)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.22)
    ax_maps = [fig.add_subplot(top[i]) for i in range(3)]
    ax_d, ax_e = (fig.add_subplot(bot[i]) for i in range(2))

    # ---------------- a-c) maps ----------------
    box_lat = np.concatenate([np.full(50, BOX_LAT[0]), np.linspace(*BOX_LAT, 50),
                              np.full(50, BOX_LAT[1]), np.linspace(*BOX_LAT[::-1], 50)])
    box_lon = np.concatenate([np.linspace(*BOX_LON, 50), np.full(50, BOX_LON[1]),
                              np.linspace(*BOX_LON[::-1], 50), np.full(50, BOX_LON[0])])
    for ax, (letter, title, d0, d1) in zip(ax_maps, PERIODS):
        plt.sca(ax)
        m = Basemap(projection="stere", lat_0=MAP_CENTRE[0], lon_0=MAP_CENTRE[1], lat_ts=MAP_CENTRE[0],
                    width=MAP_SIZE_M[0], height=MAP_SIZE_M[1], resolution="l", ax=ax)
        X, Y = m(lonx2, lat2)
        sel = (days >= pd.Timestamp(YEAR, MONTH, d0)) & (days <= pd.Timestamp(YEAR, MONTH, d1))
        tm = T[sel].mean(axis=0)
        cs = m.contourf(X, Y, tm, levels=T_LEVELS, cmap=cmap, norm=norm, extend="both", zorder=1)
        m.contour(X, Y, tm, levels=[FREEZE], colors="#5A0000", linewidths=1.0,
                  linestyles="solid", zorder=2)

        sx, sy = m(slon, slat)
        e0, e1 = pd.Timestamp(YEAR, MONTH, d0), pd.Timestamp(YEAR, MONTH, d1)
        if letter == "b" and e0 in sic and e1 in sic:
            lost = ((np.nan_to_num(sic[e0]) >= ICE_THRESHOLD) &
                    (np.nan_to_num(sic[e1]) < ICE_THRESHOLD)).astype(float)
            m.contourf(sx, sy, lost, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=3)
        if e0 in sic:
            m.contour(sx, sy, np.ma.masked_invalid(sic[e0]), levels=[ICE_THRESHOLD], zorder=6, **EDGE_START)
        if e1 in sic:
            m.contour(sx, sy, np.ma.masked_invalid(sic[e1]), levels=[ICE_THRESHOLD], zorder=7, **EDGE_END)

        m.fillcontinents(color=LAND_COLOUR, lake_color=LAND_COLOUR, zorder=4)
        m.drawcoastlines(linewidth=0.8, zorder=5)
        m.drawparallels(np.arange(-80, -44, 10), labels=[1 if letter == "a" else 0, 0, 0, 0],
                        fontsize=FS_TICK, linewidth=0.7, color="#333333", dashes=[4, 3], zorder=5)
        m.drawmeridians(np.arange(-180, 181, 20), labels=[0, 0, int(LABEL_MERIDIANS), 0],
                        fmt=lon_label, fontsize=FS_TICK - 2, linewidth=0.7,
                        color="#333333", dashes=[4, 3], zorder=5)
        m.plot(*m(box_lon, box_lat), zorder=9, **BOX_STYLE)
        panel_label(ax, letter)

    # colour bar centred under the three maps, legend in one row under the colour bar
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    p0, p2 = ax_maps[0].get_position(), ax_maps[2].get_position()
    span = p2.x1 - p0.x0
    cb_w = CB_FRAC * span
    cb_x0 = p0.x0 + (span - cb_w) / 2
    cax = fig.add_axes([cb_x0, p0.y0 - 0.035, cb_w, 0.018])
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal", extend="both", extendfrac=0.03)
    cb.set_ticks(T_LEVELS)
    cb.set_ticklabels([minus(v) for v in T_LEVELS])
    cb.ax.tick_params(labelsize=FS_TICK, length=5)
    cb.outline.set_linewidth(0.8)
    cb.set_label("2 m air temperature (°C)", fontsize=FS_LABEL, labelpad=6)
    fig.canvas.draw()
    lab_bottom = inv.transform(cb.ax.xaxis.label.get_window_extent())[0, 1]
    fig.legend(handles=[Line2D([], [], color=EDGE_START["colors"], lw=2.0, label="Ice edge, first day"),
                        Line2D([], [], color=EDGE_END["colors"], lw=1.7, label="Ice edge, last day"),
                        Patch(facecolor="white", edgecolor="black", hatch="////", label=f"Ice lost {DIP[0]}–{DIP[1]} Aug"),
                        Line2D([], [], label="ABS index box", **BOX_STYLE),
                        Line2D([], [], color="#5A0000", lw=1.2, label=f"Freezing point ({FREEZE_TXT} °C)")],
               loc="upper center", bbox_to_anchor=(cb_x0 + cb_w / 2, lab_bottom - 0.006), ncol=5,
               columnspacing=1.6, handlelength=1.8,
               fontsize=FS_LEGEND, frameon=False)

    x = tab.index
    # d) temperature over ice
    ax_d.plot(x, tab.t2m_open_water, color=COL_OPEN, lw=2, ls="--", label=f"Open water (concentration < {ICE_THRESHOLD:g} %)")
    ax_d.plot(x, tab.t2m_edge_ice, color=COL_EDGE_ICE, lw=2.6,
              label=f"Edge ice ({ICE_THRESHOLD:g}–{PACK_THRESHOLD:g} %)")
    ax_d.plot(x, tab.t2m_pack_ice, color=COL_PACK_ICE, lw=2.6, label=f"Pack ice (≥ {PACK_THRESHOLD:g} %)")
    ax_d.axhline(FREEZE, color="black", lw=1.4)
    ax_d.text(0.01, FREEZE, f"{FREEZE_TXT} °C", transform=ax_d.get_yaxis_transform(), va="bottom",
              ha="left", fontsize=FS_TICK - 1, path_effects=HALO)
    shade_episodes(ax_d, FS_TICK - 1)
    date_axis(ax_d)
    ax_d.set_ylabel("2 m air temperature (°C)", fontsize=FS_LABEL)
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=FS_LEGEND - 1, frameon=False,
                handlelength=1.8, columnspacing=1.2)
    ax_d.set_ylim(top=np.nanmax(tab.t2m_open_water) + 4.0)          # room for the episode labels
    panel_label(ax_d, "d", right=True)

    # e) share of ice area under air warmer than freezing
    ax_e.bar(x, tab.pct_ice_area_above_freeze, width=0.75, color=COL_DIP, zorder=2)
    shade_episodes(ax_e, FS_TICK - 3, two_line=True)                 # boxes are too narrow for one line
    date_axis(ax_e)
    e_top = max(15, 1.32 * np.nanmax(tab.pct_ice_area_above_freeze))  # room for the episode labels
    ax_e.set_ylim(0, e_top)
    ax_e.set_yticks([v for v in range(0, 101, 20) if v <= e_top])
    ax_e.set_ylabel(f"% of ice-covered area with\n2 m air temperature above {FREEZE_TXT} °C", fontsize=FS_LABEL)
    panel_label(ax_e, "e", right=True)

    # measured layout: (e) has exactly the width of map (c), (d) starts under map (a) and
    # stops just short of the y-axis labels of (e); bottom row just under the legend
    GAP_DE = 0.012                                                   # space between (d) and the labels of (e)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pa, pb, pc = (a.get_position() for a in ax_maps)
    p = ax_e.get_position()
    ax_e.set_position([pc.x0, p.y0, pc.width, p.height])
    fig.canvas.draw()
    e_left = inv.transform(ax_e.get_tightbbox(renderer))[0, 0]       # left edge of the (e) y-axis label
    p = ax_d.get_position()
    ax_d.set_position([pa.x0, p.y0, e_left - GAP_DE - pa.x0, p.height])
    fig.canvas.draw()
    ext = lambda artist: inv.transform(artist.get_window_extent())
    row_bottom = min(ext(cb.ax.xaxis.label)[0, 1], ext(fig.legends[0])[0, 1])
    row2_top = max(inv.transform(a.get_tightbbox(renderer))[1, 1] for a in (ax_d, ax_e))
    shift = row2_top - (row_bottom - 0.035)
    for a in (ax_d, ax_e):
        p = a.get_position()
        a.set_position([p.x0, p.y0 - shift, p.width, p.height])
    y_title = max(a.get_position().y1 for a in ax_maps) + 0.05
    if SHOW_TITLE:
        fig.text(0.5, y_title, "Air temperature over the Amundsen–Bellingshausen sea ice, August 2026 "
                 "(ERA5 2 m temperature, OSI SAF sea ice)", ha="center", va="bottom",
                 fontsize=FS_TITLE, fontweight="bold")
    fig.savefig(out_fig, dpi=1000, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_fig}")
    return fig


def make_supplement(tab, out_fig):
    """Former panel f: air the day before over ice that vanished vs edge ice that survived.
    Caption note: the vanished (outermost) ice sits in warmer air on almost every day, inside and
    outside the northerly episodes, so this shows the edge-to-pack temperature gradient rather than
    an episode-specific warming."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = tab.index
    ok = tab.lost_cells >= MIN_LOST_CELLS
    ax.plot(x, tab.t2m_daybefore_edge_ice_kept, color=COL_STAY, lw=2.4,
            label="Edge ice still there the next day")
    ax.plot(x[ok], tab.t2m_daybefore_ice_lost[ok], "o", color=COL_LOST, ms=8, mec="black", mew=0.6,
            label=f"Ice gone the next day (days with ≥ {MIN_LOST_CELLS} cells)")
    ax.axhline(FREEZE, color="black", lw=1.4)
    shade_episodes(ax, FS_TICK - 1)
    date_axis(ax)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))                          # room for the episode labels
    ax.set_ylabel("2 m air temperature,\none day earlier (°C)", fontsize=FS_LABEL)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=FS_LEGEND - 1, frameon=False)
    d_ = (tab.t2m_daybefore_ice_lost - tab.t2m_daybefore_edge_ice_kept)[ok]
    print(f"Supplement: ice that vanished was warmer the day before on {(d_ > 0).sum()} of {len(d_)} days "
          f"(mean difference {d_.mean():+.1f} °C)")
    fig.savefig(out_fig, dpi=1000, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_fig}")
    return fig


# ---------------------------------------------------------------
# 5. CONSOLE SUMMARY
# ---------------------------------------------------------------
def summary(tab):
    print("\nEpisode summary (ABS index box, ice-covered points only):")
    print(f"{'':28s}{'edge ice':>10s}{'pack ice':>10s}{'% ice warm':>12s}"
          f"{'t-1 ice lost':>14s}{'t-1 edge ice kept':>19s}")
    spans = [(f"Episode {n} ({a}–{b} Aug)", a, b) for n, a, b in EPISODES] + \
            [("Cold spell 6–7 Aug", 6, 7), ("Regrowth 15–19 Aug", 15, 19), ("Regrowth 25–31 Aug", 25, 31)]
    for name, a, b in spans:
        t = tab.loc[pd.Timestamp(YEAR, MONTH, a):pd.Timestamp(YEAR, MONTH, b)]
        lw = t.lost_cells.fillna(0)
        lost_mean = np.nansum(t.t2m_daybefore_ice_lost * lw) / lw.sum() if lw.sum() else np.nan
        print(f"{name:28s}{t.t2m_edge_ice.mean():10.1f}{t.t2m_pack_ice.mean():10.1f}"
              f"{t.pct_ice_area_above_freeze.mean():12.0f}{lost_mean:14.1f}"
              f"{t.t2m_daybefore_edge_ice_kept.mean():19.1f}")
    print("\nAll values are 2 m air temperature (°C) at ERA5 points in the ABS index box, by the surface underneath:")
    print("  edge ice = concentration 15-80 %, pack ice = 80 % or more (OSI SAF, same day)")
    print("  %% ice warm = share of the ice-covered area where the air was above %.1f °C" % FREEZE)
    print("  t-1 ice lost = air the day before, at ice cells that were gone the next day (weighted by cells lost)")
    print("  t-1 edge ice kept = air the day before, at edge-ice cells that were still there the next day")


# ---------------------------------------------------------------
# 6. RUN
# ---------------------------------------------------------------
if __name__ == "__main__":
    SIC, SLAT, SLON = read_sic(SIC_FOLDER)
    T, lat, lonx, days = read_t2m(T2M_FILE)
    lat2, lonx2 = np.meshgrid(lat, lonx, indexing="ij")

    m_polar = Basemap(projection="spstere", boundinglat=-44, lon_0=180, resolution="l")
    OCEAN = ~land_at(m_polar, *m_polar(lonx2, lat2))

    idx = match_cells(SLAT, SLON, lat2, lonx2, MAX_MATCH_KM)
    SIC_E = {d: sic_on_era5(ic, idx) for d, ic in SIC.items()}
    print(f"ERA5 points matched to an OSI SAF cell: {np.mean(idx >= 0) * 100:.0f}%")

    TAB = daily_table(T, days, lat2, lonx2, OCEAN, SIC_E)
    TAB.round(3).to_csv(OUT_CSV)
    print(f"Saved: {OUT_CSV}")
    summary(TAB)
    make_figure(T, days, lat, lonx, OCEAN, SIC, SLAT, SLON, TAB, OUT_FIG)
    if MAKE_SUPPLEMENT:
        make_supplement(TAB, OUT_FIG_S)
    plt.show()
