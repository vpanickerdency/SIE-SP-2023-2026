"""
Panel (a): May-June 2023 surface pressure anomaly (hPa) relative to 2000-2022. ERA5.

- Black contours over the continent: 2000-2022 May-June mean surface pressure (hPa).
- Grey contours: anomaly significant at p < 0.05 (2023 tested against the year-to-year
  spread of the 2000-2022 May-June means).
- Round map (nothing drawn outside 60°S), no title, "(a)" in the corner,
  longitude labels only around the circle.
- Sector lines only over the ocean; sector names in the ocean.
- Same size and style as panel (b), so the two images line up side by side.
- Results are cached after the first run; later runs only redraw.

Requirements: pip install numpy xarray dask scipy matplotlib basemap netCDF4
"""
import os
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.basemap import Basemap, addcyclic
from scipy import stats

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
SP_FILE    = r"C:\Users\vpani\Downloads\Sept_SIE_SP\surface_pressure_1.nc"
SAVE_PATH  = r"C:\Users\vpani\Downloads\Sept_SIE_SP\SP_Anomaly_MJ_2023_a.jpg"
CACHE_FILE = r"C:\Users\vpani\Downloads\Sept_SIE_SP\SP_MJ2023_cache.npz"   # delete if data change

PANEL_LETTER = "a"
MONTHS = [5, 6]
BOUNDING_LAT = -60
DATA_LAT_LIMIT = -55               # only read data south of this

SECTOR_LONS = [-130, -60, 20, 90, 160]
SECTOR_LABELS = {                  # (lon, lat): open ocean between the coast and 60°S
    "WSS": (-25, -64.5),
    "IOS": (55, -63.5),
    "WPS": (125, -63.0),
    "RSS": (-165, -66.5),
    "ABS": (-100, -65.5),
}

FS_PANEL, FS_SECTOR, FS_GRID, FS_CBAR, FS_CTICK, FS_ISO = 30, 20, 17, 20, 17, 11


# ------------------------------------------------------------------
# DATA (computed once, then cached)
# ------------------------------------------------------------------
t0 = time.time()
if os.path.exists(CACHE_FILE):
    print("Loading cached results...")
    c = np.load(CACHE_FILE)
    sp_anom, sp_p, sp_clim, lat, lon = c["sp_anom"], c["sp_p"], c["sp_clim"], c["lat"], c["lon"]
else:
    print("Computing yearly May-June means of surface pressure (one pass over the file)...")
    ds = xr.open_dataset(SP_FILE, chunks={"valid_time": 500})
    sp = ds["sp"]
    sp = sp.sel(latitude=sp["latitude"][sp["latitude"] <= DATA_LAT_LIMIT])
    sp = sp.sel(valid_time=sp["valid_time.month"].isin(MONTHS))
    sp = sp.sel(valid_time=slice("2000-01-01", "2023-12-31"))
    ym = (sp.groupby("valid_time.year").mean("valid_time") / 100.0).compute()   # Pa -> hPa

    clim_years = ym.sel(year=slice(2000, 2022)).values
    x2023 = ym.sel(year=2023).values
    n = clim_years.shape[0]
    sp_clim = clim_years.mean(axis=0)
    sd = clim_years.std(axis=0, ddof=1)
    sp_anom = x2023 - sp_clim
    with np.errstate(divide="ignore", invalid="ignore"):
        t = sp_anom / (sd * np.sqrt(1 + 1 / n))
    sp_p = 2 * stats.t.sf(np.abs(t), df=n - 1)

    lat, lon = ym["latitude"].values, ym["longitude"].values
    np.savez_compressed(CACHE_FILE, sp_anom=sp_anom, sp_p=sp_p, sp_clim=sp_clim, lat=lat, lon=lon)
    print("Saved cache:", CACHE_FILE)
print(f"Data ready in {time.time() - t0:.0f} s")


def to_plot_grid(field, lon, lat):
    """lon -> -180..180 increasing, lat increasing, plus a cyclic column (no seam at 180°)."""
    lon = ((np.asarray(lon) + 180) % 360) - 180
    lon, idx = np.unique(lon, return_index=True)
    field = np.asarray(field)[:, idx]
    lat = np.asarray(lat)
    if lat[0] > lat[-1]:
        lat, field = lat[::-1], field[::-1, :]
    field, lon = addcyclic(field, lon)
    return field, lon, lat


anom_g, lon_g, lat_g = to_plot_grid(sp_anom, lon, lat)
clim_g, _, _ = to_plot_grid(sp_clim, lon, lat)
sig_g, _, _ = to_plot_grid((sp_p < 0.05).astype(float), lon, lat)

# ------------------------------------------------------------------
# MAP
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 12))
m = Basemap(projection="spstere", boundinglat=BOUNDING_LAT, lon_0=180,
            resolution="l", round=True, ax=ax)

lon2d, lat2d = np.meshgrid(lon_g, lat_g)
x, y = m(lon2d, lat2d)

cs = m.contourf(x, y, anom_g, levels=np.arange(-10, 10.5, 1), cmap="RdBu_r",
                extend="both", zorder=1)
m.contour(x, y, sig_g, levels=[0.5], colors="grey", linewidths=1.8, zorder=2)

m.drawparallels([-80, -70], labels=[0, 0, 0, 0], linewidth=0.5, dashes=[1, 2],
                color="0.35", zorder=3)
m.drawmeridians(np.arange(-180, 180, 30), labels=[0, 0, 0, 0], linewidth=0.5,
                dashes=[1, 2], color="0.35", zorder=3)

# sector lines below the land fill -> visible only over the ocean
for L in SECTOR_LONS:
    lats = np.linspace(-90, BOUNDING_LAT, 300)
    xx, yy = m(np.full_like(lats, L), lats)
    m.plot(xx, yy, color="black", linewidth=2, zorder=4)

m.fillcontinents(color="lightgrey", zorder=5)
m.drawcoastlines(linewidth=1.0, zorder=6)
m.drawmapboundary(linewidth=2, color="black")

# climatological surface pressure, drawn over the continent only (where it shows topography)
iso = m.contour(x, y, clim_g, levels=np.arange(600, 961, 60), colors="black",
                linewidths=0.9, zorder=7)
ax.clabel(iso, inline=True, fontsize=FS_ISO, fmt="%1.0f")

# ---- labels: longitudes around the circle only ----
for L, txt in [(0, "0°"), (60, "60°E"), (120, "120°E"), (180, "180°"),
               (-120, "120°W"), (-60, "60°W")]:
    xx, yy = m(L, BOUNDING_LAT + 2.6)
    ax.text(xx, yy, txt, fontsize=FS_GRID, ha="center", va="center")
for name, (lo, la) in SECTOR_LABELS.items():
    xx, yy = m(lo, la)
    ax.text(xx, yy, name, fontsize=FS_SECTOR, fontweight="bold", ha="center", va="center",
            zorder=12, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
ax.text(0.0, 1.0, f"({PANEL_LETTER})", transform=ax.transAxes, fontsize=FS_PANEL,
        fontweight="bold", ha="left", va="top")

cb = fig.colorbar(cs, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07,
                  aspect=28, ticks=np.arange(-10, 11, 2))
cb.set_label("Surface pressure anomaly (hPa)", fontsize=FS_CBAR)
cb.ax.tick_params(labelsize=FS_CTICK)

plt.savefig(SAVE_PATH, dpi=1000, bbox_inches="tight")
print("Saved:", SAVE_PATH)
plt.show()
