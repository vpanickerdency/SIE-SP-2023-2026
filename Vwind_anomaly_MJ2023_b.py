"""
Panel (b): May-June 2023 meridional 10 m wind anomaly (shading) with 10 m wind-anomaly
vectors, relative to 2000-2022. ERA5.

- Grey contours: anomaly significant at p < 0.05 (2023 tested against the year-to-year
  spread of the 2000-2022 May-June means).
- Round map (nothing drawn outside 60°S), no title, "(b)" in the corner.
- Sector lines only over the ocean; sector names in the ocean.
- Arrows are rotated to the map, so v > 0 (northward) points away from the pole.
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
from scipy.interpolate import RegularGridInterpolator

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
WIND_FILE  = r"C:\Users\vpani\Downloads\Sept_SIE_SP\wind_u_v_vector.nc"
SAVE_PATH  = r"C:\Users\vpani\Downloads\Sept_SIE_SP\Vwind_Anomaly_MJ_2023_b.jpg"
CACHE_FILE = r"C:\Users\vpani\Downloads\Sept_SIE_SP\Vwind_MJ2023_cache.npz"  # delete if data change

PANEL_LETTER = "b"
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

N_ARROWS = 30                      # arrows per side of the evenly spaced grid
ARROW_SCALE = 60                   # larger = shorter arrows
REF_ARROW = 3                      # reference arrow length (m/s)

FS_PANEL, FS_SECTOR, FS_GRID, FS_CBAR, FS_CTICK = 30, 20, 17, 20, 17


# ------------------------------------------------------------------
# DATA (computed once, then cached)
# ------------------------------------------------------------------
def anomaly_and_pvalue(ym):
    clim = ym.sel(year=slice(2000, 2022)).values
    x = ym.sel(year=2023).values
    n = clim.shape[0]
    mu, sd = clim.mean(axis=0), clim.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (x - mu) / (sd * np.sqrt(1 + 1 / n))
    return x - mu, 2 * stats.t.sf(np.abs(t), df=n - 1)


t0 = time.time()
if os.path.exists(CACHE_FILE):
    print("Loading cached results...")
    c = np.load(CACHE_FILE)
    u_anom, v_anom, v_p, lat, lon = c["u_anom"], c["v_anom"], c["v_p"], c["lat"], c["lon"]
else:
    print("Computing yearly May-June means of u10 and v10 (one pass over the file)...")
    ds = xr.open_dataset(WIND_FILE, chunks={"valid_time": 500})
    sub = ds[["u10", "v10"]]
    sub = sub.sel(latitude=sub["latitude"][sub["latitude"] <= DATA_LAT_LIMIT])
    sub = sub.sel(valid_time=sub["valid_time.month"].isin(MONTHS))
    sub = sub.sel(valid_time=slice("2000-01-01", "2023-12-31"))
    ym = sub.groupby("valid_time.year").mean("valid_time").compute()

    v_anom, v_p = anomaly_and_pvalue(ym["v10"])
    u_anom, _ = anomaly_and_pvalue(ym["u10"])
    lat, lon = ym["latitude"].values, ym["longitude"].values
    np.savez_compressed(CACHE_FILE, u_anom=u_anom, v_anom=v_anom, v_p=v_p, lat=lat, lon=lon)
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


v_g, lon_g, lat_g = to_plot_grid(v_anom, lon, lat)
u_g, _, _ = to_plot_grid(u_anom, lon, lat)
sig_g, _, _ = to_plot_grid((v_p < 0.05).astype(float), lon, lat)

# ------------------------------------------------------------------
# MAP
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 12))
m = Basemap(projection="spstere", boundinglat=BOUNDING_LAT, lon_0=180,
            resolution="l", round=True, ax=ax)

lon2d, lat2d = np.meshgrid(lon_g, lat_g)
x, y = m(lon2d, lat2d)

cs = m.contourf(x, y, v_g, levels=np.arange(-5, 6, 1), cmap="RdBu_r", extend="both", zorder=1)
m.contour(x, y, sig_g, levels=[0.5], colors="grey", linewidths=1.5, zorder=2)

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

# ---- vectors: evenly spaced grid, directions taken from the map geometry ----
# On a polar map, north is always straight out from the pole, and east is at right
# angles to it, in the direction longitude increases on THIS map (checked below).
xp, yp = m(0, -90)                                   # pole position on the map
r60 = np.hypot(*np.subtract(m(0, BOUNDING_LAT), (xp, yp)))

xq, yq = np.meshgrid(np.linspace(m.xmin, m.xmax, N_ARROWS),
                     np.linspace(m.ymin, m.ymax, N_ARROWS))
dx, dy = xq - xp, yq - yp
r = np.hypot(dx, dy)

# longitude/latitude of every arrow, from its position relative to the pole
# (angles are measured from where 0° sits, in the direction 90°E sits)
x0, y0 = np.subtract(m(0, -70), (xp, yp))
x90, y90 = np.subtract(m(90, -70), (xp, yp))
turn = np.sign(x0 * y90 - y0 * x90)                  # +1: lon increases anticlockwise
ang0 = np.arctan2(y0, x0)
lonq = np.degrees(turn * (np.arctan2(dy, dx) - ang0))
lonq = ((lonq + 180) % 360) - 180
latq = 2 * np.degrees(np.arctan(r / r60 * np.tan(np.radians((90 + BOUNDING_LAT) / 2)))) - 90

pts = np.column_stack([latq.ravel(), lonq.ravel()])
u_pt = RegularGridInterpolator((lat_g, lon_g), u_g, bounds_error=False)(pts).reshape(latq.shape)
v_pt = RegularGridInterpolator((lat_g, lon_g), v_g, bounds_error=False)(pts).reshape(latq.shape)

with np.errstate(divide="ignore", invalid="ignore"):
    nx_, ny_ = dx / r, dy / r                        # north: outward from the pole
ex, ey = -turn * ny_, turn * nx_                     # east: along increasing longitude
uq = u_pt * ex + v_pt * nx_
vq = u_pt * ey + v_pt * ny_

# check: the arrow positions must match Basemap's own lon/lat (the ones the shading uses)
lon_bm, lat_bm = m(xq, yq, inverse=True)
inside = latq <= BOUNDING_LAT
dlon = np.abs(((lon_bm - lonq + 180) % 360) - 180)[inside & (latq < -61) & (latq > -85)]
print(f"Position check: max lon difference {dlon.max():.2f}°, "
      f"max lat difference {np.abs(lat_bm - latq)[inside].max():.2f}° (both should be ~0)")

bad = (latq > BOUNDING_LAT) | ~np.isfinite(uq) | ~np.isfinite(vq)
Q = m.quiver(xq, yq, np.ma.masked_where(bad, uq), np.ma.masked_where(bad, vq),
             scale=ARROW_SCALE, width=0.0035, headwidth=4.5, headlength=5,
             headaxislength=4.5, color="black", zorder=9)
ax.quiverkey(Q, 0.93, 0.03, REF_ARROW, f"{REF_ARROW} m/s", labelpos="N",
             coordinates="axes", fontproperties={"size": FS_GRID})

# ---- labels ----
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
                  aspect=28, ticks=np.arange(-5, 6, 1))
cb.set_label("Meridional wind anomaly (m/s)", fontsize=FS_CBAR)
cb.ax.tick_params(labelsize=FS_CTICK)

plt.savefig(SAVE_PATH, dpi=1000, bbox_inches="tight")
print("Saved:", SAVE_PATH)
plt.show()
