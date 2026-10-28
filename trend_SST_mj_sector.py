import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Circle
import matplotlib.patheffects as pe
from scipy.stats import ttest_ind
import time

start_total = time.time()

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
file_path = r"C:\Users\vpani\Downloads\Sept_SIE_SP\sst_1940_2025.nc"
save_path = r"C:\Users\vpani\Downloads\Sept_SIE_SP\SST_Anomaly_MJ_2023.jpg"

SHOW_SIGNIFICANCE = False   # True = draw grey contour around p < 0.05 areas (not in sample image)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
print("Opening dataset with Dask...")
ds = xr.open_dataset(file_path, chunks={'valid_time': 300})
print("Dataset opened!")

# ERA5 SST variable is 'sst' (Kelvin). A difference in K == difference in °C,
# but convert anyway so the climatology values are in °C.
var_name = 'sst' if 'sst' in ds else list(ds.data_vars)[0]
sst = ds[var_name] - 273.15

# May–June only
sst_mj = sst.sel(valid_time=sst['valid_time.month'].isin([5, 6]))

sst_clim = sst_mj.sel(valid_time=slice("2000-01-01", "2022-12-31"))
sst_2023 = sst_mj.sel(valid_time=slice("2023-05-01", "2023-06-30"))

# ------------------------------------------------------------------
# Anomaly
# ------------------------------------------------------------------
print("Computing climatology...")
sst_clim_mean = sst_clim.groupby("valid_time.month").mean("valid_time").compute()

print("Computing 2023 mean...")
sst_2023_mean = sst_2023.groupby("valid_time.month").mean("valid_time").compute()

sst_anom = (sst_2023_mean - sst_clim_mean).mean("month")
print("Anomaly computed")

# Land / missing points are NaN in ERA5 SST -> mask them for plotting
anom_plot = np.ma.masked_invalid(sst_anom.values)

# ------------------------------------------------------------------
# Significance (optional)
# ------------------------------------------------------------------
if SHOW_SIGNIFICANCE:
    print("Computing p-values...")
    t_stat, p_values = ttest_ind(sst_2023.values, sst_clim.values,
                                 axis=0, equal_var=False, nan_policy='omit')
    p_values = np.ma.filled(p_values, np.nan)
    print("P-values done")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
lat = ds['latitude'].values
lon = ds['longitude'].values
lon2d, lat2d = np.meshgrid(lon, lat)

fig, ax = plt.subplots(figsize=(12, 12))

# Same map setup as the SP script: square map bounded at 60°S, circle drawn on top
m = Basemap(projection='spstere', boundinglat=-60, lon_0=180, resolution='l', ax=ax)

x, y = m(lon2d, lat2d)

# -2 to 2 °C in 0.25 steps (matches the sample colorbar)
levels = np.arange(-2, 2.01, 0.25)

# Light-grey disc under the shading: fills NaN cells along the coast / under ice
# shelves so no white slivers appear between the SST colours and the land
ax.add_patch(Circle((0.5, 0.5), 0.5, transform=ax.transAxes,
                    facecolor='lightgrey', edgecolor='none', zorder=0))

cs = m.contourf(x, y, anom_plot, levels=levels, cmap='RdYlBu_r',
                extend='both', zorder=1)

sig_cs = None
if SHOW_SIGNIFICANCE:
    sig = np.where(np.isnan(p_values), 0, (p_values < 0.05).astype(int))
    sig_cs = m.contour(x, y, sig, levels=[0.5], colors='grey', linewidths=1.5, zorder=6)

coast = m.drawcoastlines(linewidth=1.0, zorder=6)
land = m.fillcontinents(color='lightgrey', alpha=1.0, zorder=5)

m.drawparallels(np.arange(-90, -49, 10), labels=[1, 0, 0, 0], fontsize=24,
                linewidth=0.5, dashes=[4, 4], color='grey')
m.drawmeridians(np.arange(-180, 181, 30), labels=[0, 0, 0, 1], fontsize=24,
                linewidth=0.5, dashes=[4, 4], color='grey')

# Circular frame
circle = Circle((0.5, 0.5), 0.5, transform=ax.transAxes, fill=False, linewidth=2, zorder=12)
ax.add_patch(circle)

# Clip shading, land and coastlines to the circle so the corners stay white
clip_circle = Circle((0.5, 0.5), 0.5, transform=ax.transAxes)

def clip_to_circle(obj):
    if obj is None:
        return
    items = obj if isinstance(obj, (list, tuple)) else [obj]
    for it in items:
        try:
            it.set_clip_path(clip_circle)
        except Exception:
            pass
        for c in getattr(it, 'collections', []):   # older matplotlib contour sets
            c.set_clip_path(clip_circle)

clip_to_circle(cs)
clip_to_circle(sig_cs)
clip_to_circle(coast)
clip_to_circle(land)

# Sector boundaries: drawn UNDER the grey land (zorder 4 < land zorder 5),
# so the continent hides the inner part and they show from the coast out to 60°S
sector_lons = [-130, -60, 20, 90, 160]
for L in sector_lons:
    lats = np.linspace(-90, -60, 200)
    lons = np.ones_like(lats) * L
    xx, yy = m(lons, lats)
    m.plot(xx, yy, color='black', linewidth=2, zorder=4)

# Sector names in open water (between the coast and 60°S), with a thin white outline
def place_text(lon_pt, lat_pt, text):
    xx, yy = m(lon_pt, lat_pt)
    plt.text(xx, yy, text, fontsize=26, fontweight='bold',
             ha='center', va='center', zorder=11,
             path_effects=[pe.withStroke(linewidth=3.5, foreground='white')])

place_text(-25, -65, "WSS")   # Weddell Sea      (60°W – 20°E)
place_text(55, -63, "IOS")    # Indian Ocean     (20°E – 90°E)
place_text(125, -63, "WPS")   # W. Pacific       (90°E – 160°E)
place_text(-170, -68, "RSS")  # Ross Sea         (160°E – 130°W)
place_text(-100, -67, "ABS")  # Amundsen–Bellingshausen (130°W – 60°W)

cbar = m.colorbar(cs, location='bottom', pad="8%")
cbar.set_ticks(np.arange(-2, 2.1, 1))
cbar.set_label('SST Anomaly (°C)', fontsize=26)
cbar.ax.tick_params(labelsize=22)


plt.savefig(save_path, dpi=1000, bbox_inches='tight')
print("\nFigure saved successfully!")

plt.show()

print("\nTOTAL TIME:", round(time.time() - start_total, 2), "seconds")
