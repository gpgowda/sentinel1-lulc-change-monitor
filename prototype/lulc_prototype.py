"""
LULC prototype: Sentinel-1 only -> 12 monthly land-use / land-cover maps for any place on Earth.

Input : latitude, longitude, start month (YYYY-MM)
Output: 12 monthly maps (10 km x 10 km box, 10 m pixels), % of area per class and month,
        month-to-month change and total change (first -> last month) in percentage points,
        a 12-panel figure, CSV tables, a GeoTIFF with 12 bands, optional comparison with Dynamic World.

The processing is the SAME as in training: Sentinel-1 GRD (IW, VV+VH, dB), monthly median composite,
Lee 5x5 speckle filter, clip + standardise with norm_stats.json, 128 x 128 patches into the U-Net.
"""
import io, json, os, time
import numpy as np
import pandas as pd

CLASS_NAMES = ['Tree', 'Grass', 'Crop', 'Built-up', 'Bare', 'Water']
COLORS = np.array([[34, 139, 34], [154, 205, 50], [240, 200, 60], [200, 40, 40],
                   [190, 170, 150], [40, 90, 200], [0, 0, 0]], dtype='uint8')      # last = no data / ignore
IGNORE = 255
BOX_PX, PATCH = 1000, 128                  # 10 km box at 10 m = 1000 x 1000 pixels
LEE_WINDOW, LEE_LOOKS = 5, 4.4
EE_PROJECT = 'deep-learning-project-511410'

# --------------------------------------------------------------------------------------------
# 1. The model (identical to the training notebook, cell 5b)
# --------------------------------------------------------------------------------------------
def build_unet():
    import torch, torch.nn as nn, torch.nn.functional as F

    class DoubleConv(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
        def forward(self, x): return self.block(x)

    class UNet(nn.Module):
        def __init__(self, in_ch=2, n_classes=6, base=32):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8, base * 16]
            self.enc1, self.enc2 = DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1])
            self.enc3, self.enc4 = DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3])
            self.bottom = DoubleConv(c[3], c[4])
            self.up4, self.dec4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2), DoubleConv(c[3] * 2, c[3])
            self.up3, self.dec3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2), DoubleConv(c[2] * 2, c[2])
            self.up2, self.dec2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2), DoubleConv(c[1] * 2, c[1])
            self.up1, self.dec1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2), DoubleConv(c[0] * 2, c[0])
            self.head = nn.Conv2d(c[0], n_classes, 1)
        def forward(self, x):
            e1 = self.enc1(x); e2 = self.enc2(F.max_pool2d(e1, 2)); e3 = self.enc3(F.max_pool2d(e2, 2))
            e4 = self.enc4(F.max_pool2d(e3, 2)); b = self.bottom(F.max_pool2d(e4, 2))
            d4 = self.dec4(torch.cat([self.up4(b), e4], 1)); d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], 1)); d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            return self.head(d1)
    return UNet()


def load_model(weights_path, device='cpu'):
    import torch
    ck = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and 'model' in ck:          # a "last" checkpoint holds more than the weights
        ck = ck['model']
    model = build_unet().to(device)
    model.load_state_dict(ck)
    return model.eval()

# --------------------------------------------------------------------------------------------
# 2. Pre-processing (identical to training: Lee filter + rescale with norm_stats.json)
# --------------------------------------------------------------------------------------------
def lee_filter(img_db, size=LEE_WINDOW, looks=LEE_LOOKS):
    from scipy.ndimage import uniform_filter
    bad = np.isnan(img_db)
    if bad.all():
        return img_db
    x = img_db.copy(); x[bad] = np.nanmedian(img_db)
    x = 10.0 ** (x / 10.0)
    mean = uniform_filter(x, size)
    var = np.maximum(uniform_filter(x * x, size) - mean ** 2, 0)
    ci2 = var / np.maximum(mean ** 2, 1e-12)
    w = np.clip(1.0 - (1.0 / looks) / np.maximum(ci2, 1e-12), 0.0, 1.0)
    out = 10.0 * np.log10(np.maximum(mean + w * (x - mean), 1e-6))
    out[bad] = np.nan
    return out.astype('float32')


def rescale(radar, stats):
    """radar (24,H,W) in dB, bands alternate VV, VH per month -> (12,2,H,W) standardised. NaN -> 0."""
    out = np.empty((12, 2) + radar.shape[1:], dtype='float32')
    for k, pol in enumerate(['VV', 'VH']):
        s = stats[pol]
        x = np.clip(radar[k::2], s['clip_low'], s['clip_high'])
        out[:, k] = (x - s['mean']) / s['std']
    return np.nan_to_num(out, nan=0.0)

# --------------------------------------------------------------------------------------------
# 3. Data retrieval from Google Earth Engine
# --------------------------------------------------------------------------------------------
def months_from(start):
    """'2021-03' -> [(2021,3), (2021,4), ..., (2022,2)]"""
    y, m = int(start[:4]), int(start[5:7])
    return [(y + (m - 1 + i) // 12, (m - 1 + i) % 12 + 1) for i in range(12)]


def ee_init(project=EE_PROJECT):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def box_grid(ee, lat, lon):
    """The same 10 km box on a 10 m UTM grid as in training."""
    from pyproj import Transformer
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    x, y = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True).transform(lon, lat)
    x0 = round((x - 5000) / 10) * 10
    y1 = round((y + 5000) / 10) * 10
    region = ee.Geometry.Rectangle([x0, y1 - 10000, x0 + 10000, y1], proj=f'EPSG:{epsg}', geodesic=False)
    return f'EPSG:{epsg}', region


def s1_collection(ee, region, y, m):
    start = ee.Date.fromYMD(y, m, 1)
    return (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filterBounds(region).filterDate(start, start.advance(1, 'month')))


def pick_orbit(ee, region, months):
    """Use one orbit direction for all 12 months (like in training): the one with more scenes."""
    n = {}
    for o in ['ASCENDING', 'DESCENDING']:
        total = 0
        for (y, m) in months:
            total += s1_collection(ee, region, y, m).filter(ee.Filter.eq('orbitProperties_pass', o)).size().getInfo()
        n[o] = total
    orbit = max(n, key=n.get)
    return orbit, n


def _download_tif(url, tries=3):
    import requests, rasterio
    from rasterio.io import MemoryFile
    last = None
    for t in range(tries):
        try:
            r = requests.get(url, timeout=300); r.raise_for_status()
            with MemoryFile(r.content) as mf, mf.open() as ds:
                return ds.read().astype('float32'), ds.profile
        except Exception as e:
            last = e; time.sleep(3 * (t + 1))
    raise RuntimeError(f'download failed after {tries} tries: {last}')


def _fit(a, h=BOX_PX, w=BOX_PX):
    """Cut or pad (with NaN) to exactly h x w."""
    out = np.full(a.shape[:-2] + (h, w), np.nan, dtype='float32')
    hh, ww = min(h, a.shape[-2]), min(w, a.shape[-1])
    out[..., :hh, :ww] = a[..., :hh, :ww]
    return out


def fetch_radar(ee, lat, lon, months, orbit, progress=print):
    """Returns radar (24,1000,1000) in dB (VV,VH per month, NaN = no data), a list of scene counts, and the raster profile."""
    crs, region = box_grid(ee, lat, lon)
    bands, counts, profile = [], [], None
    for i, (y, m) in enumerate(months):
        col = s1_collection(ee, region, y, m).filter(ee.Filter.eq('orbitProperties_pass', orbit))
        n = col.size().getInfo(); counts.append(n)
        if n == 0:
            progress(f'  month {i + 1:2d} ({y}-{m:02d}): NO Sentinel-1 scene, month stays empty')
            bands.append(np.full((2, BOX_PX, BOX_PX), np.nan, dtype='float32')); continue
        img = col.select(['VV', 'VH']).median().toFloat().unmask(-9999)
        url = img.getDownloadURL(dict(region=region, scale=10, crs=crs, format='GEO_TIFF'))
        arr, prof = _download_tif(url)
        arr[arr == -9999] = np.nan
        bands.append(_fit(arr[:2])); profile = profile or prof
        progress(f'  month {i + 1:2d} ({y}-{m:02d}): {n} scenes downloaded')
    radar = np.concatenate(bands, axis=0)                      # VV_m1, VH_m1, VV_m2, VH_m2, ...
    return radar, counts, profile, (crs, region)


def fetch_dynamic_world(ee, crs, region, months, progress=print):
    """Independent reference: Dynamic World label (mode over the 12 months) mapped to our 6 classes. Not used for training."""
    y0, m0 = months[0]; y1, m1 = months[-1]
    start = ee.Date.fromYMD(y0, m0, 1); end = ee.Date.fromYMD(y1, m1, 1).advance(1, 'month')
    img = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(region).filterDate(start, end)
           .select('label').mode().toFloat().unmask(255))
    url = img.getDownloadURL(dict(region=region, scale=10, crs=crs, format='GEO_TIFF'))
    arr, _ = _download_tif(url)
    lab = _fit(arr[:1])[0]
    # Dynamic World: 0 water, 1 trees, 2 grass, 3 flooded vegetation, 4 crops, 5 shrub&scrub, 6 built, 7 bare, 8 snow/ice
    lut = np.full(256, IGNORE, dtype='uint8')
    for dw, ours in {0: 5, 1: 0, 2: 1, 4: 2, 5: 1, 6: 3, 7: 4}.items():
        lut[dw] = ours
    lab = np.nan_to_num(lab, nan=255).astype('int64').clip(0, 255)
    return lut[lab]

# --------------------------------------------------------------------------------------------
# 4. Prediction, tables, figures
# --------------------------------------------------------------------------------------------
def predict_maps(model, z, device='cpu', batch=32):
    """z (12,2,1000,1000) standardised -> class maps (12,1000,1000) uint8. Tiles of 128 x 128 (as in training), edges mirrored."""
    import torch
    pad = (8 * PATCH - BOX_PX) // 2                                   # 12 px on every side -> 1024 x 1024
    zp = np.pad(z, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    out = np.zeros((12, 8 * PATCH, 8 * PATCH), dtype='uint8')
    tiles = [(m, r, c) for m in range(12) for r in range(8) for c in range(8)]
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            xb = np.stack([zp[m, :, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] for m, r, c in chunk])
            pr = model(torch.from_numpy(xb).to(device)).argmax(1).cpu().numpy().astype('uint8')
            for (m, r, c), p in zip(chunk, pr):
                out[m, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = p
    return out[:, pad:pad + BOX_PX, pad:pad + BOX_PX]


def class_tables(maps, labels):
    """% of area per class and month, month-to-month change and total change, both in percentage points."""
    pct = np.full((12, 6), np.nan)
    for m in range(12):
        valid = maps[m] != IGNORE                                    # months without any Sentinel-1 scene are 255 = empty
        if valid.any():
            pct[m] = 100.0 * np.bincount(maps[m][valid], minlength=6)[:6] / valid.sum()
    share = pd.DataFrame(pct, columns=CLASS_NAMES, index=labels).round(2)
    change = share.diff().iloc[1:].round(2)
    change.index = [f'{labels[i]} -> {labels[i + 1]}' for i in range(11)]
    ok = share.dropna().index                                        # first and last month that have data
    total = (share.loc[ok[-1]] - share.loc[ok[0]]).round(2).to_frame(f'{ok[0]} -> {ok[-1]}').T
    return share, change, total


def month_label(y, m):
    return f'{y}-{m:02d}'


def paint(a):
    return COLORS[np.where(a == IGNORE, 6, a)]


def plot_maps(maps, labels, title, path=None):
    import matplotlib; import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.6))
    for m, ax in enumerate(axes.ravel()):
        ax.imshow(paint(maps[m])); ax.set_title(labels[m]); ax.axis('off')
    fig.legend(handles=[Patch(color=COLORS[i] / 255, label=n) for i, n in enumerate(CLASS_NAMES)],
               loc='lower center', ncol=6, frameon=False, fontsize=12)
    fig.suptitle(title, fontsize=15); plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    if path: plt.savefig(path, dpi=130)
    return fig


def plot_share(share, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, n in enumerate(CLASS_NAMES):
        ax.plot(range(12), share[n].values, marker='o', color=COLORS[i] / 255, label=n)
    ax.set_xticks(range(12)); ax.set_xticklabels(share.index, rotation=45)
    ax.set_ylabel('% of the 10 x 10 km box'); ax.set_title('Predicted class share per month'); ax.legend(ncol=6, frameon=False)
    plt.tight_layout()
    if path: plt.savefig(path, dpi=130)
    return fig


def compare_with_reference(maps, ref):
    """Model (most frequent class over the 12 months) against an independent map. Returns dict with OA, mIoU, per-class IoU, shares."""
    mode = np.argmax(np.stack([(maps == k).sum(0) for k in range(6)]), axis=0).astype('uint8')
    ok = ref != IGNORE
    cm = np.bincount(ref[ok].astype('int64') * 6 + mode[ok], minlength=36).reshape(6, 6).astype('float64')
    tp = np.diag(cm); iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    present = cm.sum(1) > 0
    shares = pd.DataFrame({'model (12-month mode) %': 100 * np.bincount(mode.ravel(), minlength=6)[:6] / mode.size,
                           'reference %': 100 * np.bincount(ref[ok], minlength=6)[:6] / ok.sum()}, index=CLASS_NAMES).round(1)
    return dict(OA=float(tp.sum() / cm.sum()), mIoU=float(iou[present].mean()), iou=dict(zip(CLASS_NAMES, iou.round(3))),
                confusion=cm, shares=shares, reference_map=ref, model_mode=mode)

# --------------------------------------------------------------------------------------------
# 5. The whole prototype in one call
# --------------------------------------------------------------------------------------------
def run(lat, lon, start, weights_path, stats_path, out_dir='lulc_output', device=None,
        compare_dw=True, place_name=None, fetch=None, progress=print):
    """lat, lon: centre of the 10 x 10 km box. start: 'YYYY-MM' (first of 12 consecutive months)."""
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    stats = json.load(open(stats_path))
    months = months_from(start); labels = [month_label(*ym) for ym in months]
    name = place_name or f'{lat:.3f}_{lon:.3f}'
    t0 = time.time()

    ee, crs_region, profile, ref = None, None, None, None
    progress(f'1/5 Sentinel-1 for {name} ({lat}, {lon}), {labels[0]} ... {labels[-1]}')
    if fetch is None:
        ee = ee_init()
        crs, region = box_grid(ee, lat, lon)
        orbit, n = pick_orbit(ee, region, months)
        progress(f'  orbit direction used: {orbit} (scenes in 12 months: {n})')
        radar, counts, profile, (crs, region) = fetch_radar(ee, lat, lon, months, orbit, progress)
    else:                                                          # used for testing without internet
        radar, counts, profile = fetch(lat, lon, months)
    empty_months = [labels[i] for i, c in enumerate(counts) if c == 0]
    empty_pct = 100 * float(np.isnan(radar).mean())

    progress('2/5 speckle filter (Lee 5x5) and standardisation')
    radar = np.stack([lee_filter(radar[b]) for b in range(24)])
    z = rescale(radar, stats)

    progress(f'3/5 U-Net prediction on {device} (12 months x 64 tiles)')
    model = load_model(weights_path, device)
    maps = predict_maps(model, z, device)
    for i, c in enumerate(counts):
        if c == 0:
            maps[i] = IGNORE                                       # no scene in that month: no map, no numbers

    progress('4/5 class shares and changes')
    share, change, total = class_tables(maps, labels)
    share.to_csv(f'{out_dir}/class_share_per_month.csv'); change.to_csv(f'{out_dir}/change_month_to_month_pp.csv')
    total.to_csv(f'{out_dir}/change_total_pp.csv')
    fig1 = plot_maps(maps, labels, f'LULC from Sentinel-1 only, {name}', f'{out_dir}/maps_12_months.png')
    fig2 = plot_share(share, f'{out_dir}/share_per_month.png')
    np.save(f'{out_dir}/maps_12_months.npy', maps)
    if profile is not None:
        try:
            import rasterio
            prof = dict(profile); prof.update(count=12, dtype='uint8', nodata=None, compress='deflate')
            prof.update(height=BOX_PX, width=BOX_PX)
            with rasterio.open(f'{out_dir}/maps_12_months.tif', 'w', **prof) as dst:
                dst.write(maps)
        except Exception as e:
            progress(f'  (GeoTIFF not written: {str(e)[:100]})')

    result = dict(share=share, change=change, total=total, maps=maps, labels=labels, scene_counts=counts,
                  empty_months=empty_months, empty_pct=empty_pct, figs=(fig1, fig2), comparison=None)
    if compare_dw and ee is not None:
        progress('5/5 comparison with Dynamic World (independent, not used for training)')
        try:
            ref = fetch_dynamic_world(ee, crs, region, months, progress)
            cmp_ = compare_with_reference(maps, ref); result['comparison'] = cmp_
            cmp_['shares'].to_csv(f'{out_dir}/comparison_dynamic_world_shares.csv')
            json.dump(dict(OA=cmp_['OA'], mIoU=cmp_['mIoU'], iou={k: float(v) for k, v in cmp_['iou'].items()}),
                      open(f'{out_dir}/comparison_dynamic_world.json', 'w'), indent=2)
        except Exception as e:
            progress(f'  comparison failed: {str(e)[:200]}')
    progress(f'done in {(time.time() - t0) / 60:.1f} min, files in {out_dir}')
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='LULC prototype (Sentinel-1 only)')
    ap.add_argument('--lat', type=float, required=True); ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--start', required=True, help='first month, e.g. 2021-03')
    ap.add_argument('--weights', default='weights/unet_lulc_s1.pt'); ap.add_argument('--stats', default='norm_stats.json')
    ap.add_argument('--out', default='lulc_output'); ap.add_argument('--name', default=None)
    a = ap.parse_args()
    r = run(a.lat, a.lon, a.start, a.weights, a.stats, a.out, place_name=a.name)
    print(r['share'].to_string()); print(r['change'].to_string()); print(r['total'].to_string())
