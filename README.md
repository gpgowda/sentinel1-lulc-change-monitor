# Land Use / Land Cover from Sentinel-1 SAR

Semantic segmentation of land use and land cover from **Sentinel-1 radar only** - no optical
imagery, no elevation data, no auxiliary layers - plus a prototype that turns any coordinate
into twelve monthly LULC maps with class percentages and month-to-month change.

Course project, Data Analytics 3: Predictive Analytics and Deep Learning,
SRH Hochschule Heidelberg.

## Results

| Evaluation | Result |
| --- | --- |
| Validation (7 boxes) | mIoU 57.8 %, OA 74.2 % |
| **Test, 9 unseen boxes in India** | **mIoU 44.7 %, OA 67.2 %, macro-F1 58.8 %** |
| Heidelberg case study vs Dynamic World | 87.4 % agreement |

Six classes: Tree, Grass, Crop, Built-up, Bare, Water.

The India boxes were never used for training or model selection and were evaluated exactly once.

## What is here

| Path | Contents |
| --- | --- |
| `prototype/lulc_prototype.py` | The prototype: retrieval, preprocessing, inference, reporting |
| `weights/unet_lulc_s1.pt` | Trained U-Net weights, 7.76 M parameters, 31 MB |
| `config/norm_stats.json` | Clip limits and standardisation constants - **required at inference** |
| `config/class_map.json` | WorldCover code to 6-class mapping |
| `data/places_profile*.csv` | The 405 boxes with coordinates, region, split and orbit |
| `notebooks/` | The full training notebook with outputs |
| `results/` | Test scores, training histories, figures |

**Dataset** (13.9 GB of prepared 128x128 patches, too large for GitHub):
https://drive.google.com/drive/folders/1ARzErZZ3OV_e4oAIX3ez-_eNli6cg6-0

## Setup

### 1. Dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Earth Engine account (required)

The prototype downloads Sentinel-1 data from Earth Engine, so an account is needed.

1. Sign up at <https://earthengine.google.com/signup/> - free for research and education.
2. Create or note a Google Cloud project id and register it for Earth Engine.
3. Put that project id at the top of `prototype/lulc_prototype.py`:

   ```python
   EE_PROJECT = 'your-project-id'
   ```

4. Authenticate once:

   ```bash
   earthengine authenticate
   ```

No other accounts or API keys are needed. The model weights are in this repository.

## Running the prototype

```bash
python prototype/lulc_prototype.py \
    --lat 49.40 --lon 8.69 --start 2021-03 \
    --weights weights/unet_lulc_s1.pt \
    --stats config/norm_stats.json \
    --out lulc_output --name Heidelberg
```

| Argument | Meaning |
| --- | --- |
| `--lat`, `--lon` | Centre of the 10 x 10 km box |
| `--start` | First of twelve consecutive months, `YYYY-MM` |
| `--weights` | Path to the model weights |
| `--stats` | Path to `norm_stats.json` |
| `--out` | Output folder |

A run takes about 3-10 minutes, dominated by the twelve Earth Engine downloads.
A GPU is optional; inference runs on CPU in under a minute.

### Outputs

| File | Contents |
| --- | --- |
| `class_share_per_month.csv` | 12 x 6 table of area shares |
| `change_month_to_month_pp.csv` | Consecutive change in percentage points |
| `change_total_pp.csv` | First to last month change per class |
| `maps_12_months.png` | The twelve maps as one figure |
| `share_per_month.png` | Class shares over the year |
| `maps_12_months.tif` | 12-band georeferenced GeoTIFF |
| `comparison_dynamic_world.json` | Agreement with the independent reference |

### From a notebook

```python
import lulc_prototype as L
res = L.run(49.40, 8.69, '2021-03',
            weights_path='weights/unet_lulc_s1.pt',
            stats_path='config/norm_stats.json',
            out_dir='lulc_output', place_name='Heidelberg')
print(res['share'])   # % per class per month
print(res['change'])  # month-to-month change, percentage points
```

## Important limitations

Read these before using the output for anything.

- **Change figures below roughly 3-5 percentage points are not meaningful.**
  17.9 % of pixels change predicted class between consecutive months on the test set,
  although the ground truth cannot change. Month-to-month differences are dominated by
  seasonal radar behaviour and model noise, not by real land-use change.
- **Performance drops on unfamiliar regions.** mIoU falls from 57.8 % on validation to
  44.7 % on unseen Indian terrain. The model gives no warning when it is out of its depth.
- **Bare ground is unreliable** away from the training deserts: 16.4 % IoU, 19.2 % recall.
- **Grass is weak everywhere**, 16-22 % IoU across all evaluations.
- **Built-up is under-reported in leafy suburbs.** Tree canopy dominates the radar return,
  so houses under street trees are read as forest.

## Model

U-Net trained from scratch, 7.76 M parameters, 2 input channels (VV, VH) of a single month,
6 output classes. Weighted cross-entropy, AdamW at 1e-3 with cosine decay, batch size 32,
10 epochs on 228,732 samples. A variant with an ImageNet-pretrained ResNet34 encoder
(24.4 M parameters) scored lower on the robust measures and was not used.

The model classifies **one month at a time**, which is what makes per-month maps possible.
A model fed all twelve months at once would likely score higher but could only produce one
annual map.

## Data sources

| Source | Used for |
| --- | --- |
| [Sentinel-1 GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) (Copernicus / ESA) | The only model input |
| [ESA WorldCover v200](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) | Training and test labels |
| [Dynamic World V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) | Independent reference for the case study only, never used in training |

Sentinel-1 and WorldCover are Copernicus / ESA products; see their own licence terms.
