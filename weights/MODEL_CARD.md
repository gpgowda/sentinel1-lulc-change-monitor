# Model card: unet_lulc_s1.pt

| | |
| --- | --- |
| Architecture | U-Net, base width 32, 4 pooling levels |
| Parameters | 7,760,000 |
| Input | 2 channels (VV, VH), 128 x 128, one month, standardised with `config/norm_stats.json` |
| Output | 6 class scores per pixel |
| Classes | 0 Tree, 1 Grass, 2 Crop, 3 Built-up, 4 Bare, 5 Water |
| Training data | 228,732 samples from 389 boxes worldwide, Sentinel-1 2021 |
| Epochs | 10 (epoch 10 = best validation mIoU) |
| Validation | mIoU 57.8 %, OA 74.2 % |
| Test (9 unseen boxes, India) | mIoU 44.7 %, OA 67.2 %, macro-F1 58.8 % |

Per-class IoU on the test set: Tree 48.5, Grass 20.1, Crop 47.5, Built-up 66.1, Bare 16.4, Water 69.7.

The file stores a PyTorch `state_dict` only. Build the network with the `UNet` class in
`prototype/lulc_prototype.py`, then `load_state_dict`. The weights are useless without
`config/norm_stats.json`, because the model expects input standardised with exactly those constants.
