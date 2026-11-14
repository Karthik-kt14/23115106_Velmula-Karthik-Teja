# 23115106 — Reproducible Oxford-IIIT Pet Segmentation

Student: Pentapalli Sai Vardhini | Roll No.: 23115106

This experiment compares a compact U-Net baseline with a controlled variant that changes only bottleneck convolution dilation (1 → 2). Dataset: Oxford-IIIT Pet, official trimap annotations and official trainval/test lists.

## Dataset contract
- Trimap 1 = pet foreground; 2 = background; 3 = uncertain boundary (ignored).
- RGB images resized to 256×256 (bilinear); masks resized with nearest-neighbor.
- Official trainval split is divided 85%/15% (seed 42); official test remains held out.
- Images scaled to [0,1]. Training augmentation: horizontal flip only.
- No pseudo-ground-truth labels are used.

## Run
```bash
pip install -r requirements.txt
python -m src.prepare_data --output data/oxford_pet --raw data/raw
python -m src.train --data data/oxford_pet --model unet --output outputs/unet
python -m src.train --data data/oxford_pet --model dilated_unet --output outputs/dilated_unet
python -m src.evaluate --data data/oxford_pet --checkpoint outputs/unet/best.pt --model unet --output outputs/eval_unet
python -m src.evaluate --data data/oxford_pet --checkpoint outputs/dilated_unet/best.pt --model dilated_unet --output outputs/eval_dilated
```
Default: seed 42, 256px, batch 8, 20 epochs, Adam lr=0.001, masked BCE-with-logits, early stopping patience 5 on validation Dice. Run on Colab GPU if available.

## Status / integrity
Code and report structure are supplied. **Metrics are not fabricated.** Run experiments before filling report tables or claiming results. Latency is hardware-dependent. The experiment does not establish generalization beyond this dataset/split.
Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
