# Detection Module — Model Training Guide

## Why Fine-Tuning is Needed

The base YOLOv8 model (trained on COCO) can detect `person` out of the box.
However, the four compliance classes require detecting:
- `green_vest` vs `red_black_vest` (vest color classification)
- `open_panel` vs `closed_panel` (electrical panel state)
- `block` (standardized blocks on forklift forks)
- `forklift` (factory forklift)

These are NOT in COCO. You must fine-tune on factory-specific data.

## Dataset

Use the Kaggle dataset provided in the assignment:
https://www.kaggle.com/datasets/trnhhnggiang/videodataset-for-safe-and-unsafe-behaviours

### Download
```bash
pip install kaggle
kaggle datasets download -d trnhhnggiang/videodataset-for-safe-and-unsafe-behaviours
unzip videodataset-for-safe-and-unsafe-behaviours.zip -d data/raw_kaggle/
```

## Labeling Strategy

Use [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com/) to annotate frames.

### Classes to label (match these names exactly in your dataset):
```
0: person
1: green_vest
2: red_black_vest
3: open_panel
4: closed_panel
5: block
6: forklift
```

### Export format: YOLO v8 (txt annotations + images)

## Training

```bash
# Install ultralytics
pip install ultralytics

# Create dataset.yaml
cat > dataset.yaml << 'EOF'
path: ../data/factory_dataset
train: images/train
val: images/val

nc: 7
names: ['person', 'green_vest', 'red_black_vest', 'open_panel', 'closed_panel', 'block', 'forklift']
EOF

# Fine-tune from pretrained weights
yolo detect train \
  model=yolov8n.pt \
  data=dataset.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=runs/factory \
  name=compliance_model
```

## After Training

Update `config.py`:
```python
YOLO_MODEL_PATH = "runs/factory/compliance_model/weights/best.pt"
```

## Running Without a Fine-Tuned Model

The system will still run with `yolov8n.pt` (base COCO model).
- `person` detection works natively.
- Vest color is determined by HSV color analysis (no fine-tuned vest class needed).
- `forklift` detections can still come from COCO `truck` proxies.
- `block` and `open_panel` remain much more reliable with a fine-tuned model, but
  the backend now includes fallback heuristics so Section 6.3.2 can still fire on
  clear forklift-overload scenes.

Classes 0 and 1 (walkway + vest analysis) work immediately with the base model.
Classes 2 and 3 are still best served by a fine-tuned detector, but they are no
longer hard-disabled when only the base model is available.
