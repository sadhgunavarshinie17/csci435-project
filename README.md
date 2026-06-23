# Food Waste AI System
Intelligent food waste management using computer vision to identify and measure
discarded food, analyse waste patterns, and provide recommendations to help
kitchens reduce waste and costs.

---

## Project Files

| File | Purpose |
|------|---------|
| `simple_split.py` | Step 1 — cleans and splits the raw dataset into train/val/test |
| `demo_pipeline.py` | Step 2 — runs enhancement, edge detection, segmentation, and detection |
| `quick_train_pipeline.py` | Step 3 — trains, fine-tunes, evaluates, and deploys the model (5–20 min) |
| `config.py` | Shared settings used by both pipelines (paths, classes, thresholds) |
| `requirements.txt` | All Python dependencies needed to run the project |
| `quick_train_data.yaml` | Auto-generated dataset config used during training |
| `yolov8s.pt` | Pretrained YOLOv8s model weights — used for quick training |
| `yolov8m.pt` | Pretrained YOLOv8m model weights — used for the visual demo |

---

## Generated Folders

| Folder | What's inside |
|--------|--------------|
| `split_food_waste_dataset/` | Cleaned dataset — output of simple_split.py |
| `enhanced_dataset/` | Enhanced images after running demo_pipeline.py |
| `edge_dataset/` | Edge detection maps (canny, sobel, laplacian, composite) |
| `segmented_dataset/` | Segmented images (grabcut, watershed, k-means) |
| `demo_output/` | 4-panel comparison images showing the full pipeline visually |
| `quick_sample_dataset/` | Small 200-image sample used for quick training |
| `runs/` | Training results, weights, and evaluation charts |
| `quick_eval_results/` | mAP, precision, recall scores from evaluation |

---

## How to Run

### 1. Install dependencies (once only)
```
pip install -r requirements.txt
```

### 2. Clean and split the dataset (already done)
```
python simple_split.py
```
Output: `split_food_waste_dataset/` with 4,713 train / 1,346 val / 675 test images.

### 3. Run the visual demo pipeline
```
python demo_pipeline.py
```
Runs: Image Acquisition → Enhancement → Edge Detection → Segmentation → Object Detection → Object Recognition

Output: `demo_output/` — open any `_demo.jpg` to see the 4-panel result.

### 4. Run the quick training pipeline
```
python quick_train_pipeline.py
```
Runs: Model Research → Training → Fine-Tuning → Evaluation → Deployment

Expected time: 5–20 minutes on a GPU.
When finished, the API server starts at `http://localhost:8000`.
Press `Ctrl+C` to stop it.

---

## API Usage (after running quick_train_pipeline.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Check server is running |
| `/detect` | POST | Upload an image, get waste analysis back |
| `/docs` | GET | Interactive API documentation |

Example response from `/detect`:
```json
{
  "total_detections": 5,
  "food_count": 3,
  "waste_count": 2,
  "waste_ratio": 0.4,
  "recommendation": "MODERATE WASTE: Review menu items with most waste.",
  "detections": [
    { "class": "Rice", "confidence": 0.87, "is_waste": false, "bbox": [10, 20, 100, 120] },
    { "class": "Bone", "confidence": 0.76, "is_waste": true,  "bbox": [200, 50, 300, 150] }
  ]
}
```

---

## Food Waste Classes (32 total)

**Food items (unconsumed):**
Apple, Bread, Bun, Egg-hard, Egg-scramble, Egg-steam, Fish, Meat, Mussel,
Noodle, Orange, Pancake, Pasta, Pear, Potato, Rice, Shrimp, Tofu, Tomato, Vegetable

**Waste items (non-edible remains):**
Apple-core, Apple-peel, Bone, Bone-fish, Egg-shell, Egg-yolk, Mussel-shell,
Orange-peel, Pear-core, Pear-peel, Shrimp-shell, Other-waste

---

## Waste Recommendation Thresholds

| Waste Ratio | Recommendation |
|-------------|---------------|
| 60% or above | HIGH WASTE — Reduce portion sizes immediately |
| 30% – 59% | MODERATE WASTE — Review menu items with most waste |
| Below 30% | LOW WASTE — Good waste management |

---

## Dataset
- Source: Roboflow — Food Waste Detection v11
- License: CC BY 4.0
- Original images: 7,622 (augmented 3x from source)
- After cleaning: 6,734 valid image-label pairs
- Format: YOLOv8 (640×640, YOLO bounding box annotations)
