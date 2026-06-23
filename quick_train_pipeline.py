"""
Food Waste AI System
FILE: quick_train_pipeline.py

Runs in 5-20 minutes — no full training needed.
Fine-tunes a pretrained YOLOv8m on a small sample of your dataset.

Includes:
  1. Model Research   — picks the best model for your setup
  2. Quick Training   — 5 epochs on 200 images (~5-10 min)
  3. Fine-Tuning      — 3 epochs frozen backbone (~2-5 min)
  4. Model Evaluation — mAP, precision, recall on test sample
  5. Model Deployment — launches FastAPI server

HOW TO RUN:
  pip install ultralytics opencv-python numpy fastapi uvicorn python-multipart pyyaml
  python quick_train_pipeline.py
"""

import os
import json
import shutil
import random
import yaml
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
ENHANCED_DIR     = "./enhanced_dataset"       # output of demo_pipeline.py / model_pipeline.py
RAW_DIR          = "./split_food_waste_dataset"  # fallback if enhanced not ready
SAMPLE_DIR       = "./quick_sample_dataset"   # small sample we train on
DATA_YAML        = "./quick_train_data.yaml"
TRAIN_PROJECT    = "./runs/quick_train"
TRAIN_NAME       = "food_waste_quick"
FINETUNE_PROJECT = "./runs/quick_finetune"
FINETUNE_NAME    = "food_waste_finetuned_quick"
EVAL_DIR         = "./quick_eval_results"

# ── Sample size — keeps training fast ──
TRAIN_SAMPLES = 200   # images for training
VAL_SAMPLES   = 50    # images for validation
TEST_SAMPLES  = 30    # images for evaluation

# ── Training settings — minimal for speed ──
QUICK_EPOCHS    = 5    # full pass through data
FINETUNE_EPOCHS = 3    # fine-tune pass
BATCH           = 8
IMGSZ           = 416  # smaller than 640 → faster per epoch
DEVICE          = 0    # GPU 0
CONFIDENCE      = 0.25

CLASSES = [
    'Apple', 'Apple-core', 'Apple-peel', 'Bone', 'Bone-fish', 'Bread', 'Bun',
    'Egg-hard', 'Egg-scramble', 'Egg-shell', 'Egg-steam', 'Egg-yolk', 'Fish',
    'Meat', 'Mussel', 'Mussel-shell', 'Noodle', 'Orange', 'Orange-peel',
    'Other-waste', 'Pancake', 'Pasta', 'Pear', 'Pear-core', 'Pear-peel',
    'Potato', 'Rice', 'Shrimp', 'Shrimp-shell', 'Tofu', 'Tomato', 'Vegetable'
]

WASTE_ITEMS = [
    "Apple-core", "Apple-peel", "Bone", "Bone-fish", "Egg-shell",
    "Egg-yolk", "Mussel-shell", "Orange-peel", "Pear-core",
    "Pear-peel", "Shrimp-shell", "Other-waste"
]


# ─────────────────────────────────────────────────────────────
# HELPER — find best available dataset directory
# ─────────────────────────────────────────────────────────────
def get_source_dir():
    # Prefer enhanced dataset, fall back to raw split
    for split_name in ["train", "valid"]:
        enhanced = os.path.join(ENHANCED_DIR, "train" if split_name == "train" else "val", "images")
        if os.path.exists(enhanced) and len(os.listdir(enhanced)) > 0:
            print(f"  Using enhanced dataset.")
            return ENHANCED_DIR, {"train": "train", "valid": "val", "test": "test"}
    print(f"  Enhanced dataset not found — using raw split dataset.")
    return RAW_DIR, {"train": "train", "valid": "valid", "test": "test"}


# ═════════════════════════════════════════════════════════════
# MODULE 1: MODEL RESEARCH
# ═════════════════════════════════════════════════════════════
def run_model_research():
    print("\n── Module 1: Model Research ─────────────────────────────")

    models = {
        "YOLOv8n (nano)":   {"params": "3.2M",  "mAP": 37.3, "speed_ms": 1.5,  "use": "Fastest — good for quick demo"},
        "YOLOv8s (small)":  {"params": "11.2M", "mAP": 44.9, "speed_ms": 2.4,  "use": "Balanced — USED IN THIS DEMO"},
        "YOLOv8m (medium)": {"params": "25.9M", "mAP": 50.2, "speed_ms": 5.0,  "use": "Best accuracy — use for full training"},
        "YOLOv8l (large)":  {"params": "43.7M", "mAP": 52.9, "speed_ms": 7.8,  "use": "High accuracy, slower"},
        "YOLOv8x (xlarge)": {"params": "68.2M", "mAP": 53.9, "speed_ms": 13.7, "use": "Maximum accuracy, strong GPU needed"},
    }

    print(f"\n  {'Model':<22} {'Params':<10} {'mAP50':<10} {'Speed(ms)':<12} {'Best For'}")
    print("  " + "-" * 72)
    for name, info in models.items():
        marker = " ◄" if "USED" in info["use"] else ""
        print(f"  {name:<22} {info['params']:<10} {info['mAP']:<10} {info['speed_ms']:<12} {info['use']}{marker}")

    print("\n  ✅ Using YOLOv8s for quick demo — fast training, good accuracy.")
    print("     For production: switch to YOLOv8m in train_pipeline.py")


# ═════════════════════════════════════════════════════════════
# MODULE 2: QUICK TRAINING
# Samples a small subset → fine-tunes pretrained YOLOv8s
# ═════════════════════════════════════════════════════════════
def build_sample_dataset(source_dir, split_map):
    """Copies a small random sample into quick_sample_dataset/"""
    print(f"\n  Building sample dataset...")
    random.seed(42)

    if os.path.exists(SAMPLE_DIR):
        shutil.rmtree(SAMPLE_DIR)

    sample_counts = {
        "train":  TRAIN_SAMPLES,
        "valid":  VAL_SAMPLES,
        "test":   TEST_SAMPLES,
    }

    for split, n in sample_counts.items():
        src_split  = split_map[split]
        src_img    = os.path.join(source_dir, src_split, "images")
        src_lbl    = os.path.join(source_dir, src_split, "labels")
        dst_img    = os.path.join(SAMPLE_DIR, split, "images")
        dst_lbl    = os.path.join(SAMPLE_DIR, split, "labels")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        if not os.path.exists(src_img):
            print(f"  [SKIP] {src_img} not found.")
            continue

        all_files = [f for f in os.listdir(src_img) if not f.startswith(".")]
        sampled   = random.sample(all_files, min(n, len(all_files)))

        copied = 0
        for fname in sampled:
            src_i = os.path.join(src_img, fname)
            dst_i = os.path.join(dst_img, fname)
            lname = os.path.splitext(fname)[0] + ".txt"
            src_l = os.path.join(src_lbl, lname)
            dst_l = os.path.join(dst_lbl, lname)

            if os.path.exists(src_i) and os.path.exists(src_l):
                shutil.copy(src_i, dst_i)
                shutil.copy(src_l, dst_l)
                copied += 1

        print(f"  [{split}] {copied} images sampled.")


def write_data_yaml():
    config = {
        "train": f"{SAMPLE_DIR}/train/images",
        "val":   f"{SAMPLE_DIR}/valid/images",
        "test":  f"{SAMPLE_DIR}/test/images",
        "nc":    32,
        "names": CLASSES
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Data config written → {DATA_YAML}")


def run_quick_training():
    print("\n── Module 2: Quick Training ─────────────────────────────")
    print(f"  Training on {TRAIN_SAMPLES} images for {QUICK_EPOCHS} epochs...")
    print(f"  Image size: {IMGSZ}x{IMGSZ} | Batch: {BATCH}")
    print("  Expected time: 3–8 minutes on GPU\n")

    source_dir, split_map = get_source_dir()
    build_sample_dataset(source_dir, split_map)
    write_data_yaml()

    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")   # small model — downloads ~22MB automatically

    model.train(
        data=DATA_YAML,
        epochs=QUICK_EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        project=TRAIN_PROJECT,
        name=TRAIN_NAME,
        patience=5,
        save=True,
        device=DEVICE,
        workers=2,
        augment=True,
        cos_lr=True,
        plots=True,
        verbose=True,
    )

    trained_model = f"{TRAIN_PROJECT}/{TRAIN_NAME}/weights/best.pt"
    print(f"\n  Quick training complete!")
    print(f"  Model saved → {trained_model}")
    return trained_model


# ═════════════════════════════════════════════════════════════
# MODULE 3: QUICK FINE-TUNING
# Freezes backbone, trains detection head only
# ═════════════════════════════════════════════════════════════
def run_quick_finetuning(trained_model):
    print("\n── Module 3: Quick Fine-Tuning ──────────────────────────")

    if not os.path.exists(trained_model):
        print(f"  [SKIP] Model not found at {trained_model}")
        return trained_model

    print(f"  Freezing backbone, training detection head only...")
    print(f"  {FINETUNE_EPOCHS} epochs — Expected time: 1–4 minutes\n")

    from ultralytics import YOLO
    model = YOLO(trained_model)

    model.train(
        data=DATA_YAML,
        epochs=FINETUNE_EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        lr0=0.0001,
        lrf=0.01,
        freeze=10,
        project=FINETUNE_PROJECT,
        name=FINETUNE_NAME,
        patience=5,
        device=DEVICE,
        cos_lr=True,
        plots=True,
        verbose=True,
    )

    finetuned_model = f"{FINETUNE_PROJECT}/{FINETUNE_NAME}/weights/best.pt"
    print(f"\n  Fine-tuning complete!")
    print(f"  Model saved → {finetuned_model}")
    return finetuned_model


# ═════════════════════════════════════════════════════════════
# MODULE 4: MODEL EVALUATION
# ═════════════════════════════════════════════════════════════
def run_evaluation(model_path):
    print("\n── Module 4: Model Evaluation ───────────────────────────")

    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found at {model_path}")
        return

    os.makedirs(EVAL_DIR, exist_ok=True)

    from ultralytics import YOLO
    model   = YOLO(model_path)
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        plots=True,
        project=EVAL_DIR,
        name="eval",
        verbose=False,
    )

    results = {
        "model":     model_path,
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50-95":  round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),    4),
        "recall":    round(float(metrics.box.mr),    4),
        "note":      "Quick demo — trained on 200 images x 5 epochs. Full training = higher scores."
    }

    print("\n  ── Evaluation Results ───────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<12}: {v}")

    print("\n  ── Score Guide ──────────────────────────────────────")
    print("  mAP50 > 0.70  → Good for production")
    print("  mAP50 > 0.50  → Acceptable for demo")
    print("  mAP50 < 0.30  → Expected for 5-epoch quick train")
    print("  NOTE: Full 100-epoch training on all 6,734 images = much higher scores")

    with open(os.path.join(EVAL_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {EVAL_DIR}/summary.json")

    return results


# ═════════════════════════════════════════════════════════════
# MODULE 5: MODEL DEPLOYMENT
# Launches FastAPI inference server
# ═════════════════════════════════════════════════════════════
def run_deployment(model_path):
    print("\n── Module 5: Model Deployment ───────────────────────────")

    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found at {model_path}")
        return

    import uvicorn
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    from ultralytics import YOLO

    model = YOLO(model_path)
    app   = FastAPI(title="Food Waste Detection API")

    @app.get("/")
    def root():
        return {
            "status":  "running",
            "model":   model_path,
            "classes": 32,
            "docs":    "http://localhost:8000/docs"
        }

    @app.post("/detect")
    async def detect(file: UploadFile = File(...)):
        contents = await file.read()
        np_arr   = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image."})

        results     = model(img, conf=CONFIDENCE, verbose=False)[0]
        detections  = []
        waste_count = 0
        food_count  = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf     = float(box.conf[0])
            cls_id   = int(box.cls[0])
            cls_name = CLASSES[cls_id]
            is_waste = cls_name in WASTE_ITEMS

            if is_waste:
                waste_count += 1
            else:
                food_count += 1

            detections.append({
                "class":      cls_name,
                "confidence": round(conf, 3),
                "is_waste":   is_waste,
                "bbox":       [x1, y1, x2, y2]
            })

        total       = waste_count + food_count
        waste_ratio = round(waste_count / total, 2) if total > 0 else 0.0

        if waste_ratio >= 0.6:
            rec = "HIGH WASTE: Reduce portion sizes immediately."
        elif waste_ratio >= 0.3:
            rec = "MODERATE WASTE: Review menu items with most waste."
        else:
            rec = "LOW WASTE: Good waste management."

        return {
            "total_detections": len(detections),
            "food_count":       food_count,
            "waste_count":      waste_count,
            "waste_ratio":      waste_ratio,
            "recommendation":   rec,
            "detections":       detections
        }

    print("  API is live!")
    print("  Endpoint : http://localhost:8000/detect")
    print("  Live docs : http://localhost:8000/docs")
    print("  Press Ctrl+C to stop the server.\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Food Waste AI — Quick Train Pipeline")
    print("  Research → Train → Fine-Tune → Evaluate → Deploy")
    print("  Total expected time: 5–20 minutes")
    print("=" * 60)

    # Module 1 — research (instant)
    run_model_research()

    # Module 2 — quick training (~5–10 min)
    trained_model = run_quick_training()

    # Module 3 — fine-tuning (~2–5 min)
    finetuned_model = run_quick_finetuning(trained_model)

    # Module 4 — evaluation (~1 min)
    run_evaluation(finetuned_model)

    # Module 5 — deployment (runs until Ctrl+C)
    run_deployment(finetuned_model)
