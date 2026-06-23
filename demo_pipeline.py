"""
Food Waste AI System
FILE: demo_pipeline.py

Runs in under 10 minutes — no training needed.
Uses a pretrained YOLOv8m model to demonstrate the full pipeline.

HOW TO RUN:
  pip install ultralytics opencv-python numpy
  python demo_pipeline.py
"""

import cv2
import numpy as np
import os
import json
from collections import defaultdict
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
INPUT_DIR   = "./split_food_waste_dataset"   # your cleaned dataset
OUTPUT_DIR  = "./demo_output"
DEMO_LIMIT  = 20        # number of images to demo (keeps it fast)
CONFIDENCE  = 0.25
SPLITS      = ["train", "valid", "test"]

# YOLOv8m pretrained on COCO — downloads automatically (~50MB, 1 min)
# We map COCO food-related classes to our waste categories for the demo
MODEL_NAME  = "yolov8m.pt"

# COCO classes relevant to food waste detection
FOOD_COCO_CLASSES = {
    46: "Banana", 47: "Apple", 48: "Sandwich", 49: "Orange",
    50: "Broccoli", 51: "Carrot", 52: "Hot dog", 53: "Pizza",
    54: "Donut", 55: "Cake", 56: "Chair", 57: "Couch",
    58: "Potted plant", 59: "Bed", 60: "Dining table",
}

FOOD_IDS  = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
WASTE_IDS = set()  # in COCO demo, we flag low-confidence food as potential waste

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_model():
    print("\n── Step 1: Loading pretrained YOLOv8m ───────────────────")
    print("  Downloading model weights (~50MB) if not cached...")
    model = YOLO(MODEL_NAME)
    print("  Model loaded successfully!")
    return model


# ─────────────────────────────────────────────────────────────
# STEP 2: IMAGE ENHANCEMENT (fast version)
# ─────────────────────────────────────────────────────────────
def enhance_image(img):
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    lut = np.array([((i / 255.0) ** (1.0 / 1.2)) * 255 for i in range(256)], dtype=np.uint8)
    img = cv2.LUT(img, lut)
    return img


# ─────────────────────────────────────────────────────────────
# STEP 3: EDGE DETECTION (fast version)
# ─────────────────────────────────────────────────────────────
def detect_edges(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny   = cv2.Canny(blurred, 50, 150)
    sobelx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel   = cv2.normalize(np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    composite = cv2.normalize(
        0.6 * canny.astype(np.float32) + 0.4 * sobel.astype(np.float32),
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return composite


# ─────────────────────────────────────────────────────────────
# STEP 4: SEGMENTATION (fast K-Means only)
# ─────────────────────────────────────────────────────────────
def segment_image(img):
    pixels   = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img.shape)


# ─────────────────────────────────────────────────────────────
# STEP 5 & 6: DETECTION + RECOGNITION
# ─────────────────────────────────────────────────────────────
def detect_and_recognize(model, img, img_path):
    results    = model(img_path, conf=CONFIDENCE, verbose=False)[0]
    detections = []
    food_count  = 0
    waste_count = 0

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = results.names[cls_id]

        # Flag low-confidence detections as potential waste for demo
        is_waste = conf < 0.4 and cls_id in FOOD_IDS

        if is_waste:
            waste_count += 1
            color = (0, 0, 255)   # red for waste
            label = f"WASTE:{cls_name} {conf:.2f}"
        else:
            food_count += 1
            color = (0, 200, 0)   # green for food
            label = f"{cls_name} {conf:.2f}"

        detections.append({
            "class":      cls_name,
            "confidence": round(conf, 3),
            "is_waste":   is_waste,
            "bbox":       [x1, y1, x2, y2]
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    total       = food_count + waste_count
    waste_ratio = round(waste_count / total, 2) if total > 0 else 0.0

    if waste_ratio >= 0.6:
        recommendation = "HIGH WASTE — Reduce portion sizes"
    elif waste_ratio >= 0.3:
        recommendation = "MODERATE WASTE — Review menu items"
    else:
        recommendation = "LOW WASTE — Good management"

    # Draw summary on image
    summary_text = f"Food: {food_count}  Waste: {waste_count}  Ratio: {waste_ratio}"
    cv2.rectangle(img, (0, 0), (640, 28), (20, 20, 20), -1)
    cv2.putText(img, summary_text, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return img, {
        "food_count":     food_count,
        "waste_count":    waste_count,
        "waste_ratio":    waste_ratio,
        "recommendation": recommendation,
        "detections":     detections
    }


# ─────────────────────────────────────────────────────────────
# SAVE DEMO RESULTS — side-by-side comparison image
# ─────────────────────────────────────────────────────────────
def save_comparison(original, enhanced, edges, segmented, detected, out_path):
    # Convert grayscale edge map to BGR for stacking
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    top    = np.hstack([original, enhanced])
    bottom = np.hstack([edges_bgr, segmented])
    middle = np.hstack([detected, detected])   # detection shown large

    # Add labels
    for i, (row, label) in enumerate([(top, ["Original", "Enhanced"]),
                                       (bottom, ["Edge Detection", "Segmentation"])]):
        for j, text in enumerate(label):
            cv2.putText(row, text, (j * 640 + 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    combined = np.vstack([top, bottom])
    cv2.imwrite(out_path, combined)


# ─────────────────────────────────────────────────────────────
# MAIN DEMO RUNNER
# ─────────────────────────────────────────────────────────────
def run_demo():
    print("=" * 60)
    print("  Food Waste AI System — DEMO")
    print(f"  Processing {DEMO_LIMIT} images per split")
    print("=" * 60)

    model = load_model()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for split in SPLITS:
        img_dir = os.path.join(INPUT_DIR, split, "images")
        if not os.path.exists(img_dir):
            print(f"\n  [SKIP] {img_dir} not found.")
            continue

        out_split = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_split, exist_ok=True)

        files = [f for f in sorted(os.listdir(img_dir))
                 if not f.startswith(".")][:DEMO_LIMIT]

        print(f"\n── Processing {split} ({len(files)} images) ─────────────")

        split_results = []

        for i, fname in enumerate(files):
            img_path = os.path.join(img_dir, fname)
            original = cv2.imread(img_path)
            if original is None:
                continue

            original  = cv2.resize(original, (640, 640))
            enhanced  = enhance_image(original.copy())
            edges     = detect_edges(enhanced)
            segmented = segment_image(enhanced)
            detected, result = detect_and_recognize(model, enhanced.copy(), img_path)

            # Save comparison image
            stem     = os.path.splitext(fname)[0]
            comp_path = os.path.join(out_split, f"{stem}_demo.jpg")
            save_comparison(original, enhanced, edges, segmented, detected, comp_path)

            result["file"]  = fname
            result["split"] = split
            split_results.append(result)

            print(f"  [{i+1}/{len(files)}] {fname} — "
                  f"waste ratio: {result['waste_ratio']} — {result['recommendation']}")

        all_results.extend(split_results)

        # Split summary
        if split_results:
            avg_waste = round(sum(r["waste_ratio"] for r in split_results) / len(split_results), 2)
            print(f"\n  [{split}] Average waste ratio: {avg_waste}")

    # Save full JSON report
    report_path = os.path.join(OUTPUT_DIR, "demo_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print(f"  Output images  → {OUTPUT_DIR}/")
    print(f"  Full report    → {report_path}")
    print("\n  Each output image shows a 4-panel comparison:")
    print("    Top left:     Original image")
    print("    Top right:    Enhanced image")
    print("    Bottom left:  Edge detection")
    print("    Bottom right: Segmentation")
    print("    Detections overlaid with food/waste labels")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
