# =======================================================================
# hyperspectral_app_sp_pi.py
# A full FastAPI service that preserves the NAIP NIR band, computes
# NDVI / NDBI / Solar-Panel Spectral Index (SPSI) and Solar Photovoltaic
# Panel Index (SPPI) from the recent research, filters vegetation
# false-positives, offers optional sharpening, test-time augmentation (TTA),
# and an optional multi-channel feature stack ready for custom multispectral
# YOLO training including SPPI.
# =======================================================================

import base64
import io
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ───────────────────────────────
# 1. FastAPI & CORS Boilerplate
# ───────────────────────────────
app = FastAPI(title="Hyperspectral Solar-Panel Detector with SPPI")

if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────
# 2. YOLO Model
# ───────────────────────────────
def download_model() -> str:
    repo_id = "finloop/yolov8s-seg-solar-panels"  # RGB-trained weights
    model_path = hf_hub_download(repo_id, filename="best.pt")
    return model_path

MODEL_PATH = download_model()
MODEL = YOLO(MODEL_PATH)  # expects 3-channel RGB

# ───────────────────────────────
# 3. Spectral Utilities & Indices
# ───────────────────────────────

# Color constants (BGR)
PANEL_COLOR = (60, 220, 60)
BOX_COLOR = (40, 180, 255)
TEXT_COLOR = (255, 255, 255)

def numpy_from_upload_multispectral(file_bytes: bytes) -> np.ndarray:
    """
    Keep all bands (e.g. NAIP 4-band RGB+NIR). If more than 4 bands are
    present, cut to first 4; YOLO uses RGB, but spectral indices still
    exploit NIR.
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image.")

    # Promote grayscale to 3-band BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Keep RGB+NIR
    if img.ndim == 3 and img.shape[2] >= 4:
        return img[:, :, :4]  # R,G,B,NIR
    return img

def compute_spectral_indices(img_4band: np.ndarray) -> dict:
    """
    Return NDVI, NDBI, SPSI and SPPI.
    Input must be 4-band NAIP order: R,G,B,NIR.
    """
    if img_4band.shape[2] < 4:
        raise ValueError("Need ≥4 bands to compute spectral indices.")

    red   = img_4band[:, :, 0].astype(np.float32)
    green = img_4band[:, :, 1].astype(np.float32)
    blue  = img_4band[:, :, 2].astype(np.float32)
    nir   = img_4band[:, :, 3].astype(np.float32)

    # NDVI: Vegetation index
    ndvi = (nir - red) / (nir + red + 1e-7)

    # NDBI: Built-up index (positive for built-up including solar panels)
    ndbi = (red - nir) / (red + nir + 1e-7)

    # SPSI: Simple Solar Panel Spectral Index (custom)
    spsi = (red / (nir + 1.0)) - (blue / (green + 1.0))

    # SPPI: Solar Photovoltaic Panel Index from the cited paper's principles
    # Placeholder formula adapting their spectral peak (400-500 nm in blue channel)
    # Here assumed as (blue / (nir + 1)) - (red / (green + 1)); adjust if band wavelengths known precisely
    sppi = (blue / (nir + 1.0)) - (red / (green + 1.0))

    return {"ndvi": ndvi, "ndbi": ndbi, "spsi": spsi, "sppi": sppi}

def create_feature_stack(img_4band: np.ndarray) -> np.ndarray:
    """
    Build a 8-channel tensor: R,G,B,NIR,NDVI,NDBI,SPSI,SPPI
    suitable for multispectral network training.
    """
    idx = compute_spectral_indices(img_4band)
    stack = np.stack(
        [
            img_4band[:, :, 0],      # R
            img_4band[:, :, 1],      # G
            img_4band[:, :, 2],      # B
            img_4band[:, :, 3],      # NIR
            idx["ndvi"],             # NDVI vegetation index
            idx["ndbi"],             # NDBI built-up index
            idx["spsi"],             # Solar Panel Spectral Index
            idx["sppi"],             # Solar Photovoltaic Panel Index (SPPI)
        ],
        axis=2,
    )
    # Optional: normalize each band (mean/std) before training
    return stack

def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    return buf.tobytes()

# ───────────────────────────────
# 4. Drawing & Post-processing
# ───────────────────────────────
def draw_detections(
    img_bgr: np.ndarray,
    polygons: List[np.ndarray],
    boxes: Optional[np.ndarray],
    confs: Optional[np.ndarray],
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = np.zeros_like(img_bgr)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        cv2.fillPoly(overlay, [poly.astype(np.int32)], PANEL_COLOR)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(blended, (x1, y1), (x2, y2), BOX_COLOR, 2)
            label = f"panel {confs[i]:.2f}" if confs is not None else "panel"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1t = max(0, y1 - th - 6)
            cv2.rectangle(blended, (x1, y1t), (x1 + tw + 6, y1), BOX_COLOR, -1)
            cv2.putText(
                blended,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
    return blended

def filter_vegetation_false_positives(polygons, boxes, ndvi, threshold=0.3):
    """
    Remove detections whose average NDVI is high (likely vegetation).
    """
    if boxes is None:
        return polygons, boxes
    keep_polys, keep_boxes = [], []
    for poly, box in zip(polygons, boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        roi_ndvi = ndvi[max(y1, 0) : y2, max(x1, 0) : x2]
        if roi_ndvi.size == 0:
            continue
        if np.mean(roi_ndvi) < threshold:
            keep_polys.append(poly)
            keep_boxes.append(box)
    return keep_polys, np.array(keep_boxes) if keep_boxes else None

def sharpen_image_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Apply sharpening to RGB image using a kernel."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(img_rgb, -1, kernel)
    return sharp

def yolo_infer_with_tta(model, img_rgb: np.ndarray, conf=0.25, iou=0.5, imgsz=1280):
    """
    Perform YOLO inference with test-time augmentation (TTA).
    Returns combined polygon masks, boxes, confs.
    """
    augmentations = [lambda x: x,
                     lambda x: cv2.flip(x, 1),  # horizontal flip
                     lambda x: cv2.flip(x, 0)]  # vertical flip

    all_polygons = []
    all_boxes = []
    all_confs = []

    for aug in augmentations:
        aug_img = aug(img_rgb)
        results = model(aug_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        r = results[0]

        # Reverse augmentation on output boxes and polygons
        if r.boxes is not None and r.boxes.xyxy is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            # Flip boxes back if needed
            if aug == augmentations[1]:  # horizontal flip
                boxes[:, [0, 2]] = img_rgb.shape[1] - boxes[:, [2, 0]]
            elif aug == augmentations[2]:  # vertical flip
                boxes[:, [1, 3]] = img_rgb.shape[0] - boxes[:, [3, 1]]
        else:
            boxes, confs = None, None

        polygons = []
        if r.masks is not None and hasattr(r.masks, "xy"):
            for poly in r.masks.xy:
                p = np.array(poly)
                # Flip polygons back
                if aug == augmentations[1]:
                    p[:, 0] = img_rgb.shape[1] - p[:, 0]
                elif aug == augmentations[2]:
                    p[:, 1] = img_rgb.shape[0] - p[:, 1]
                polygons.append(p)
        all_polygons.extend(polygons)
        if boxes is not None:
            all_boxes.extend(boxes)
        if confs is not None:
            all_confs.extend(confs)

    # Optionally, apply Non-Max Suppression or merge results here
    # For simplicity, returning concatenated detections
    all_boxes = np.array(all_boxes) if all_boxes else None
    all_confs = np.array(all_confs) if all_confs else None

    return all_polygons, all_boxes, all_confs

def yolo_infer_rgb(img_bgr: np.ndarray, conf=0.25, iou=0.5, imgsz=1280):
    """Run YOLO on RGB image and return result object."""
    results = MODEL(img_bgr[:, :, :3], conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    return results[0]

def polygons_from_masks(result) -> List[np.ndarray]:
    polys = []
    if result.masks is not None and hasattr(result.masks, "xy"):
        for poly in result.masks.xy:
            polys.append(np.array(poly))
    elif result.masks is not None and hasattr(result.masks, "data"):
        mdata = result.masks.data.cpu().numpy().astype(np.uint8)
        for m in mdata:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys.append(max(cnts, key=cv2.contourArea).reshape(-1, 2) if cnts else None)
    return polys

# ───────────────────────────────
# 6. FastAPI End-points
# ───────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")

@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.5),
    imgsz: int = Form(1280),
    ndvi_threshold: float = Form(0.3),          # vegetation filter
    sharpen: bool = Form(False),
    augment: bool = Form(False),
):
    """
    Generic image inference (PNG/JPG/TIFF). If 4-band image is supplied,
    NDVI vegetation filter is applied to suppress false positives.
    Supports optional sharpening and test-time augmentation.
    """
    try:
        img_bytes = await file.read()
        img = numpy_from_upload_multispectral(img_bytes)

        # Compute NDVI for false positive filtering if possible
        ndvi = compute_spectral_indices(img)["ndvi"] if img.shape[2] >= 4 else None

        # Prepare RGB image for inference
        img_rgb = img[:, :, :3]
        if sharpen:
            img_rgb = sharpen_image_rgb(img_rgb)

        if augment:
            polygons, boxes, confs = yolo_infer_with_tta(MODEL, img_rgb, conf, iou, imgsz)
        else:
            result = MODEL(img_rgb, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            polygons = polygons_from_masks(result)
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else None
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None

        # Filter detections in high NDVI areas (vegetation false positives)
        if ndvi is not None and boxes is not None:
            polygons, boxes = filter_vegetation_false_positives(polygons, boxes, ndvi, threshold=ndvi_threshold)
            if boxes is None or len(boxes) == 0:
                polygons, boxes, confs = [], None, None

        vis = draw_detections(img_rgb, polygons, boxes, confs)
        png = encode_png(vis)
        b64 = base64.b64encode(png).decode()
        count = sum(1 for p in polygons if p is not None)

        return JSONResponse({
            "count": count,
            "image_data_url": f"data:image/png;base64,{b64}",
            "used_sharpen": sharpen,
            "used_augment": augment,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ───────────────────────────────
# 7. Optional Endpoint: Return 8-channel Feature Stack (for training)
# ───────────────────────────────
@app.post("/extract_features")
async def extract_features(file: UploadFile = File(...)):
    """
    Upload a 4-band image → return a zipped .npz file containing:
    R, G, B, NIR, NDVI, NDBI, SPSI, and SPPI (8 bands).
    Handy for training a custom multispectral YOLO model.
    """
    try:
        img_bytes = await file.read()
        img = numpy_from_upload_multispectral(img_bytes)
        if img.shape[2] < 4:
            return JSONResponse({"error": "Need 4-band imagery to extract features."}, status_code=400)

        feat = create_feature_stack(img).astype(np.float32)
        # Save to an in-memory buffer
        buf = io.BytesIO()
        np.savez_compressed(buf, features=feat)
        buf.seek(0)
        return FileResponse(buf, media_type="application/npz", filename="features.npz")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ───────────────────────────────
# 8. Run via:  uvicorn hyperspectral_app_sp_pi:app --reload
# ───────────────────────────────
