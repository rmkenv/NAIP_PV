# =======================================================================
# hyperspectral_app.py
# A full FastAPI service that preserves the NAIP NIR band, computes
# NDVI / NDBI / Solar-Panel Spectral Index (SPSI), filters vegetation
# false-positives, and offers an optional multi-channel feature stack
# ready for custom 4-channel (RGB+NIR) or 7-channel YOLO training.
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
app = FastAPI(title="Hyperspectral Solar-Panel Detector")

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
    repo_id = "finloop/yolov8s-seg-solar-panels"          # RGB-trained weights
    model_path = hf_hub_download(repo_id, filename="best.pt")
    return model_path

MODEL_PATH = download_model()
MODEL = YOLO(MODEL_PATH)                                  # expects 3-channel RGB

# ───────────────────────────────
# 3. Spectral Utilities
# ───────────────────────────────
# Color constants (BGR)
PANEL_COLOR = (60, 220, 60)
BOX_COLOR   = (40, 180, 255)
TEXT_COLOR  = (255, 255, 255)

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
    Return NDVI, NDBI and Solar-Panel Spectral Index (SPSI).
    Input must be 4-band NAIP order: R,G,B,NIR.
    """
    if img_4band.shape[2] < 4:
        raise ValueError("Need ≥4 bands to compute spectral indices.")

    red   = img_4band[:, :, 0].astype(np.float32)
    green = img_4band[:, :, 1].astype(np.float32)
    blue  = img_4band[:, :, 2].astype(np.float32)
    nir   = img_4band[:, :, 3].astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-7)
    ndbi = (red - nir) / (red + nir + 1e-7)
    spsi = (red / (nir + 1.0)) - (blue / (green + 1.0))      # simple SPSI

    return {"ndvi": ndvi, "ndbi": ndbi, "spsi": spsi}

def create_feature_stack(img_4band: np.ndarray) -> np.ndarray:
    """
    Build a 7-channel tensor: R,G,B,NIR,NDVI,NDBI,SPSI
    suitable for 7-channel network training.
    """
    idx = compute_spectral_indices(img_4band)
    stack = np.stack(
        [
            img_4band[:, :, 0],            # R
            img_4band[:, :, 1],            # G
            img_4band[:, :, 2],            # B
            img_4band[:, :, 3],            # NIR
            idx["ndvi"],
            idx["ndbi"],
            idx["spsi"],
        ],
        axis=2,
    )
    # Optional: normalize each band (mean/std) here.
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

# ───────────────────────────────
# 5. Core Inference Routine
# ───────────────────────────────
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
):
    """
    Generic image inference (PNG/JPG/TIFF). If 4-band image is supplied,
    NDVI vegetation filter is applied to suppress false positives.
    """
    try:
        img_bytes = await file.read()
        img = numpy_from_upload_multispectral(img_bytes)

        # 4-band → compute NDVI for FP filtering
        ndvi = compute_spectral_indices(img)["ndvi"] if img.shape[2] >= 4 else None

        # NOTE: model still runs on RGB (first 3 channels)
        result = yolo_infer_rgb(img, conf, iou, imgsz)

        polygons = polygons_from_masks(result)
        boxes    = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else None
        confs    = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None

        # Vegetation filter
        if ndvi is not None and boxes is not None:
            polygons, boxes = filter_vegetation_false_positives(
                polygons, boxes, ndvi, threshold=ndvi_threshold
            )
            if boxes is None or len(boxes) == 0:
                polygons, boxes, confs = [], None, None

        vis   = draw_detections(img[:, :, :3], polygons, boxes, confs)
        png   = encode_png(vis)
        b64   = base64.b64encode(png).decode()
        count = sum(1 for p in polygons if p is not None)

        return JSONResponse(
            {
                "count": count,
                "image_data_url": f"data:image/png;base64,{b64}",
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ───────────────────────────────
# 7. Optional Endpoint: Return 7-channel Feature Stack (for training)
# ───────────────────────────────
@app.post("/extract_features")
async def extract_features(file: UploadFile = File(...)):
    """
    Upload a 4-band image → return a zipped .npz file containing:
    R, G, B, NIR, NDVI, NDBI, SPSI (7 bands).  Handy for training a
    custom 7-channel YOLO model.
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
# 8. Run via:  uvicorn hyperspectral_app:app --reload
# ───────────────────────────────
