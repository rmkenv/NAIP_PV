# =======================================================================
# hyperspectral_app_sp_pi.py
# A FastAPI service for solar panel detection with NAIP (default) or
# Maryland Six Inch high-res aerial imagery. Preserves multispectral bands,
# computes NDVI/NDBI/SPSI/SPPI indices, optional sharpening and TTA,
# and supports high-resolution image inference and training feature extraction.
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

app = FastAPI(title="Hyperspectral Solar-Panel Detector with SPPI and High-Res MD")

if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_model() -> str:
    repo_id = "finloop/yolov8s-seg-solar-panels"
    model_path = hf_hub_download(repo_id, filename="best.pt")
    return model_path

MODEL_PATH = download_model()
MODEL = YOLO(MODEL_PATH)

# Utility Constants
PANEL_COLOR = (60, 220, 60)
BOX_COLOR = (40, 180, 255)
TEXT_COLOR = (255, 255, 255)
R_MAJOR = 6378137.0  # For Mercator projection

# ------------ Multispectral Band Processing -----------------
def numpy_from_upload_multispectral(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] >= 4:
        return img[:, :, :4]
    return img

def compute_spectral_indices(img_4band: np.ndarray) -> dict:
    if img_4band.shape[2] < 4:
        raise ValueError("Need ≥4 bands to compute spectral indices.")
    red = img_4band[:, :, 0].astype(np.float32)
    green = img_4band[:, :, 1].astype(np.float32)
    blue = img_4band[:, :, 2].astype(np.float32)
    nir = img_4band[:, :, 3].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-7)
    ndbi = (red - nir) / (red + nir + 1e-7)
    spsi = (red / (nir + 1.0)) - (blue / (green + 1.0))
    sppi = (blue / (nir + 1.0)) - (red / (green + 1.0))
    return {"ndvi": ndvi, "ndbi": ndbi, "spsi": spsi, "sppi": sppi}

def create_feature_stack(img_4band: np.ndarray) -> np.ndarray:
    idx = compute_spectral_indices(img_4band)
    stack = np.stack([
        img_4band[:, :, 0], img_4band[:, :, 1], img_4band[:, :, 2], img_4band[:, :, 3],
        idx["ndvi"], idx["ndbi"], idx["spsi"], idx["sppi"]
    ], axis=2)
    return stack

def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    return buf.tobytes()

# --------------- Geometry Utilities -----------------
def lonlat_to_webmerc(lon: float, lat: float) -> Tuple[float, float]:
    x = R_MAJOR * np.deg2rad(lon)
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = R_MAJOR * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return x, y

def bbox4326_to_3857(min_lon, min_lat, max_lon, max_lat):
    x1, y1 = lonlat_to_webmerc(min_lon, min_lat)
    x2, y2 = lonlat_to_webmerc(max_lon, max_lat)
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    return xmin, ymin, xmax, ymax

def webmerc_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    lon = np.rad2deg(x / R_MAJOR)
    lat = np.rad2deg(2 * np.arctan(np.exp(y / R_MAJOR)) - np.pi / 2)
    return lon, lat

# --------------- Drawing/Detection Utils -----------------
def draw_detections(img_bgr, polygons, boxes, confs, alpha=0.35):
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
            cv2.putText(blended, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    return blended

def filter_vegetation_false_positives(polygons, boxes, ndvi, threshold=0.3):
    if boxes is None:
        return polygons, boxes
    keep_polys, keep_boxes = [], []
    for poly, box in zip(polygons, boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        roi_ndvi = ndvi[max(y1, 0):y2, max(x1, 0):x2]
        if roi_ndvi.size == 0:
            continue
        if np.mean(roi_ndvi) < threshold:
            keep_polys.append(poly)
            keep_boxes.append(box)
    return keep_polys, np.array(keep_boxes) if keep_boxes else None

def sharpen_image_rgb(img_rgb: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img_rgb, -1, kernel)

def yolo_infer_with_tta(model, img_rgb, conf=0.25, iou=0.5, imgsz=1280):
    augmentations = [lambda x: x, lambda x: cv2.flip(x, 1), lambda x: cv2.flip(x, 0)]
    all_polygons = []
    all_boxes = []
    all_confs = []
    for aug in augmentations:
        aug_img = aug(img_rgb)
        results = model(aug_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        r = results[0]
        if r.boxes is not None and r.boxes.xyxy is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            if aug == augmentations[1]:
                boxes[:, [0, 2]] = img_rgb.shape[1] - boxes[:, [2, 0]]
            elif aug == augmentations[2]:
                boxes[:, [1, 3]] = img_rgb.shape[0] - boxes[:, [3, 1]]
        else:
            boxes, confs = None, None
        polygons = []
        if r.masks is not None and hasattr(r.masks, "xy"):
            for poly in r.masks.xy:
                p = np.array(poly)
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
    all_boxes = np.array(all_boxes) if all_boxes else None
    all_confs = np.array(all_confs) if all_confs else None
    return all_polygons, all_boxes, all_confs

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

# ------------ Imagery Fetchers -----------------
def fetch_naip_bbox_mercator(xmin, ymin, xmax, ymax, width_px=1536):
    url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
    w_m, h_m = xmax - xmin, ymax - ymin
    width_px = int(max(256, min(3000, width_px)))
    aspect = h_m / w_m
    height_px = int(round(width_px * aspect))
    height_px = max(256, min(3000, height_px))
    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857, "imageSR": 3857,
        "size": f"{width_px},{height_px}",
        "format": "png",
    }
    resp = requests.get(url, params=params, timeout=30)
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("NAIP image decoding failed")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 4:
        img = img[:, :, :4]
    mpp_x = w_m / img.shape[1]
    mpp_y = h_m / img.shape[0]
    return img, mpp_x, mpp_y

def fetch_md_sixinch_bbox_mercator(xmin, ymin, xmax, ymax, width_px=1536):
    url = "https://mdgeodata.md.gov/imagery/rest/services/SixInch/SixInchImagery/ImageServer/exportImage"
    w_m = xmax - xmin
    h_m = ymax - ymin
    width_px = int(max(256, min(3500, width_px)))
    aspect = h_m / w_m
    height_px = int(round(width_px * aspect))
    height_px = max(256, min(3500, height_px))
    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857, "imageSR": 3857,
        "size": f"{width_px},{height_px}",
        "format": "png",
    }
    resp = requests.get(url, params=params, timeout=30)
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        raise RuntimeError("Maryland Six Inch image decoding failed")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 4:
        img = img[:, :, :4]
    mpp_x = w_m / img.shape[1]
    mpp_y = h_m / img.shape[0]
    return img, mpp_x, mpp_y

def box_center_to_latlon(box, bbox_3857, img_shape):
    xmin, ymin, xmax, ymax = bbox_3857
    pixel_x_center = (box[0] + box[2]) / 2
    pixel_y_center = (box[1] + box[3]) / 2
    img_width = img_shape[1]
    img_height = img_shape[0]
    merc_x = xmin + (xmax - xmin) * (pixel_x_center / img_width)
    merc_y = ymax - (ymax - ymin) * (pixel_y_center / img_height)
    lon, lat = webmerc_to_lonlat(merc_x, merc_y)
    return lat, lon

# ------------ API Routes -----------------
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")

@app.post("/infer_imagery")
async def infer_imagery(
    service: str = Form("naip"),  # "naip" or "md"
    min_lon: float = Form(...),
    min_lat: float = Form(...),
    max_lon: float = Form(...),
    max_lat: float = Form(...),
    width_px: int = Form(1536),
    conf: float = Form(0.25),
    iou: float = Form(0.5),
    imgsz: int = Form(1280),
    sharpen: bool = Form(False),
    augment: bool = Form(False),
    ndvi_threshold: float = Form(0.3),
):
    try:
        xmin, ymin, xmax, ymax = bbox4326_to_3857(min_lon, min_lat, max_lon, max_lat)
        if service == "md":
            img, mpp_x, mpp_y = fetch_md_sixinch_bbox_mercator(xmin, ymin, xmax, ymax, width_px)
            imagery_source = "Maryland Six Inch"
        else:
            img, mpp_x, mpp_y = fetch_naip_bbox_mercator(xmin, ymin, xmax, ymax, width_px)
            imagery_source = "NAIP"
        ndvi = compute_spectral_indices(img)["ndvi"] if img.shape[2] >= 4 else None
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
        if ndvi is not None and boxes is not None:
            polygons, boxes = filter_vegetation_false_positives(polygons, boxes, ndvi, threshold=ndvi_threshold)
            if boxes is None or len(boxes) == 0:
                polygons, boxes, confs = [], None, None
        vis = draw_detections(img_rgb, polygons, boxes, confs)
        png = encode_png(vis)
        b64 = base64.b64encode(png).decode()
        # Add detection list
        detection_rows = []
        if boxes is not None and confs is not None:
            for i, box in enumerate(boxes):
                lat, lon = box_center_to_latlon(box, (xmin, ymin, xmax, ymax), img.shape)
                score = float(confs[i])
                detection_rows.append({"lat": lat, "lon": lon, "score": score})
        return JSONResponse({
            "count": sum(1 for p in polygons if p is not None),
            "image_data_url": f"data:image/png;base64,{b64}",
            "used_sharpen": sharpen,
            "used_augment": augment,
            "imagery_source": imagery_source,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "export_bbox_3857": [xmin, ymin, xmax, ymax],
            "detections": detection_rows,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/extract_features")
async def extract_features(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = numpy_from_upload_multispectral(img_bytes)
        if img.shape[2] < 4:
            return JSONResponse({"error": "Need 4-band imagery to extract features."}, status_code=400)
        feat = create_feature_stack(img).astype(np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, features=feat)
        buf.seek(0)
        return FileResponse(buf, media_type="application/npz", filename="features.npz")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# Usage: uvicorn hyperspectral_app_sp_pi:app --reload
