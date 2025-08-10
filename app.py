# app.py
import base64
import io
import os
from typing import Optional, List, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

app = FastAPI(title="Solar Panel Detector (NAIP)")

# Serve static files (index.html)
if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def download_model() -> str:
    # YOLOv8 segmentation weights for solar panels (MIT):
    # https://huggingface.co/finloop/yolov8s-seg-solar-panels
    repo_id = "finloop/yolov8s-seg-solar-panels"
    filename = "best.pt"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return model_path


MODEL_PATH = download_model()
MODEL = YOLO(MODEL_PATH)

PANEL_COLOR = (60, 220, 60)   # BGR green
BOX_COLOR = (40, 180, 255)    # BGR orange
TEXT_COLOR = (255, 255, 255)


def numpy_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid PNG/JPG.")
    # Some NAIP PNGs have 4 channels (RGB + NIR or alpha). Drop to 3-channel.
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode result image.")
    return buf.tobytes()


def draw_detections(
    img_bgr: np.ndarray,
    polygons: List[np.ndarray],
    boxes: Optional[np.ndarray],
    confs: Optional[np.ndarray],
    alpha: float = 0.35
) -> np.ndarray:
    overlay = np.zeros_like(img_bgr)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        p = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [p], PANEL_COLOR)
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


def polygon_areas_in_pixels(polygons: List[np.ndarray]) -> List[float]:
    areas = []
    for poly in polygons:
        if poly is None or len(poly) < 3:
            areas.append(0.0)
            continue
        p = np.array(poly, dtype=np.float32)
        areas.append(float(cv2.contourArea(p)))
    return areas


# --- Web Mercator helpers (EPSG:3857) ---
R_MAJOR = 6378137.0

def lonlat_to_webmerc(lon: float, lat: float) -> Tuple[float, float]:
    x = R_MAJOR * np.deg2rad(lon)
    # clamp latitude to Web Mercator valid range
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = R_MAJOR * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return float(x), float(y)

def bbox4326_to_3857(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Tuple[float, float, float, float]:
    x1, y1 = lonlat_to_webmerc(min_lon, min_lat)
    x2, y2 = lonlat_to_webmerc(max_lon, max_lat)
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    return xmin, ymin, xmax, ymax


# --- NAIP fetch via ArcGIS ImageServer Export Image ---
# You can switch between USGS and USDA ImageServers. Both are NAIP mosaics.
USGS_NAIP_EXPORT = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
USDA_NAIP_EXPORT = "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage"

def fetch_naip_bbox_mercator(
    xmin: float, ymin: float, xmax: float, ymax: float,
    width_px: int = 1536,
    use_usgs: bool = True
) -> Tuple[np.ndarray, float, float]:
    """
    Fetch a NAIP image for a given EPSG:3857 bbox at a requested pixel width.
    Returns (image_bgr, mpp_x, mpp_y) where mpp is meters per pixel.
    """
    # Keep size within service limits (MaxImageWidth/Height ~4000)
    width_px = int(max(256, min(3000, width_px)))
    w_m = xmax - xmin
    h_m = ymax - ymin
    if w_m <= 0 or h_m <= 0:
        raise ValueError("Invalid bbox.")
    aspect = h_m / w_m
    height_px = int(round(width_px * aspect))
    height_px = max(256, min(3000, height_px))

    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{width_px},{height_px}",
        "format": "png",
    }
    url = USGS_NAIP_EXPORT if use_usgs else USDA_NAIP_EXPORT
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"ExportImage failed: HTTP {resp.status_code}")
    # The service returns image bytes when f=image. If an error, it often returns small HTML/JSON.
    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type:
        # try to surface error message
        text = resp.text[:500]
        raise RuntimeError(f"ExportImage returned non-image response: {text}")

    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode image from ImageServer.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    # meters per pixel in x and y
    mpp_x = w_m / float(img.shape[1])
    mpp_y = h_m / float(img.shape[0])
    return img, mpp_x, mpp_y


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    meters_per_pixel: Optional[float] = Form(default=None),
    conf: float = Form(default=0.25),
    iou: float = Form(default=0.5),
    imgsz: int = Form(default=1280)
):
    try:
        data = await file.read()
        img = numpy_from_upload(data)

        results = MODEL(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        result = results[0]

        polygons: List[np.ndarray] = []
        if result.masks is not None and hasattr(result.masks, "xy"):
            for poly in result.masks.xy:
                polygons.append(np.array(poly))
        elif result.masks is not None and hasattr(result.masks, "data"):
            mdata = result.masks.data.cpu().numpy().astype(np.uint8)
            for m in mdata:
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    polygons.append(None)
                else:
                    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
                    polygons.append(cnt)

        boxes = None
        confs = None
        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            if hasattr(result.boxes, "conf") and result.boxes.conf is not None:
                confs = result.boxes.conf.cpu().numpy()

        vis = draw_detections(result.orig_img if hasattr(result, "orig_img") else img, polygons, boxes, confs)

        areas_px = polygon_areas_in_pixels(polygons)
        total_area_px = float(np.sum(areas_px))
        mpp = float(meters_per_pixel) if meters_per_pixel is not None else None
        total_area_m2 = (total_area_px * (mpp ** 2)) if mpp is not None else None
        count = int(len([a for a in areas_px if a > 0]))

        if count == 0:
            cv2.putText(vis, "No panels detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        png_bytes = encode_png(vis)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        return JSONResponse({
            "count": count,
            "total_area_pixels": total_area_px,
            "meters_per_pixel": mpp,
            "total_area_m2": total_area_m2,
            "image_data_url": data_url
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/infer_naip")
async def infer_naip(
    min_lon: float = Form(...),
    min_lat: float = Form(...),
    max_lon: float = Form(...),
    max_lat: float = Form(...),
    width_px: int = Form(1024),
    use_usgs: bool = Form(True),  # True=USGS, False=USDA
    conf: float = Form(0.25),
    iou: float = Form(0.5),
    imgsz: int = Form(1280)
):
    """
    Fetch NAIP via ArcGIS ImageServer for the given lon/lat bbox, then run detection.
    """
    try:
        xmin, ymin, xmax, ymax = bbox4326_to_3857(min_lon, min_lat, max_lon, max_lat)
        img, mpp_x, mpp_y = fetch_naip_bbox_mercator(xmin, ymin, xmax, ymax, width_px=width_px, use_usgs=use_usgs)

        # Run model
        results = MODEL(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        result = results[0]

        # Extract polygons and boxes
        polygons: List[np.ndarray] = []
        if result.masks is not None and hasattr(result.masks, "xy"):
            for poly in result.masks.xy:
                polygons.append(np.array(poly))
        elif result.masks is not None and hasattr(result.masks, "data"):
            mdata = result.masks.data.cpu().numpy().astype(np.uint8)
            for m in mdata:
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    polygons.append(None)
                else:
                    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
                    polygons.append(cnt)

        boxes = None
        confs = None
        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            if hasattr(result.boxes, "conf") and result.boxes.conf is not None:
                confs = result.boxes.conf.cpu().numpy()

        vis = draw_detections(result.orig_img if hasattr(result, "orig_img") else img, polygons, boxes, confs)

        # Areas
        areas_px = polygon_areas_in_pixels(polygons)
        total_area_px = float(np.sum(areas_px))
        # Area conversion uses mpp_x * mpp_y to account for non-square pixels if any
        total_area_m2 = total_area_px * (mpp_x * mpp_y)
        count = int(len([a for a in areas_px if a > 0]))

        if count == 0:
            cv2.putText(vis, "No panels detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        png_bytes = encode_png(vis)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        return JSONResponse({
            "count": count,
            "total_area_pixels": total_area_px,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "total_area_m2": total_area_m2,
            "export_bbox_3857": [xmin, ymin, xmax, ymax],
            "image_data_url": data_url
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
