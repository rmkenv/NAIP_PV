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



# --- Generic imagery fetchers (ImageServer, MapServer, WMS) ---

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def compute_size_from_target_mpp(xmin, ymin, xmax, ymax, target_mpp: Optional[float], max_dim: int = 3000) -> Tuple[int, int, float, float]:
    w_m = xmax - xmin
    h_m = ymax - ymin
    if w_m <= 0 or h_m <= 0:
        raise ValueError("Invalid bbox.")
    if target_mpp and target_mpp > 0:
        width_px = int(clamp(round(w_m / target_mpp), 256, max_dim))
        height_px = int(clamp(round(h_m / target_mpp), 256, max_dim))
    else:
        # fallback size similar to NAIP endpoint
        width_px = 1536
        aspect = h_m / w_m
        height_px = int(clamp(round(width_px * aspect), 256, max_dim))
    mpp_x = w_m / float(width_px)
    mpp_y = h_m / float(height_px)
    return width_px, height_px, mpp_x, mpp_y


def strip_query(url: str) -> str:
    return url.split("?")[0]


def try_pick_wms_layer_from_capabilities(base_url: str) -> Optional[str]:
    try:
        caps_url = f"{base_url}?service=WMS&request=GetCapabilities&version=1.3.0"
        r = requests.get(caps_url, timeout=20)
        if r.status_code != 200:
            return None
        text = r.text
        # naive parse: find first <Layer>...<Name>... element that isn't the root layer
        # Prefer the first child Named Layer
        import re
        # find all <Name>...</Name>
        names = re.findall(r"<Name>([^<]+)</Name>", text)
        # Usually first name is the root; pick the second if available
        if not names:
            return None
        if len(names) >= 2:
            return names[1]
        return names[0]
    except Exception:
        return None


def fetch_imageserver(url: str, xmin: float, ymin: float, xmax: float, ymax: float, target_mpp: Optional[float]) -> Tuple[np.ndarray, float, float]:
    url = strip_query(url)
    if not url.lower().endswith("/exportimage"):
        if url.lower().endswith("/imageserver"):
            url = url + "/exportImage"
        else:
            # not an imageserver
            raise RuntimeError("Not an ImageServer URL")
    width_px, height_px, mpp_x, mpp_y = compute_size_from_target_mpp(xmin, ymin, xmax, ymax, target_mpp)
    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{width_px},{height_px}",
        "format": "png",
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"ImageServer exportImage failed: HTTP {resp.status_code}")
    if "image" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(f"ImageServer non-image response: {resp.text[:300]}")
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode ImageServer image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return img, mpp_x, mpp_y


def fetch_mapserver(url: str, xmin: float, ymin: float, xmax: float, ymax: float, target_mpp: Optional[float]) -> Tuple[np.ndarray, float, float]:
    # Use export on MapServer
    base = strip_query(url)
    if not base.lower().endswith("/export"):
        if base.lower().endswith("/mapserver"):
            base = base + "/export"
        else:
            raise RuntimeError("Not a MapServer URL")
    width_px, height_px, mpp_x, mpp_y = compute_size_from_target_mpp(xmin, ymin, xmax, ymax, target_mpp)
    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{width_px},{height_px}",
        "format": "png",
    }
    resp = requests.get(base, params=params, timeout=30)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"MapServer export failed: HTTP {resp.status_code}")
    if "image" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(f"MapServer non-image response: {resp.text[:300]}")
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode MapServer image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return img, mpp_x, mpp_y


def fetch_wms(url: str, xmin: float, ymin: float, xmax: float, ymax: float, target_mpp: Optional[float], layer: Optional[str]) -> Tuple[np.ndarray, float, float]:
    base = strip_query(url)
    width_px, height_px, mpp_x, mpp_y = compute_size_from_target_mpp(xmin, ymin, xmax, ymax, target_mpp)
    lyr = layer
    if not lyr:
        lyr = try_pick_wms_layer_from_capabilities(base)
        if not lyr:
            raise RuntimeError("WMS layer not specified and auto-pick failed. Provide a LAYERS name.")
    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": lyr,
        "styles": "",
        "crs": "EPSG:3857",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "width": str(width_px),
        "height": str(height_px),
        "format": "image/png",
        "transparent": "false",
    }
    resp = requests.get(base, params=params, timeout=40)
    if resp.status_code != 200 or not resp.content:

@app.post("/infer_generic_imagery")
async def infer_generic_imagery(
    service_url: str = Form(...),
    min_lon: float = Form(...),
    min_lat: float = Form(...),
    max_lon: float = Form(...),
    max_lat: float = Form(...),
    wms_layer: Optional[str] = Form(default=None),
    target_mpp: Optional[float] = Form(default=None),
    conf: float = Form(default=0.25),
    iou: float = Form(default=0.5),
    imgsz: int = Form(default=1280)
):
    try:
        img, mpp_x, mpp_y = fetch_generic_imagery(service_url, min_lon, min_lat, max_lon, max_lat, target_mpp, wms_layer)
        # Run model
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
            "image_data_url": data_url
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

        raise RuntimeError(f"WMS GetMap failed: HTTP {resp.status_code}")
    ctype = resp.headers.get("Content-Type", "").lower()
    if "image" not in ctype:
        # Many WMS return XML exception reports
        raise RuntimeError(f"WMS GetMap returned non-image: {resp.text[:400]}")
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode WMS image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return img, mpp_x, mpp_y


def fetch_generic_imagery(service_url: str, min_lon: float, min_lat: float, max_lon: float, max_lat: float, target_mpp: Optional[float], wms_layer: Optional[str]) -> Tuple[np.ndarray, float, float]:
    xmin, ymin, xmax, ymax = bbox4326_to_3857(min_lon, min_lat, max_lon, max_lat)
    low = service_url.lower()
    if "imageserver" in low:
        return fetch_imageserver(service_url, xmin, ymin, xmax, ymax, target_mpp)
    if "mapserver" in low and "wms" not in low:
        return fetch_mapserver(service_url, xmin, ymin, xmax, ymax, target_mpp)
    if "wms" in low:
        return fetch_wms(service_url, xmin, ymin, xmax, ymax, target_mpp, wms_layer)
    raise RuntimeError("Unsupported service_url. Must contain ImageServer, MapServer, or WMS")

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
