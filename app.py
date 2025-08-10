# app.py
import base64
import io
import os
from typing import Optional, List, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# --------------------------------------
# App setup
# --------------------------------------
app = FastAPI(title="Solar Panel Detector (Imagery-Agnostic)")

# Static
if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS for local dev and external UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------
# Model
# --------------------------------------

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

# --------------------------------------
# Utilities
# --------------------------------------

def numpy_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid PNG/JPG.")
    # Some PNGs may have 4 channels (RGB + alpha/NIR). Drop to 3-channel RGB-like.
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
    alpha: float = 0.35,
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


def webmerc_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    lon = np.rad2deg(x / R_MAJOR)
    lat = np.rad2deg(2 * np.arctan(np.exp(y / R_MAJOR)) - np.pi / 2)
    return float(lon), float(lat)

# --------------------------------------
# NAIP via ArcGIS ImageServer (existing flow)
# --------------------------------------
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
    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type:
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

    mpp_x = w_m / float(img.shape[1])
    mpp_y = h_m / float(img.shape[0])
    return img, mpp_x, mpp_y

# --------------------------------------
# Generic imagery ingestion (ImageServer / MapServer / WMS)
# --------------------------------------

def sanitize_service_url(url: str) -> str:
    u = url.strip()
    # Remove query strings for base endpoints
    if "?" in u:
        u = u.split("?", 1)[0]
    # Common trailing paths
    lowers = u.lower()
    if "imageserver" in lowers:
        return u[:lowers.rfind("imageserver")] + u[lowers.rfind("imageserver"):]
    if "mapserver" in lowers:
        return u[:lowers.rfind("mapserver")] + u[lowers.rfind("mapserver"):]
    if "wmss" in lowers:
        return u[:lowers.find("wmss")] + "WMSServer"
    if u.lower().endswith("wmssrver"):
        return u[:-1] + "er"  # fix typo case
    if "wmsserver" in lowers:
        return u[:lowers.rfind("wmsserver")] + u[lowers.rfind("wmsserver"):]
    return u


def export_image_from_imageserver(service_url: str, xmin: float, ymin: float, xmax: float, ymax: float,
                                  target_mpp: Optional[float] = None, max_px: int = 3000) -> Tuple[np.ndarray, float, float]:
    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "format": "png",
        "transparent": "false",
    }
    w_m = xmax - xmin
    h_m = ymax - ymin
    if target_mpp:
        params["pixelSize"] = f"{target_mpp},{target_mpp}"
    else:
        aspect = h_m / w_m
        width_px = max(256, min(max_px, int(round(w_m / 0.3))))  # aim for ~30 cm if unknown
        height_px = max(256, min(max_px, int(round(width_px * aspect))))
        params["size"] = f"{width_px},{height_px}"

    u = sanitize_service_url(service_url).rstrip("/")
    if not u.lower().endswith("/exportimage"):
        if u.lower().endswith("/imageserver"):
            u = u + "/exportImage"
        else:
            u = u + "/exportImage"

    r = requests.get(u, params=params, timeout=45)
    # Retry with slightly coarser resolution if response is error JSON
    if "image" not in r.headers.get("Content-Type", ""):
        # try coarser
        if target_mpp is not None:
            params.pop("pixelSize", None)
        aspect = h_m / w_m
        width_px = max(256, min(max_px, int(round(w_m / 0.5))))  # 50 cm
        height_px = max(256, min(max_px, int(round(width_px * aspect))))
        params["size"] = f"{width_px},{height_px}"
        r = requests.get(u, params=params, timeout=45)
        if "image" not in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"ImageServer exportImage failed: {r.text[:300]}")

    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode exportImage response.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    mpp_x = (xmax - xmin) / float(img.shape[1])
    mpp_y = (ymax - ymin) / float(img.shape[0])
    return img, mpp_x, mpp_y


def export_map_from_mapserver(service_url: str, xmin: float, ymin: float, xmax: float, ymax: float,
                              target_mpp: Optional[float] = None, max_px: int = 3000) -> Tuple[np.ndarray, float, float]:
    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "format": "png",
        "transparent": "false",
    }
    w_m = xmax - xmin
    h_m = ymax - ymin
    if target_mpp:
        params["pixelSize"] = f"{target_mpp},{target_mpp}"
    else:
        aspect = h_m / w_m
        width_px = max(256, min(max_px, int(round(w_m / 0.3))))
        height_px = max(256, min(max_px, int(round(width_px * aspect))))
        params["size"] = f"{width_px},{height_px}"

    u = sanitize_service_url(service_url).rstrip("/")
    if not u.lower().endswith("/export"):
        if u.lower().endswith("/mapserver"):
            u = u + "/export"
        else:
            u = u + "/export"

    r = requests.get(u, params=params, timeout=45)
    if "image" not in r.headers.get("Content-Type", ""):
        # Try coarser/fallback
        if target_mpp is not None:
            params.pop("pixelSize", None)
        aspect = h_m / w_m
        width_px = max(256, min(max_px, int(round(w_m / 0.5))))
        height_px = max(256, min(max_px, int(round(width_px * aspect))))
        params["size"] = f"{width_px},{height_px}"
        r = requests.get(u, params=params, timeout=45)
        if "image" not in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"MapServer export failed: {r.text[:300]}")

    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode MapServer export response.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    mpp_x = (xmax - xmin) / float(img.shape[1])
    mpp_y = (ymax - ymin) / float(img.shape[0])
    return img, mpp_x, mpp_y


def pick_wms_layer_from_capabilities(cap_xml: str) -> Tuple[Optional[str], Optional[str]]:
    # lightweight parse: pick first named, non-group layer preferring EPSG:3857 then EPSG:4326
    import re
    layers = re.findall(r"<Layer[^>]*>(.*?)</Layer>", cap_xml, flags=re.DOTALL | re.IGNORECASE)
    for block in layers:
        name_m = re.search(r"<Name>([^<]+)</Name>", block, flags=re.IGNORECASE)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        crs_list = re.findall(r"<(CRS|SRS)>(EPSG:\d+)</(CRS|SRS)>", block, flags=re.IGNORECASE)
        crs_vals = [m[1].upper() for m in crs_list]
        if "EPSG:3857" in crs_vals:
            return name, "EPSG:3857"
        if "EPSG:4326" in crs_vals:
            return name, "EPSG:4326"
    return None, None


def getmap_from_wms(wms_url: str, xmin: float, ymin: float, xmax: float, ymax: float,
                    layer_name: Optional[str], img_format: str = "image/png", max_px: int = 3000) -> Tuple[np.ndarray, float, float]:
    base = sanitize_service_url(wms_url)
    # Ensure base ends with WMSServer
    if "WMSServer" not in base and "wmsserver" not in base.lower():
        if "WMSServer" in wms_url:
            base = wms_url.split("?", 1)[0]
        else:
            # attempt to append
            if not base.endswith("/"):
                base += "/"
            base += "WMSServer"

    crs = "EPSG:3857"
    version = "1.3.0"

    # If no layer provided, fetch GetCapabilities and pick one
    if not layer_name:
        cap = requests.get(base, params={"service": "WMS", "request": "GetCapabilities"}, timeout=30)
        if cap.status_code != 200:
            raise RuntimeError("Failed to fetch WMS GetCapabilities; please provide layer name.")
        ln, crs_sel = pick_wms_layer_from_capabilities(cap.text)
        if not ln:
            raise RuntimeError("Could not pick WMS layer; please provide layer name.")
        layer_name = ln
        if crs_sel:
            crs = crs_sel

    # Compute bbox param depending on CRS and version axis order rules
    if crs == "EPSG:3857":
        bbox_param = f"{xmin},{ymin},{xmax},{ymax}"
    else:
        # EPSG:4326 axis order in WMS 1.3.0 is lat,lon
        min_lon, min_lat = webmerc_to_lonlat(xmin, ymin)
        max_lon, max_lat = webmerc_to_lonlat(xmax, ymax)
        if version == "1.3.0":
            bbox_param = f"{min_lat},{min_lon},{max_lat},{max_lon}"
        else:
            bbox_param = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    w_m = xmax - xmin
    h_m = ymax - ymin
    aspect = h_m / max(w_m, 1e-9)
    width_px = max(256, min(max_px, 2048))
    height_px = max(256, min(max_px, int(round(width_px * aspect))))

    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": version,
        "layers": layer_name,
        "styles": "",
        "format": img_format,
        "transparent": "false",
        "crs": crs,
        "bbox": bbox_param,
        "width": width_px,
        "height": height_px,
    }

    r = requests.get(base, params=params, timeout=45)
    if "image" not in r.headers.get("Content-Type", ""):
        # Try smaller request if server rejected size
        width_px = max(256, int(width_px * 0.75))
        height_px = max(256, int(height_px * 0.75))
        params["width"], params["height"] = width_px, height_px
        r = requests.get(base, params=params, timeout=45)
        if "image" not in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"WMS GetMap failed: {r.text[:300]}")

    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode WMS GetMap response.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    mpp_x = (xmax - xmin) / float(img.shape[1])
    mpp_y = (ymax - ymin) / float(img.shape[0])
    return img, mpp_x, mpp_y

# --------------------------------------
# Routes
# --------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    meters_per_pixel: Optional[float] = Form(default=None),
    conf: float = Form(default=0.25),
    iou: float = Form(default=0.5),
    imgsz: int = Form(default=1280),
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
            "image_data_url": data_url,
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
    imgsz: int = Form(1280),
):
    """
    Fetch NAIP via ArcGIS ImageServer for the given lon/lat bbox, then run detection.
    """
    try:
        xmin, ymin, xmax, ymax = bbox4326_to_3857(min_lon, min_lat, max_lon, max_lat)
        img, mpp_x, mpp_y = fetch_naip_bbox_mercator(xmin, ymin, xmax, ymax, width_px=width_px, use_usgs=use_usgs)

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
            "export_bbox_3857": [xmin, ymin, xmax, ymax],
            "image_data_url": data_url,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/infer_generic_imagery")
async def infer_generic_imagery(
    service_url: str = Form(...),
    min_lon: float = Form(...),
    min_lat: float = Form(...),
    max_lon: float = Form(...),
    max_lat: float = Form(...),
    service_type: str = Form(default="auto"),  # auto | image_server | map_server | wms
    wms_layer: Optional[str] = Form(default=None),
    target_mpp: Optional[float] = Form(default=None),
    conf: float = Form(default=0.25),
    iou: float = Form(default=0.5),
    imgsz: int = Form(default=1280),
):
    try:
        xmin, ymin, xmax, ymax = bbox4326_to_3857(min_lon, min_lat, max_lon, max_lat)
        su = sanitize_service_url(service_url)
        st = (service_type or "auto").lower()

        if st == "auto":
            low = su.lower()
            if "imageserver" in low:
                st = "image_server"
            elif "mapserver" in low:
                st = "map_server"
            elif "wmsserver" in low or "service=wms" in low or "request=getcapabilities" in low:
                st = "wms"
            else:
                # heuristic: try ImageServer export first
                st = "image_server"

        # Fetch image per type
        if st == "image_server":
            img, mpp_x, mpp_y = export_image_from_imageserver(su, xmin, ymin, xmax, ymax, target_mpp=target_mpp)
        elif st == "map_server":
            img, mpp_x, mpp_y = export_map_from_mapserver(su, xmin, ymin, xmax, ymax, target_mpp=target_mpp)
        elif st == "wms":
            img, mpp_x, mpp_y = getmap_from_wms(su, xmin, ymin, xmax, ymax, layer_name=wms_layer)
        else:
            return JSONResponse({"error": "Unsupported service type"}, status_code=400)

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
                polygons.append(max(contours, key=cv2.contourArea).reshape(-1, 2) if contours else None)

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
            "export_bbox_3857": [xmin, ymin, xmax, ymax],
            "source_service_url": su,
            "image_data_url": data_url,
            "service_type": st,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
