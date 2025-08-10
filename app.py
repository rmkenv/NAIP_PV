# app.py
import base64
import os
from typing import Optional, List, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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


# --------------------- Image utils ---------------------

def to_3ch_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Empty image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def numpy_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid PNG/JPG.")
    return to_3ch_bgr(img)


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode result image.")
    return buf.tobytes()


def unsharp_mask(img: np.ndarray, radius: int = 3, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """Simple unsharp mask: enhance small edges; conservative defaults."""
    if radius <= 0 or amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low, high = cv2.threshold(cv2.cvtColor(cv2.absdiff(img, blurred), cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(high, cv2.COLOR_GRAY2BGR)
        return np.where(mask > 0, sharp, img)
    return np.clip(sharp, 0, 255).astype(np.uint8)


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
USGS_NAIP_EXPORT = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
USDA_NAIP_EXPORT = "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage"

def fetch_naip_bbox_mercator(
    xmin: float, ymin: float, xmax: float, ymax: float,
    width_px: int = 2048,
    use_usgs: bool = True
) -> Tuple[np.ndarray, float, float]:
    width_px = int(max(512, min(3500, width_px)))
    w_m = xmax - xmin
    h_m = ymax - ymin
    if w_m <= 0 or h_m <= 0:
        raise ValueError("Invalid bbox.")
    aspect = h_m / w_m
    height_px = int(round(width_px * aspect))
    height_px = max(512, min(3500, height_px))

    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{width_px},{height_px}",
        "format": "png",
    }
    url = USGS_NAIP_EXPORT if use_usgs else USDA_NAIP_EXPORT
    resp = requests.get(url, params=params, timeout=45)
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
    img = to_3ch_bgr(img)

    mpp_x = w_m / float(img.shape[1])
    mpp_y = h_m / float(img.shape[0])
    return img, mpp_x, mpp_y


# ---------- YOLO inference helpers (tiling + options) ----------

def run_yolo_single(img_bgr: np.ndarray, conf: float, iou: float, imgsz: int, augment: bool, max_det: int = 5000):
    return MODEL(img_bgr, conf=conf, iou=iou, imgsz=imgsz, augment=augment, max_det=max_det, verbose=False)[0]


def iou_with(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=int)
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = iou_with(boxes[i], boxes[rest])
        idxs = rest[ious < iou_thresh]
    return np.array(keep, dtype=int)


def tile_inference_yolo(
    model,
    img_bgr: np.ndarray,
    tile: int = 1024,
    overlap: int = 256,
    imgsz: int = 1280,
    conf: float = 0.2,
    iou: float = 0.4,
    augment: bool = False,
    max_det: int = 5000,
):
    H, W = img_bgr.shape[:2]
    stride = max(1, tile - overlap)
    all_boxes, all_scores, all_classes, all_polys = [], [], [], []

    for y in range(0, max(H - tile, 0) + 1, stride):
        for x in range(0, max(W - tile, 0) + 1, stride):
            crop = img_bgr[y:y + tile, x:x + tile]
            if crop.shape[0] < tile or crop.shape[1] < tile:
                pad_h = tile - crop.shape[0]
                pad_w = tile - crop.shape[1]
                crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            res = run_yolo_single(crop, conf=conf, iou=iou, imgsz=imgsz, augment=augment, max_det=max_det)

            if res.masks is not None and hasattr(res.masks, "xy"):
                for poly in res.masks.xy:
                    p = np.array(poly)
                    p[:, 0] += x
                    p[:, 1] += y
                    all_polys.append(p)
            elif res.masks is not None and hasattr(res.masks, "data"):
                mdata = res.masks.data.cpu().numpy().astype(np.uint8)
                for m in mdata:
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                    cnt[:, 0] += x
                    cnt[:, 1] += y
                    all_polys.append(cnt)

            if res.boxes is not None and res.boxes.xyxy is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") and res.boxes.conf is not None else np.ones((boxes.shape[0],), dtype=float)
                classes = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") and res.boxes.cls is not None else np.zeros((boxes.shape[0],), dtype=float)
                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)

    if all_boxes:
        all_boxes = np.vstack(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)
        keep = nms_numpy(all_boxes, all_scores, iou_thresh=0.5)
        all_boxes = all_boxes[keep]
        all_scores = all_scores[keep]
        all_classes = all_classes[keep]
    else:
        all_boxes = np.empty((0, 4))
        all_scores = np.empty((0,))
        all_classes = np.empty((0,))

    return all_boxes, all_scores, all_classes, all_polys


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    meters_per_pixel: Optional[float] = Form(default=None),
    conf: float = Form(default=0.2),
    iou: float = Form(default=0.4),
    imgsz: int = Form(default=1280),
    augment: bool = Form(default=False),
    sharpen: bool = Form(default=False)
):
    try:
        data = await file.read()
        img = numpy_from_upload(data)
        if sharpen:
            img = unsharp_mask(img, radius=1.6, amount=0.8, threshold=0)

        H, W = img.shape[:2]
        if max(H, W) > 1500:
            boxes, scores, classes, polygons = tile_inference_yolo(MODEL, img, tile=1024, overlap=256, imgsz=imgsz, conf=conf, iou=iou, augment=augment, max_det=5000)
            result_img = draw_detections(img, polygons, boxes, scores)
        else:
            res = run_yolo_single(img, conf=conf, iou=iou, imgsz=imgsz, augment=augment, max_det=5000)
            polygons: List[np.ndarray] = []
            if res.masks is not None and hasattr(res.masks, "xy"):
                for poly in res.masks.xy:
                    polygons.append(np.array(poly))
            elif res.masks is not None and hasattr(res.masks, "data"):
                mdata = res.masks.data.cpu().numpy().astype(np.uint8)
                for m in mdata:
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        polygons.append(None)
                    else:
                        cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
                        polygons.append(cnt)

            boxes = None
            scores = None
            if res.boxes is not None and res.boxes.xyxy is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                if hasattr(res.boxes, "conf") and res.boxes.conf is not None:
                    scores = res.boxes.conf.cpu().numpy()

            result_img = draw_detections(res.orig_img if hasattr(res, "orig_img") else img, polygons, boxes, scores)

        areas_px = polygon_areas_in_pixels(polygons)
        total_area_px = float(np.sum(areas_px))
        mpp = float(meters_per_pixel) if meters_per_pixel is not None else None
        total_area_m2 = (total_area_px * (mpp ** 2)) if mpp is not None else None
        count = int(len([a for a in areas_px if a > 0]))

        if count == 0:
            cv2.putText(result_img, "No panels detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        png_bytes = encode_png(result_img)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        return JSONResponse({
            "count": count,
            "total_area_pixels": total_area_px,
            "meters_per_pixel": mpp,
            "total_area_m2": total_area_m2,
            "used_augment": bool(augment),
            "used_sharpen": bool(sharpen),
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
    width_px: int = Form(2048),
    use_usgs: bool = Form(True),
    conf: float = Form(0.2),
    iou: float = Form(0.4),
    imgsz: int = Form(1280),
    augment: bool = Form(default=False),
    sharpen: bool = Form(default=False)
):
    """
    Fetch NAIP via ArcGIS ImageServer for the given lon/lat bbox, then run detection.
    Options: augment (TTA) and unsharp mask to improve small residential panel recall.
    """
    try:
        xmin, ymin, xmax, ymax = bbox4326_to_3857(min_lon, min_lat, max_lon, max_lat)
        img, mpp_x, mpp_y = fetch_naip_bbox_mercator(xmin, ymin, xmax, ymax, width_px=width_px, use_usgs=use_usgs)
        img = to_3ch_bgr(img)
        if sharpen:
            img = unsharp_mask(img, radius=1.6, amount=0.8, threshold=0)

        H, W = img.shape[:2]
        if max(H, W) > 1500:
            boxes, scores, classes, polygons = tile_inference_yolo(MODEL, img, tile=1024, overlap=256, imgsz=imgsz, conf=conf, iou=iou, augment=augment, max_det=5000)
            vis = draw_detections(img, polygons, boxes, scores)
        else:
            res = run_yolo_single(img, conf=conf, iou=iou, imgsz=imgsz, augment=augment, max_det=5000)
            polygons: List[np.ndarray] = []
            if res.masks is not None and hasattr(res.masks, "xy"):
                for poly in res.masks.xy:
                    polygons.append(np.array(poly))
            elif res.masks is not None and hasattr(res.masks, "data"):
                mdata = res.masks.data.cpu().numpy().astype(np.uint8)
                for m in mdata:
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        polygons.append(None)
                    else:
                        cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
                        polygons.append(cnt)

            boxes = None
            scores = None
            if res.boxes is not None and res.boxes.xyxy is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                if hasattr(res.boxes, "conf") and res.boxes.conf is not None:
                    scores = res.boxes.conf.cpu().numpy()
            vis = draw_detections(res.orig_img if hasattr(res, "orig_img") else img, polygons, boxes, scores)

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
            "used_augment": bool(augment),
            "used_sharpen": bool(sharpen),
            "image_data_url": data_url
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
