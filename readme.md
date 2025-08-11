# Solar Panel Detector with NAIP and High-Resolution Maryland Six Inch Imagery Integration

A web application that detects solar panels on residential homes using high-resolution aerial imagery. The app can process uploaded images or fetch imagery directly from public services using bounding box coordinates, supporting NAIP multispectral imagery as well as Maryland Six Inch ultra high-res imagery for enhanced rooftop detection accuracy.

## Features

- **Option A**: Upload local multispectral images (e.g., NAIP 4-band RGB+NIR) for solar panel detection
- **Option B**: Fetch high-resolution aerial imagery by drawing bounding boxes on an interactive map
- Support for **NAIP** and **Maryland Six Inch** imagery sources
- Uses YOLOv8 segmentation model trained specifically for solar panels
- Supports 4-band multispectral imagery including NIR for enhanced detection and spectral filtering
- Computes spectral indices including NDVI, NDBI, SPSI, and Solar Photovoltaic Panel Index (SPPI) from recent research
- Optional image sharpening and test-time augmentation (TTA) toggles for improved recall
- Automatic area calculation in pixels and square meters
- Choice between USGS and USDA NAIP services
- Real-time detection with visual overlays on static image and interactive map
- Detection parameter controls: confidence threshold, IoU threshold, image size
- Interactive map interface with drawing tools, hybrid basemap (satellite + labels), area validation (0.5–100 km²)
- Table display listing each detected panel with confidence score and approximate latitude/longitude coordinates

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/solar-panel-detector.git
   cd solar-panel-detector
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   uvicorn app:app --reload
   ```

5. **Open your browser** to `http://127.0.0.1:8000`

## Usage

### Option A: Upload Image

- Upload local multispectral images with up to 4 bands (Red, Green, Blue, Near-Infrared)
- Automatically detects solar panels using multispectral data and spectral indices
- Uses NDVI-based false positive filtering for vegetation
- Visual results shown with detection overlays

### Option B: Fetch High-Resolution Imagery by Bounding Box (with map UI)

1. Draw a bounding box anywhere in the U.S. (valid area between 0.5 and 100 km²)
2. Choose image width (256 to 3500 pixels)
3. Select imagery source:
   - **NAIP** (USGS or USDA)
   - **Maryland Six Inch** high-resolution imagery (MD)
4. Adjust detection parameters:
   - **Confidence threshold** (min detection confidence)
   - **IoU threshold** (Non-Maximum Suppression overlap)
   - **Image size** (input resolution for model inference)
5. Toggle advanced options:
   - **augment** (test-time augmentation for higher recall)
   - **sharpen** (image sharpening to improve small panel detection)
6. Click "Run Detection"
7. View visual overlays on both static image and interactive hybrid map in real time
8. Inspect detected panels in an expandable table with each detection’s confidence and approximate geographic coordinates

**Example bounding box coordinates for Seattle area:**

- Min longitude: -122.337
- Min latitude: 47.610  
- Max longitude: -122.329
- Max latitude: 47.616

## Technical Details

### Imagery Services Used

- **USGS NAIP**: https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer
- **USDA NAIP**: https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer
- **Maryland Six Inch Imagery**: https://mdgeodata.md.gov/imagery/rest/services/SixInch/SixInchImagery/ImageServer  
  *Note: Maryland Six Inch imagery is available only in Maryland. The interface validates bounding boxes to ensure selection within Maryland when this source is chosen.*

### Model

- **YOLOv8s segmentation** trained for solar panels
- Source: [finloop/yolov8s-seg-solar-panels](https://huggingface.co/finloop/yolov8s-seg-solar-panels)
- License: MIT

### Spectral Processing

- Supports 4-band (RGB + Near-Infrared) imagery to compute spectral indices
- Computes NDVI and NDBI indices used to filter vegetation false positives
- Includes SPSI and Solar Photovoltaic Panel Index (SPPI) for enhanced discrimination informed by recent research
- Optional image sharpening and test-time augmentation for robust detection

### Coordinate System

- Input: WGS84 (EPSG:4326) longitude/latitude
- Processing: Web Mercator (EPSG:3857) for image fetching and mapping
- Automatic coordinate transformation included

## Project Structure

```
solar-panel-detector/
├── app.py              # FastAPI backend
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html     # Web interface with map, detection table, and controls
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## API Endpoints

- `GET /`: Serve the web interface
- `POST /infer`: Process uploaded images (multispectral support)
- `POST /infer_naip`: Fetch NAIP imagery and detect panels (with augment and sharpen)
- `POST /infer_imagery`: Fetch imagery by bounding box and selected service (NAIP or MD Six Inch), run detection with advanced options
- `POST /extract_features`: Export multispectral spectral indices as 8-band feature stack for custom model training

## Dependencies

- FastAPI: Web framework
- Ultralytics YOLOv8: Model for segmentation
- OpenCV: Image processing
- NumPy: Numerical operations
- Requests: HTTP requests for imagery fetching
- Hugging Face Hub: Model downloading

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Submit a pull request  

## Support

For issues or questions, please open a GitHub issue.

