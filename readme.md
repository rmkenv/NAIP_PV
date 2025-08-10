# Solar Panel Detector with NAIP Integration

A web application that detects solar panels on residential homes using NAIP (National Agriculture Imagery Program) imagery. The app can either process uploaded images or fetch NAIP imagery directly from public services using bounding box coordinates.

## Features

- **Option A**: Upload local NAIP images for solar panel detection
- **Option B**: Fetch NAIP imagery by entering longitude/latitude coordinates
- Uses YOLOv8 segmentation model trained specifically for solar panels
- Supports 4-band multispectral imagery including NIR for enhanced detection and spectral filtering
- Automatic area calculation in both pixels and square meters
- Choice between USGS and USDA NAIP services
- Real-time detection with visual overlays on image and interactive map
- Detection parameter controls: confidence threshold, IoU threshold, image size
- Advanced options: augment (test-time augmentation) and sharpen toggles for improved detection recall
- Interactive map interface with drawing tools and area validation (0.5–100 km²)

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

- Upload local NAIP images with up to 4 bands (Red, Green, Blue, Near-Infrared)
- Automatically detects solar panels using multispectral data
- Uses NDVI-based false positive filtering for vegetation
- Visual results shown with detection overlays

### Option B: Fetch NAIP by Bounding Box (with map UI)

1. Draw a bounding box anywhere in the U.S. (valid area between 0.5 and 100 km²)
2. Choose image width (256 to 3000 pixels)
3. Select NAIP data source (USGS or USDA)
4. Adjust detection parameters:
   - **Confidence threshold** (min detection confidence)
   - **IoU threshold** (Non-Maximum Suppression overlap)
   - **Image size** (input resolution for model inference)
5. Toggle advanced options:
   - **augment** (test-time augmentation for higher recall)
   - **sharpen** (image sharpening to improve small panel detection)
6. Click "Run detection"
7. View visual overlays on both static image and interactive map in real time

**Example bounding box coordinates for Seattle area:**

- Min longitude: -122.337
- Min latitude: 47.610  
- Max longitude: -122.329
- Max latitude: 47.616

## Technical Details

### NAIP Services Used

- **USGS NAIP**: https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer
- **USDA NAIP**: https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer

### Model

- **YOLOv8s segmentation** trained for solar panels
- Source: [finloop/yolov8s-seg-solar-panels](https://huggingface.co/finloop/yolov8s-seg-solar-panels)
- License: MIT

### Spectral Processing

- Supports 4-band (RGB + Near-Infrared) NAIP imagery to compute spectral indices
- Computes NDVI and NDBI indices used to filter vegetation false positives
- Lays groundwork for improved detection using multispectral/hyperspectral features

### Coordinate System

- Input: WGS84 (EPSG:4326) longitude/latitude
- Processing: Web Mercator (EPSG:3857) for NAIP services
- Automatic coordinate transformation included

## Project Structure

```
solar-panel-detector/
├── app.py              # FastAPI backend
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html     # Web interface with map and controls
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## API Endpoints

- `GET /`: Serve the web interface
- `POST /infer`: Process uploaded images (multispectral support)
- `POST /infer_naip`: Fetch NAIP imagery and detect panels with augment and sharpen options

## Dependencies

- FastAPI: Web framework
- Ultralytics YOLOv8: Model for segmentation
- OpenCV: Image processing
- NumPy: Numerical operations
- Requests: HTTP requests for NAIP fetching
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

***
