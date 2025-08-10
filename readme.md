# Solar Panel Detector with NAIP Integration

A web application that detects solar panels on residential homes using NAIP (National Agriculture Imagery Program) imagery. The app can either process uploaded images or fetch NAIP imagery directly from public services using bounding box coordinates.

## Features

- **Option A**: Upload local NAIP images for solar panel detection
- **Option B**: Fetch NAIP imagery by entering longitude/latitude coordinates
- Uses YOLOv8 segmentation model trained specifically for solar panels
- Automatic area calculation in both pixels and square meters
- Choice between USGS and USDA NAIP services
- Real-time detection with visual overlays

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

### Option B: Fetch NAIP by Bounding Box

1. Enter the bounding box coordinates (longitude/latitude)
2. Choose image width (256-3000 pixels)
3. Select NAIP source (USGS or USDA)
4. Adjust detection parameters if needed
5. Click "Fetch NAIP + Detect"

**Example coordinates** (Seattle area):
- Min longitude: -122.337
- Min latitude: 47.610  
- Max longitude: -122.329
- Max latitude: 47.616

### Detection Parameters

- **Confidence threshold**: Minimum confidence for detections (default: 0.25)
- **IoU threshold**: Non-maximum suppression threshold (default: 0.5)
- **Image width**: Export resolution in pixels (default: 1024)

## Technical Details

### NAIP Services Used

- **USGS NAIP**: https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer
- **USDA NAIP**: https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer

### Model

- **YOLOv8s segmentation** trained for solar panels
- Source: [finloop/yolov8s-seg-solar-panels](https://huggingface.co/finloop/yolov8s-seg-solar-panels)
- License: MIT

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
│   └── index.html     # Web interface
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## API Endpoints

- `GET /`: Serve the web interface
- `POST /infer`: Process uploaded images
- `POST /infer_naip`: Fetch NAIP imagery and detect panels

## Dependencies

- FastAPI: Web framework
- Ultralytics: YOLOv8 model
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
