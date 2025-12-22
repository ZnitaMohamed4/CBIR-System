# CBIR System Backend - Flask REST API

Content-Based Image Retrieval (CBIR) system with YOLOv8 object detection and comprehensive visual feature extraction.

## Architecture

### Services
- **ObjectDetectionService**: YOLO-based object detection using fine-tuned YOLOv8n model (15 categories)
- **FeatureExtractionService**: Extract color, texture, and shape features
- **SimilaritySearchService**: Find similar objects using weighted feature matching
- **ImageManager**: Handle image upload, storage, download, and transformations

### Features Extracted

#### Color Features
- RGB and HSV histograms (16 bins per channel)
- **Dominant colors** (K-Means clustering, 5 colors with percentages)
- Color moments (mean, standard deviation)

#### Texture Features
- **Tamura descriptors**: coarseness, contrast, directionality
- **Gabor filters**: Multi-scale and orientation responses (4 orientations × 2 frequencies)
- **Local Binary Pattern (LBP)**: Rotation-invariant texture descriptor

#### Shape Features
- **Hu moments**: 7 invariant moments from object contours
- **HOG**: Histogram of Oriented Gradients
- **Contour Orientation Histogram**: Edge direction distribution from significant contours

## API Endpoints

### Image Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/images/upload | POST | Upload images (single or batch) |
| /api/images | GET | List all images |
| /api/images/<id> | GET | Get image details |
| /api/images/<id> | DELETE | Delete image |
| /api/images | DELETE | Delete multiple images (batch) |
| /api/images/<id>/transform | POST | Apply transformations |
| /api/images/download/<id> | GET | Download image |
| /api/images/file/<filename> | GET | Serve image file |

### Object Detection
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/detect | POST | Detect objects in image |
| /api/detect/batch | POST | Batch detection |

### Feature Extraction
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/features/extract | POST | Extract features from object |
| /api/features/extract/batch | POST | Batch extraction |
| /api/features/<image_id>/<object_id> | GET | Get formatted features |

### Similarity Search
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/search/similar | POST | Find similar objects |

### Utilities
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/stats | GET | Database statistics |
| /api/health | GET | Health check |

## Image Transformations

The API supports the following transformations:

| Transform Type | Parameters | Description |
|---------------|------------|-------------|
| crop | x, y, width, height | Crop region from image |
| resize | width, height | Resize to exact dimensions |
| scale | scale (float) | Scale with aspect ratio preserved |
| resize_keep_aspect | max_width, max_height | Resize within bounds |
| rotate | angle (degrees) | Rotate image |
| flip | direction (1=h, 0=v, -1=both) | Flip image |

## Similarity Search Weights

Default weights for similarity computation:

| Feature | Default Weight |
|---------|---------------|
| Color | 0.25 |
| Tamura Texture | 0.15 |
| Gabor Texture | 0.15 |
| LBP Texture | 0.10 |
| Hu Moments | 0.10 |
| HOG Shape | 0.15 |
| Contour Orientation | 0.10 |

## Setup

\`\`\`bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
\`\`\`

The server will start at http://localhost:5000

## 15 Object Categories

The YOLOv8n model is fine-tuned on these ImageNet categories:

1. Person
2. Bicycle
3. Car
4. Airplane
5. Boat
6. Traffic Light
7. Bird
8. Cat
9. Dog
10. Horse
11. Umbrella
12. Bottle
13. Apple
14. Pizza
15. Laptop

See [README.md](../README.md) for detailed information.
