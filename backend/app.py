# /home/muhammed/Documents/SmartGallery/backend/app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import json

from services.object_detection import ObjectDetectionService
from services.feature_extraction import FeatureExtractionService
from services.similarity_search import SimilaritySearchService
from services.image_manager import ImageManager

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend
api = Api(app)

# Configuration
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['MODEL_PATH'] = Path(__file__).parent.parent / 'models' / 'yolov8n_15classes_finetuned.pt'
app.config['DATABASE_PATH'] = Path(__file__).parent / 'database' / 'features.json'

# Create necessary directories
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
(Path(__file__).parent / 'database').mkdir(parents=True, exist_ok=True)

# Initialize services
detection_service = ObjectDetectionService(str(app.config['MODEL_PATH']))
feature_service = FeatureExtractionService()
similarity_service = SimilaritySearchService(str(app.config['DATABASE_PATH']))
image_manager = ImageManager(str(app.config['UPLOAD_FOLDER']))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# REST API Resources
class ImageUpload(Resource):
    """Upload single or multiple images"""
    def post(self):
        if 'images' not in request.files:
            return {'error': 'No images provided'}, 400
        
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                result = image_manager.save_image(file)
                results.append(result)
        
        return {'uploaded': results}, 201


class ImageList(Resource):
    """Get all images or delete multiple"""
    def get(self):
        images = image_manager.get_all_images()
        return {'images': images}, 200
    
    def delete(self):
        """Delete multiple images"""
        data = request.get_json()
        image_ids = data.get('image_ids', [])
        results = image_manager.delete_images(image_ids)
        return {'deleted': results}, 200


class ImageDetail(Resource):
    """Get, update or delete single image"""
    def get(self, image_id):
        image_info = image_manager.get_image(image_id)
        if not image_info:
            return {'error': 'Image not found'}, 404
        return image_info, 200
    
    def delete(self, image_id):
        success = image_manager.delete_image(image_id)
        if not success:
            return {'error': 'Image not found'}, 404
        return {'message': 'Image deleted successfully'}, 200


class ImageTransform(Resource):
    """Apply transformations to images (crop, resize, rotate)"""
    def post(self, image_id):
        data = request.get_json()
        transform_type = data.get('transform_type')
        params = data.get('params', {})
        
        result = image_manager.transform_image(image_id, transform_type, params)
        if 'error' in result:
            return result, 400
        
        return result, 201


class ObjectDetect(Resource):
    """Detect objects in an image"""
    def post(self):
        data = request.get_json()
        image_id = data.get('image_id')
        
        if not image_id:
            return {'error': 'image_id required'}, 400
        
        image_path = image_manager.get_image_path(image_id)
        if not image_path:
            return {'error': 'Image not found'}, 404
        
        detections = detection_service.detect(image_path)
        
        # Save detections to database
        similarity_service.save_detections(image_id, detections)
        
        return {
            'image_id': image_id,
            'detections': detections
        }, 200


class ObjectDetectBatch(Resource):
    """Detect objects in multiple images"""
    def post(self):
        data = request.get_json()
        image_ids = data.get('image_ids', [])
        
        results = []
        for image_id in image_ids:
            image_path = image_manager.get_image_path(image_id)
            if image_path:
                detections = detection_service.detect(image_path)
                similarity_service.save_detections(image_id, detections)
                results.append({
                    'image_id': image_id,
                    'detections': detections
                })
        
        return {'results': results}, 200


class FeatureExtract(Resource):
    """Extract visual features from detected objects"""
    def post(self):
        data = request.get_json()
        image_id = data.get('image_id')
        object_id = data.get('object_id')  # Index of the detected object
        
        if not image_id:
            return {'error': 'image_id required'}, 400
        
        image_path = image_manager.get_image_path(image_id)
        if not image_path:
            return {'error': 'Image not found'}, 404
        
        # Get detection bbox
        detections = similarity_service.get_detections(image_id)
        if not detections or object_id >= len(detections):
            return {'error': 'Object not found'}, 404
        
        bbox = detections[object_id]['bbox']
        
        # Extract features
        features = feature_service.extract_all_features(image_path, bbox)
        
        # Save features
        similarity_service.save_features(image_id, object_id, features)
        
        return {
            'image_id': image_id,
            'object_id': object_id,
            'features': features
        }, 200


class FeatureExtractBatch(Resource):
    """Extract features for all objects in one or multiple images"""
    def post(self):
        data = request.get_json()
        image_ids = data.get('image_ids', [])
        
        results = []
        for image_id in image_ids:
            image_path = image_manager.get_image_path(image_id)
            if not image_path:
                continue
            
            detections = similarity_service.get_detections(image_id)
            if not detections:
                continue
            
            for obj_idx, detection in enumerate(detections):
                bbox = detection['bbox']
                features = feature_service.extract_all_features(image_path, bbox)
                similarity_service.save_features(image_id, obj_idx, features)
                results.append({
                    'image_id': image_id,
                    'object_id': obj_idx,
                    'class': detection['class'],
                    'confidence': detection['confidence']
                })
        
        return {'processed': results}, 200


class SimilaritySearch(Resource):
    """Search for similar objects"""
    def post(self):
        data = request.get_json()
        query_image_id = data.get('query_image_id')
        query_object_id = data.get('query_object_id')
        top_k = data.get('top_k', 10)
        weights = data.get('weights', None)
        
        if not query_image_id or query_object_id is None:
            return {'error': 'query_image_id and query_object_id required'}, 400
        
        query_features = similarity_service.get_features(query_image_id, query_object_id)
        if not query_features:
            return {'error': 'Features not found. Extract features first.'}, 404
        
        detections = similarity_service.get_detections(query_image_id)
        if not detections or query_object_id >= len(detections):
            return {'error': 'Detection not found'}, 404
        
        query_class = detections[query_object_id]['class']
        
        similar_objects = similarity_service.find_similar(
            query_features=query_features,
            query_class=query_class,
            top_k=top_k, 
            weights=weights,
            exclude_image_id=query_image_id,
            same_class_only=True
        )
        
        # ✅ ADD FILENAME TO EACH RESULT
        for obj in similar_objects:
            image_info = image_manager.get_image(obj['image_id'])
            if image_info:
                obj['filename'] = image_info['filename']
        
        return {
            'query_image_id': query_image_id,
            'query_object_id': query_object_id,
            'query_class': query_class,
            'similar_objects': similar_objects
        }, 200

class FeatureVisualize(Resource):
    """Get formatted features for visualization"""
    def get(self, image_id, object_id):
        features = similarity_service.get_features(image_id, int(object_id))
        if not features:
            return {'error': 'Features not found'}, 404
        
        # Format features for display
        formatted = feature_service.format_features_for_display(features)
        
        return {
            'image_id': image_id,
            'object_id': object_id,
            'features': formatted
        }, 200


class DatabaseStats(Resource):
    """Get database statistics"""
    def get(self):
        stats = similarity_service.get_statistics()
        return stats, 200


# Register API routes
api.add_resource(ImageUpload, '/api/images/upload')
api.add_resource(ImageList, '/api/images')
api.add_resource(ImageDetail, '/api/images/<string:image_id>')
api.add_resource(ImageTransform, '/api/images/<string:image_id>/transform')
api.add_resource(ObjectDetect, '/api/detect')
api.add_resource(ObjectDetectBatch, '/api/detect/batch')
api.add_resource(FeatureExtract, '/api/features/extract')
api.add_resource(FeatureExtractBatch, '/api/features/extract/batch')
api.add_resource(SimilaritySearch, '/api/search/similar')
api.add_resource(FeatureVisualize, '/api/features/<string:image_id>/<int:object_id>')
api.add_resource(DatabaseStats, '/api/stats')

# Serve uploaded images
@app.route('/api/images/file/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Download image endpoint
@app.route('/api/images/download/<string:image_id>')
def download_image(image_id):
    """Download an image by its ID"""
    image_path = image_manager.get_image_path(image_id)
    if not image_path:
        return jsonify({'error': 'Image not found'}), 404
    
    from pathlib import Path
    filepath = Path(image_path)
    return send_from_directory(
        filepath.parent, 
        filepath.name, 
        as_attachment=True,
        download_name=filepath.name
    )


@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': detection_service.is_loaded()}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)