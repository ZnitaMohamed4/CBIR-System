import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource
from flask_cors import CORS
from pathlib import Path

from services.object_detection import ObjectDetectionService
from services.feature_extraction import FeatureExtractionService
from services.similarity_search import SimilaritySearchService
from services.image_manager import ImageManager

app = Flask(__name__)
CORS(app)
api = Api(app)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
app.config['UPLOAD_FOLDER'] = Path(os.getenv('UPLOAD_FOLDER', '/app/uploads'))
app.config['DATABASE_PATH'] = Path(os.getenv('DATABASE_PATH', '/app/database/features.json'))
app.config['MODEL_PATH'] = Path(os.getenv('MODEL_PATH', '/app/models/yolov8n_15classes_finetuned.pt'))
app.config['SEG_MODEL_PATH'] = Path(os.getenv('SEG_MODEL_PATH', '/app/models/yolov8n-seg.pt'))

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure directories exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['DATABASE_PATH'].parent.mkdir(parents=True, exist_ok=True)

# --- SERVICE INITIALIZATION ---
detection_service = ObjectDetectionService(str(app.config['MODEL_PATH']))
feature_service = FeatureExtractionService(seg_model_path=str(app.config['SEG_MODEL_PATH']))
similarity_service = SimilaritySearchService(str(app.config['DATABASE_PATH']))
image_manager = ImageManager(str(app.config['UPLOAD_FOLDER']))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- API RESOURCES ---

class ImageUpload(Resource):
    def post(self):
        if 'images' not in request.files:
            return {'error': 'No images provided'}, 400
        files = request.files.getlist('images')
        results = [image_manager.save_image(f) for f in files if f and allowed_file(f.filename)]
        return {'uploaded': results}, 201

class ImageList(Resource):
    def get(self):
        return {'images': image_manager.get_all_images()}, 200
    def delete(self):
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])
        results = image_manager.delete_images(image_ids)
        return {'deleted': results}, 200

class ImageDetail(Resource):
    def get(self, image_id):
        info = image_manager.get_image(image_id)
        return (info, 200) if info else ({'error': 'Image not found'}, 404)

class ObjectDetect(Resource):
    def post(self):
        data = request.get_json() or {}
        image_id = data.get('image_id')
        path = image_manager.get_image_path(image_id)
        if not path: return {'error': 'Image not found'}, 404
        detections = detection_service.detect(path)
        similarity_service.save_detections(image_id, detections)
        return {'image_id': image_id, 'detections': detections}, 200

class ObjectDetectBatch(Resource):
    def post(self):
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])
        results = []
        for rid in image_ids:
            path = image_manager.get_image_path(rid)
            if path:
                det = detection_service.detect(path)
                similarity_service.save_detections(rid, det)
                results.append({'image_id': rid, 'detections': det})
        return {'results': results}, 200

class FeatureExtract(Resource):
    def post(self):
        data = request.get_json() or {}
        img_id, obj_id = data.get('image_id'), data.get('object_id')
        path = image_manager.get_image_path(img_id)
        detections = similarity_service.get_detections(img_id)
        if not path or not detections or obj_id >= len(detections):
            return {'error': 'Object or detections not found'}, 404
        features = feature_service.extract_all_features(path, detections[obj_id]['bbox'])
        similarity_service.save_features(img_id, obj_id, features)
        return {'image_id': img_id, 'object_id': obj_id, 'features': features}, 200

class FeatureExtractBatch(Resource):
    def post(self):
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])
        results = []
        for rid in image_ids:
            path = image_manager.get_image_path(rid)
            detections = similarity_service.get_detections(rid)
            if not path or not detections: continue
            for idx, d in enumerate(detections):
                feat = feature_service.extract_all_features(path, d['bbox'])
                similarity_service.save_features(rid, idx, feat)
                results.append({'image_id': rid, 'object_id': idx, 'class': d['class']})
        return {'processed': results}, 200

class SimilaritySearch(Resource):
    def post(self):
        data = request.get_json() or {}
        q_img, q_obj = data.get('query_image_id'), data.get('query_object_id')
        top_k = data.get('top_k', 10)
        
        # 1. Get query features and detections
        features = similarity_service.get_features(q_img, q_obj)
        detections = similarity_service.get_detections(q_img)
        
        if not features or not detections:
            return {'error': 'Features or detections not found. Extract features first.'}, 404
        
        # 2. Perform the search
        query_class = detections[q_obj]['class']
        results = similarity_service.find_similar(
            query_features=features, 
            query_class=query_class,
            top_k=top_k * 2, # Get extra to account for potential deleted files
            weights=data.get('weights'),
            exclude_image_id=q_img
        )
        
        # 3. Add filenames for the frontend to display images
        valid_results = []
        for item in results:
            info = image_manager.get_image(item['image_id'])
            if info:
                item['filename'] = info['filename']
                valid_results.append(item)
                if len(valid_results) >= top_k:
                    break
        
        # --- FIXED KEY: 'similar_objects' instead of 'results' ---
        return {
            'query_image_id': q_img,
            'query_object_id': q_obj,
            'query_class': query_class,
            'similar_objects': valid_results 
        }, 200

class FeatureVisualize(Resource):
    def get(self, image_id, object_id):
        feats = similarity_service.get_features(image_id, int(object_id))
        if not feats: return {'error': 'Not found'}, 404
        return {'features': feature_service.format_features_for_display(feats)}, 200

class DatabaseStats(Resource):
    def get(self):
        return similarity_service.get_statistics(), 200

# --- ROUTES ---
api.add_resource(ImageUpload, '/api/images/upload')
api.add_resource(ImageList, '/api/images')
api.add_resource(ImageDetail, '/api/images/<string:image_id>')
api.add_resource(ObjectDetect, '/api/detect')
api.add_resource(ObjectDetectBatch, '/api/detect/batch')
api.add_resource(FeatureExtract, '/api/features/extract')
api.add_resource(FeatureExtractBatch, '/api/features/extract/batch')
api.add_resource(SimilaritySearch, '/api/search/similar')
api.add_resource(FeatureVisualize, '/api/features/<string:image_id>/<int:object_id>')
api.add_resource(DatabaseStats, '/api/stats')

@app.route('/api/images/file/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)