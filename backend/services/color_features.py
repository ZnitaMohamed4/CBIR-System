import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from ultralytics import YOLO

class ColorFeatureExtractor:
    """Service for extracting color features from image regions"""
    
    def __init__(self, segmentation_model_path=None):
        # Priority: 1. Argument, 2. Env Var, 3. Default (which may trigger download)
        if segmentation_model_path is None:
            segmentation_model_path = os.getenv('SEG_MODEL_PATH', 'yolov8n-seg.pt')
            
        print(f"[*] ColorFeatureExtractor loading model: {segmentation_model_path}")
        self.seg_model = YOLO(segmentation_model_path)
    
    def extract_color_features(self, roi, use_segmentation=True, object_class=None):
        if use_segmentation:
            masked_roi = self._apply_segmentation_mask(roi, object_class)
            return self._extract_features_from_roi(masked_roi)
        return self._extract_features_from_roi(roi)
    
    def _apply_segmentation_mask(self, image, object_class=None):
        try:
            results = self.seg_model(image, verbose=False)
            if len(results) == 0 or results[0].masks is None:
                return image
            
            result = results[0]
            target_mask = None
            if object_class:
                for i, cls in enumerate(result.boxes.cls):
                    if result.names[int(cls)] == object_class:
                        target_mask = result.masks.data[i].cpu().numpy()
                        break
            else:
                target_mask = result.masks.data[0].cpu().numpy()
            
            if target_mask is None: return image
            
            mask = cv2.resize(target_mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
            masked_image = image.copy()
            masked_image[mask == 0] = 0
            return masked_image
        except Exception as e:
            print(f"Segmentation failed: {e}")
            return image

    def _extract_features_from_roi(self, roi):
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        non_black_mask = np.any(rgb != [0, 0, 0], axis=-1)
        
        if not np.any(non_black_mask):
            return self._get_empty_features()
        
        rgb_filtered = rgb[non_black_mask]
        hsv_filtered = hsv[non_black_mask]
        
        # Histograms
        hist_r = cv2.calcHist([rgb_filtered.reshape(-1, 1, 3)], [0], None, [16], [0, 256]).flatten()
        hist_g = cv2.calcHist([rgb_filtered.reshape(-1, 1, 3)], [1], None, [16], [0, 256]).flatten()
        hist_b = cv2.calcHist([rgb_filtered.reshape(-1, 1, 3)], [2], None, [16], [0, 256]).flatten()
        
        hist_h = cv2.calcHist([hsv_filtered.reshape(-1, 1, 3)], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv_filtered.reshape(-1, 1, 3)], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv_filtered.reshape(-1, 1, 3)], [2], None, [16], [0, 256]).flatten()
        
        # Normalize
        norm = lambda x: (x / (x.sum() + 1e-7)).tolist()
        
        return {
            'hist_rgb': norm(hist_r) + norm(hist_g) + norm(hist_b),
            'hist_hsv': norm(hist_h) + norm(hist_s) + norm(hist_v),
            'mean_rgb': np.mean(rgb_filtered, axis=0).tolist(),
            'std_rgb': np.std(rgb_filtered, axis=0).tolist(),
            'dominant_colors': self._extract_dominant_colors(rgb_filtered)
        }
    
    def _extract_dominant_colors(self, rgb_pixels, n_colors=5):
        pixels = rgb_pixels[np.random.choice(len(rgb_pixels), min(5000, len(rgb_pixels)), replace=False)]
        try:
            kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), n_init=3, max_iter=100)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            _, counts = np.unique(kmeans.labels_, return_counts=True)
            percentages = counts / counts.sum() * 100
            
            dominant_colors = []
            for idx in np.argsort(percentages)[::-1]:
                c = colors[idx].tolist()
                dominant_colors.append({
                    'rgb': c,
                    'hex': '#{:02x}{:02x}{:02x}'.format(*c),
                    'percentage': round(float(percentages[idx]), 2)
                })
            return dominant_colors
        except:
            m = np.mean(pixels, axis=0).astype(int).tolist()
            return [{'rgb': m, 'hex': '#{:02x}{:02x}{:02x}'.format(*m), 'percentage': 100.0}]
    
    def _get_empty_features(self):
        return {'hist_rgb': [0.0]*48, 'hist_hsv': [0.0]*48, 'mean_rgb': [0,0,0], 'std_rgb': [0,0,0], 'dominant_colors': []}