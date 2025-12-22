import cv2
from .color_features import ColorFeatureExtractor
from .texture_features import TextureFeatureExtractor
from .shape_features import ShapeFeatureExtractor

class FeatureExtractionService:
    """Main service that coordinates all feature extraction"""
    
    def __init__(self, seg_model_path=None):
        # Pass the segmentation model path to color extractor to avoid download
        self.color_extractor = ColorFeatureExtractor(segmentation_model_path=seg_model_path)
        self.texture_extractor = TextureFeatureExtractor()
        self.shape_extractor = ShapeFeatureExtractor()
    
    def extract_all_features(self, image_path, bbox):
        img = cv2.imread(str(image_path))
        if img is None: return None
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0: return None
        
        return {
            'color': self.color_extractor.extract_color_features(roi),
            'texture_tamura': self.texture_extractor.extract_tamura_features(roi),
            'texture_gabor': self.texture_extractor.extract_gabor_features(roi),
            'texture_lbp': self.texture_extractor.extract_lbp_features(roi),
            'shape_hu': self.shape_extractor.extract_hu_moments(roi),
            'shape_hog': self.shape_extractor.extract_hog_features(roi),
            'shape_contour': self.shape_extractor.extract_contour_orientation_histogram(roi)
        }
    
    def format_features_for_display(self, features):
        if not features: return None
        return {
            'color': {
                'dominant_colors': features['color'].get('dominant_colors', []),
                'mean_rgb': features['color'].get('mean_rgb', []),
                'std_rgb': features['color'].get('std_rgb', []),
                'histogram_rgb': features['color'].get('hist_rgb', []),
                'histogram_hsv': features['color'].get('hist_hsv', [])
            },
            'texture': {
                'tamura_coarseness': features['texture_tamura'].get('coarseness', 0),
                'tamura_contrast': features['texture_tamura'].get('contrast', 0),
                'tamura_directionality': features['texture_tamura'].get('directionality', 0),
                'gabor_features_count': len(features['texture_gabor'].get('gabor_responses', [])),
                'lbp_histogram': features.get('texture_lbp', {}).get('lbp_hist', []),
                'lbp_mean': features.get('texture_lbp', {}).get('lbp_mean', 0),
                'lbp_std': features.get('texture_lbp', {}).get('lbp_std', 0)
            },
            'shape': {
                'hu_moments': features['shape_hu'].get('hu_moments', []),
                'hog_features_count': len(features['shape_hog'].get('hog', [])),
                'contour_orientation_hist': features.get('shape_contour', {}).get('orientation_hist', []),
                'main_orientation': features.get('shape_contour', {}).get('main_orientation', 0),
                'orientation_variance': features.get('shape_contour', {}).get('orientation_variance', 0)
            }
        }