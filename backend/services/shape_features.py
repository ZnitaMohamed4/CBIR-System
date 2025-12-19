# /home/muhammed/Documents/SmartGallery/backend/services/shape_features.py
"""
Shape Feature Extraction Service
Extracts shape-based features from image regions including:
- Hu moments (7 invariant moments)
- HOG (Histogram of Oriented Gradients)
- Contour orientation histogram
"""

import cv2
import numpy as np
from skimage.feature import hog


class ShapeFeatureExtractor:
    """Service for extracting shape features from image regions"""
    
    def __init__(self):
        pass
    
    def extract_hu_moments(self, roi):
        """
        Extract Hu moments from object contour
        
        Args:
            roi: Image region (numpy array in BGR format)
            
        Returns:
            Dictionary with Hu moments
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Log transform for better scale invariance
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            return {'hu_moments': hu_moments.tolist()}
        else:
            return {'hu_moments': [0.0] * 7}
    
    def extract_hog_features(self, roi):
        """
        Extract HOG (Histogram of Oriented Gradients) features
        
        Args:
            roi: Image region (numpy array in BGR format)
            
        Returns:
            Dictionary with HOG features
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 64))
        
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=False
        )
        
        return {'hog': features.tolist()}
    
    def extract_contour_orientation_histogram(self, roi):
        """
        Extract contour orientation histogram
        
        Args:
            roi: Image region (numpy array in BGR format)
            
        Returns:
            Dictionary with contour orientation features
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'orientation_hist': [0.0] * 18,
                'main_orientation': 0.0,
                'orientation_variance': 0.0
            }
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 3:
            return {
                'orientation_hist': [0.0] * 18,
                'main_orientation': 0.0,
                'orientation_variance': 0.0
            }
        
        orientations = []
        for i in range(len(largest_contour)):
            p1 = largest_contour[i][0]
            p2 = largest_contour[(i + 1) % len(largest_contour)][0]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            if dx != 0 or dy != 0:
                angle = np.arctan2(dy, dx)
                orientations.append(angle)
        
        if not orientations:
            return {
                'orientation_hist': [0.0] * 18,
                'main_orientation': 0.0,
                'orientation_variance': 0.0
            }
        
        orientations = np.array(orientations)
        normalized_angles = (orientations + np.pi) % np.pi
        
        hist, _ = np.histogram(normalized_angles, bins=18, range=(0, np.pi))
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)
        
        main_bin = np.argmax(hist)
        main_orientation = float(main_bin * 10)
        orientation_variance = float(np.var(normalized_angles))
        
        return {
            'orientation_hist': hist.tolist(),
            'main_orientation': main_orientation,
            'orientation_variance': orientation_variance
        }