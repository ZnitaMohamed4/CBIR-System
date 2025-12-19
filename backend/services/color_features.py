"""
Color Feature Extraction Service with Background Removal
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO


class ColorFeatureExtractor:
    """Service for extracting color features from image regions"""
    
    def __init__(self, segmentation_model_path='yolov8n-seg.pt'):
        """
        Initialize with segmentation model
        
        Args:
            segmentation_model_path: Path to YOLO segmentation model
        """
        self.seg_model = YOLO(segmentation_model_path)
    
    def extract_color_features(self, roi, use_segmentation=True, object_class=None):
        """
        Extract all color features from a region of interest
        
        Args:
            roi: Image region (numpy array in BGR format)
            use_segmentation: Whether to remove background using segmentation
            object_class: Specific object class to extract (e.g., 'bear', 'person')
                         If None, uses the first detected object
            
        Returns:
            Dictionary with color features
        """
        # Apply segmentation to remove background if requested
        if use_segmentation:
            roi = self._apply_segmentation_mask(roi, object_class)
            if roi is None:
                # Fallback to original if segmentation fails
                return self._extract_features_from_roi(roi)
        
        return self._extract_features_from_roi(roi)
    
    def _apply_segmentation_mask(self, image, object_class=None):
        """
        Apply segmentation to isolate object from background
        
        Args:
            image: Input image
            object_class: Specific object class to extract
            
        Returns:
            Masked image with background removed (set to black)
        """
        try:
            # Run segmentation
            results = self.seg_model(image, verbose=False)
            
            if len(results) == 0 or results[0].masks is None:
                return image  # No objects detected, return original
            
            result = results[0]
            
            # Find the target object
            target_mask = None
            if object_class:
                # Look for specific class
                for i, cls in enumerate(result.boxes.cls):
                    if result.names[int(cls)] == object_class:
                        target_mask = result.masks.data[i].cpu().numpy()
                        break
            else:
                # Use the first (largest) detected object
                target_mask = result.masks.data[0].cpu().numpy()
            
            if target_mask is None:
                return image  # Target not found
            
            # Resize mask to image dimensions
            mask = cv2.resize(target_mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
            
            # Apply mask - set background to black
            masked_image = image.copy()
            masked_image[mask == 0] = 0


            # DEBUG: Save masked image to verify segmentation
            # cv2.imwrite('debug_masked_output.jpg', masked_image)  # ← ADD THIS LINE

            
            return masked_image
            
        except Exception as e:
            print(f"Segmentation failed: {e}")
            return image  # Fallback to original
    
    def _extract_features_from_roi(self, roi):
        """
        Extract color features from ROI
        
        Args:
            roi: Image region (numpy array in BGR format)
            
        Returns:
            Dictionary with color features
        """
        # Convert to RGB and HSV
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Filter out black pixels (background) for feature extraction
        non_black_mask = np.any(rgb != [0, 0, 0], axis=-1)
        
        if not np.any(non_black_mask):
            # All pixels are black (no object), return empty features
            return self._get_empty_features()
        
        # Extract only non-black pixels for histogram and moments
        rgb_filtered = rgb[non_black_mask]
        hsv_filtered = hsv[non_black_mask]
        
        # Calculate histograms with 16 bins per channel
        hist_r = cv2.calcHist([rgb_filtered.reshape(-1, 1, 3)], [0], None, [16], [0, 256]).flatten()
        hist_g = cv2.calcHist([rgb_filtered.reshape(-1, 1, 3)], [1], None, [16], [0, 256]).flatten()
        hist_b = cv2.calcHist([rgb_filtered.reshape(-1, 1, 3)], [2], None, [16], [0, 256]).flatten()
        hist_h = cv2.calcHist([hsv_filtered.reshape(-1, 1, 3)], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv_filtered.reshape(-1, 1, 3)], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv_filtered.reshape(-1, 1, 3)], [2], None, [16], [0, 256]).flatten()
        
        # Normalize histograms
        hist_r = hist_r / (hist_r.sum() + 1e-7)
        hist_g = hist_g / (hist_g.sum() + 1e-7)
        hist_b = hist_b / (hist_b.sum() + 1e-7)
        hist_h = hist_h / (hist_h.sum() + 1e-7)
        hist_s = hist_s / (hist_s.sum() + 1e-7)
        hist_v = hist_v / (hist_v.sum() + 1e-7)
        
        # Compute color moments on filtered pixels
        mean_rgb = np.mean(rgb_filtered, axis=0)
        std_rgb = np.std(rgb_filtered, axis=0)
        
        # Extract dominant colors using K-Means clustering
        dominant_colors = self._extract_dominant_colors(rgb_filtered, n_colors=5)
        
        return {
            'hist_rgb': np.concatenate([hist_r, hist_g, hist_b]).tolist(),
            'hist_hsv': np.concatenate([hist_h, hist_s, hist_v]).tolist(),
            'mean_rgb': mean_rgb.tolist(),
            'std_rgb': std_rgb.tolist(),
            'dominant_colors': dominant_colors
        }
    
    def _extract_dominant_colors(self, rgb_pixels, n_colors=5):
        """
        Extract dominant colors using K-Means clustering
        
        Args:
            rgb_pixels: RGB pixel array (N x 3)
            n_colors: Number of dominant colors to extract
            
        Returns:
            List of dominant colors with their percentages
        """
        # Use a subset of pixels for speed (max 5000 pixels)
        if len(rgb_pixels) > 5000:
            indices = np.random.choice(len(rgb_pixels), 5000, replace=False)
            pixels = rgb_pixels[indices]
        else:
            pixels = rgb_pixels
        
        # Apply K-Means clustering
        try:
            kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), n_init=3, max_iter=100, random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Calculate percentage of each color
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            percentages = counts / counts.sum() * 100
            
            # Sort by percentage (most dominant first)
            sorted_indices = np.argsort(percentages)[::-1]
            
            dominant_colors = []
            for idx in sorted_indices:
                color = colors[idx].tolist()
                percentage = float(percentages[idx])
                # Convert to hex for display
                hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                dominant_colors.append({
                    'rgb': color,
                    'hex': hex_color,
                    'percentage': round(percentage, 2)
                })
            
            return dominant_colors
        except Exception as e:
            # Fallback: return mean color only
            mean_color = np.mean(pixels, axis=0).astype(int).tolist()
            return [{
                'rgb': mean_color,
                'hex': '#{:02x}{:02x}{:02x}'.format(*mean_color),
                'percentage': 100.0
            }]
    
    def _get_empty_features(self):
        """Return empty feature set when no object is detected"""
        return {
            'hist_rgb': [0.0] * 48,
            'hist_hsv': [0.0] * 48,
            'mean_rgb': [0, 0, 0],
            'std_rgb': [0, 0, 0],
            'dominant_colors': []
        }