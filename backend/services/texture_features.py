# /home/muhammed/Documents/SmartGallery/backend/services/texture_features.py
"""
Texture Feature Extraction Service
Extracts texture-based features from image regions including:
- Tamura descriptors (coarseness, contrast, directionality)
- Gabor filter responses
- Local Binary Patterns (LBP)
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage
from scipy.signal import convolve2d


class TextureFeatureExtractor:
    """Service for extracting texture features from image regions"""
    
    def __init__(self):
        self.gabor_kernels = self._prepare_gabor_kernels()
    
    def _prepare_gabor_kernels(self):
        """Prepare Gabor filter bank"""
        kernels = []
        # 4 orientations, 2 frequencies
        for theta in range(4):
            theta = theta / 4. * np.pi
            for frequency in (0.1, 0.2):
                kernel = np.real(gabor_kernel(frequency, theta=theta))
                kernels.append(kernel)
        return kernels
    
    def extract_tamura_features(self, roi):
        """
        Extract Tamura texture features
        
        Args:
            roi: Image region (numpy array in BGR format)
            
        Returns:
            Dictionary with Tamura features
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        coarseness = self._compute_coarseness(gray)
        contrast = self._compute_contrast(gray)
        directionality = self._compute_directionality(gray)
        
        return {
            'coarseness': float(coarseness),
            'contrast': float(contrast),
            'directionality': float(directionality)
        }
    
    def _compute_coarseness(self, gray):
        """Compute Tamura coarseness"""
        h, w = gray.shape
        gray = gray.astype(np.float64)
        
        k_max = 5
        avg_windows = []
        
        for k in range(k_max):
            size = 2 ** k
            if size >= min(h, w) // 2:
                break
            kernel = np.ones((size, size)) / (size * size)
            avg = convolve2d(gray, kernel, mode='same', boundary='symm')
            avg_windows.append(avg)
        
        if not avg_windows:
            return 0.0
        
        s_best = np.zeros_like(gray)
        for k in range(len(avg_windows) - 1):
            diff_h = np.abs(avg_windows[k] - np.roll(avg_windows[k], 2**k, axis=1))
            diff_v = np.abs(avg_windows[k] - np.roll(avg_windows[k], 2**k, axis=0))
            diff = np.maximum(diff_h, diff_v)
            mask = diff > s_best
            s_best[mask] = diff[mask]
        
        return np.mean(s_best)
    
    def _compute_contrast(self, gray):
        """Compute Tamura contrast"""
        std = np.std(gray)
        kurtosis = np.mean((gray - np.mean(gray)) ** 4) / (std ** 4 + 1e-7)
        contrast = std / (kurtosis ** 0.25 + 1e-7)
        return contrast
    
    def _compute_directionality(self, gray):
        """Compute Tamura directionality"""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        threshold = np.percentile(magnitude, 75)
        significant = magnitude > threshold
        
        if not significant.any():
            return 0.0
        
        hist, _ = np.histogram(direction[significant], bins=16, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-7)
        
        directionality = -np.sum(hist * np.log(hist + 1e-7))
        return directionality
    
    def extract_gabor_features(self, roi):
        """
        Extract Gabor filter responses
        
        Args:
            roi: Image region (numpy array in BGR format)
            
        Returns:
            Dictionary with Gabor features
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64)
        
        features = []
        for kernel in self.gabor_kernels:
            filtered = ndimage.convolve(gray, kernel, mode='wrap')
            features.append(np.mean(filtered))
            features.append(np.std(filtered))
        
        return {'gabor_responses': features}
    
    def extract_lbp_features(self, roi):
        """
        Extract Local Binary Pattern texture features
        
        Args:
            roi: Image region (numpy array in BGR format)
            
        Returns:
            Dictionary with LBP features
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        radius = 1
        n_points = 8 * radius
        
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)
        
        return {
            'lbp_hist': hist.tolist(),
            'lbp_mean': float(np.mean(lbp)),
            'lbp_std': float(np.std(lbp))
        }