# /home/muhammed/Documents/SmartGallery/backend/services/similarity_search.py

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import normalize

class SimilaritySearchService:
    """Service for similarity search and feature database management"""
    
    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.database = self._load_database()
    
    def _load_database(self):
        """Load feature database from JSON"""
        if self.database_path.exists():
            with open(self.database_path, 'r') as f:
                return json.load(f)
        return {'images': {}, 'metadata': {'created': datetime.now().isoformat()}}
    
    def _save_database(self):
        """Save feature database to JSON"""
        self.database['metadata']['updated'] = datetime.now().isoformat()
        with open(self.database_path, 'w') as f:
            json.dump(self.database, f, indent=2)
    
    def save_detections(self, image_id, detections):
        """Save object detections for an image"""
        if image_id not in self.database['images']:
            self.database['images'][image_id] = {'detections': [], 'features': []}
        
        self.database['images'][image_id]['detections'] = detections
        self._save_database()
    
    def get_detections(self, image_id):
        """Get detections for an image"""
        if image_id in self.database['images']:
            return self.database['images'][image_id].get('detections', [])
        return None
    
    def save_features(self, image_id, object_id, features):
        """Save extracted features for an object"""
        if image_id not in self.database['images']:
            self.database['images'][image_id] = {'detections': [], 'features': []}
        
        # Ensure features list is long enough
        while len(self.database['images'][image_id]['features']) <= object_id:
            self.database['images'][image_id]['features'].append(None)
        
        self.database['images'][image_id]['features'][object_id] = features
        self._save_database()
    
    def get_features(self, image_id, object_id):
        """Get features for a specific object"""
        if image_id in self.database['images']:
            features_list = self.database['images'][image_id].get('features', [])
            if object_id < len(features_list):
                return features_list[object_id]
        return None
    
    def find_similar(self, query_features, query_class, top_k=10, weights=None, 
                 exclude_image_id=None, same_class_only=True, class_weight=0.8,
                 normalize_scores=True):
        """
        Find similar objects based on feature similarity with advanced normalization
        
        Args:
            query_features: Features of query object
            query_class: Class name of query object (e.g., "airplane")
            top_k: Number of results to return
            weights: Optional dict of feature weights (auto-adjusted by class if None)
            exclude_image_id: Image ID to exclude from results
            same_class_only: If True, only return objects of same class (RECOMMENDED)
            class_weight: Weight for class matching bonus (0.0 to 1.0)
            normalize_scores: If True, apply score normalization for better distribution
            
        Returns:
            List of similar objects with scores
        """
        # Get class-specific weights if not provided
        if weights is None:
            weights = self._get_class_weights(query_class)
        
        similarities = []
        
        for image_id, data in self.database['images'].items():
            # Skip excluded image
            if image_id == exclude_image_id:
                continue
            
            features_list = data.get('features', [])
            detections = data.get('detections', [])
            
            for obj_idx, target_features in enumerate(features_list):
                if not target_features:
                    continue
                
                # Get detection info
                detection_info = detections[obj_idx] if obj_idx < len(detections) else {}
                target_class = detection_info.get('class', 'unknown')
                
                # ✅ CLASS FILTERING - Only compare same classes!
                if same_class_only and target_class.lower() != query_class.lower():
                    continue  # Skip different classes entirely
                
                # Compute visual similarity
                visual_similarity = self._compute_similarity(
                    query_features, 
                    target_features, 
                    weights
                )
                
                # ✅ CLASS BONUS - If not filtering, give bonus for same class
                if not same_class_only:
                    if target_class.lower() == query_class.lower():
                        # Same class: boost similarity
                        final_similarity = visual_similarity * (1 - class_weight) + class_weight
                    else:
                        # Different class: reduce similarity
                        final_similarity = visual_similarity * (1 - class_weight)
                else:
                    # Already filtered by class, no need for bonus
                    final_similarity = visual_similarity
                
                # ✅ FIX: Clamp to [0, 1] range before storing
                final_similarity = max(0.0, min(1.0, final_similarity))
                visual_similarity = max(0.0, min(1.0, visual_similarity))
                
                similarities.append({
                    'image_id': image_id,
                    'object_id': obj_idx,
                    'similarity': final_similarity,
                    'visual_similarity': visual_similarity,
                    'class': target_class,
                    'confidence': detection_info.get('confidence', 0.0),
                    'bbox': detection_info.get('bbox', [])
                })
        
        # ✅ SCORE NORMALIZATION - Better distribution
        if normalize_scores and len(similarities) > 1:
            similarities = self._normalize_similarity_scores(similarities)
        
        # ✅ APPLY CLASS-SPECIFIC THRESHOLD
        min_threshold = self._get_class_threshold(query_class)
        similarities = [s for s in similarities if s['similarity'] >= min_threshold]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]

    def _get_class_weights(self, class_name):
        """
        Get feature weights optimized for specific object classes
        Different objects have different discriminative features
        """
        class_name = class_name.lower()
        
        # Class-specific weight profiles
        weight_profiles = {
            # Vehicles (shape and edges are most important)
            'car': {
                'color': 0.20,
                'texture_tamura': 0.10,
                'texture_gabor': 0.10,
                'texture_lbp': 0.05,
                'shape_hu': 0.15,
                'shape_hog': 0.30,
                'shape_contour': 0.10
            },
            'bicycle': {
                'color': 0.15,
                'texture_tamura': 0.10,
                'texture_gabor': 0.10,
                'texture_lbp': 0.05,
                'shape_hu': 0.20,
                'shape_hog': 0.30,
                'shape_contour': 0.10
            },
            'airplane': {
                'color': 0.15,
                'texture_tamura': 0.10,
                'texture_gabor': 0.10,
                'texture_lbp': 0.05,
                'shape_hu': 0.20,
                'shape_hog': 0.30,
                'shape_contour': 0.10
            },
            'boat': {
                'color': 0.20,
                'texture_tamura': 0.10,
                'texture_gabor': 0.10,
                'texture_lbp': 0.05,
                'shape_hu': 0.15,
                'shape_hog': 0.30,
                'shape_contour': 0.10
            },
            
            # Animals (texture and color are important)
            'dog': {
                'color': 0.25,
                'texture_tamura': 0.15,
                'texture_gabor': 0.15,
                'texture_lbp': 0.10,
                'shape_hu': 0.10,
                'shape_hog': 0.15,
                'shape_contour': 0.10
            },
            'cat': {
                'color': 0.25,
                'texture_tamura': 0.15,
                'texture_gabor': 0.15,
                'texture_lbp': 0.10,
                'shape_hu': 0.10,
                'shape_hog': 0.15,
                'shape_contour': 0.10
            },
            'horse': {
                'color': 0.25,
                'texture_tamura': 0.15,
                'texture_gabor': 0.15,
                'texture_lbp': 0.10,
                'shape_hu': 0.10,
                'shape_hog': 0.15,
                'shape_contour': 0.10
            },
            'bird': {
                'color': 0.30,
                'texture_tamura': 0.15,
                'texture_gabor': 0.15,
                'texture_lbp': 0.10,
                'shape_hu': 0.10,
                'shape_hog': 0.10,
                'shape_contour': 0.10
            },
            
            # People (shape is most important)
            'person': {
                'color': 0.15,
                'texture_tamura': 0.10,
                'texture_gabor': 0.10,
                'texture_lbp': 0.10,
                'shape_hu': 0.15,
                'shape_hog': 0.30,
                'shape_contour': 0.10
            },
            
            # Food (color and texture are critical)
            'pizza': {
                'color': 0.35,
                'texture_tamura': 0.20,
                'texture_gabor': 0.15,
                'texture_lbp': 0.15,
                'shape_hu': 0.05,
                'shape_hog': 0.05,
                'shape_contour': 0.05
            },
            'apple': {
                'color': 0.40,
                'texture_tamura': 0.15,
                'texture_gabor': 0.15,
                'texture_lbp': 0.10,
                'shape_hu': 0.10,
                'shape_hog': 0.05,
                'shape_contour': 0.05
            },
            
            # Objects (balanced approach)
            'bottle': {
                'color': 0.20,
                'texture_tamura': 0.15,
                'texture_gabor': 0.15,
                'texture_lbp': 0.10,
                'shape_hu': 0.15,
                'shape_hog': 0.15,
                'shape_contour': 0.10
            },
            'laptop': {
                'color': 0.20,
                'texture_tamura': 0.15,
                'texture_gabor': 0.15,
                'texture_lbp': 0.10,
                'shape_hu': 0.15,
                'shape_hog': 0.15,
                'shape_contour': 0.10
            },
            'umbrella': {
                'color': 0.30,
                'texture_tamura': 0.15,
                'texture_gabor': 0.10,
                'texture_lbp': 0.10,
                'shape_hu': 0.10,
                'shape_hog': 0.15,
                'shape_contour': 0.10
            },
            'traffic_light': {
                'color': 0.35,
                'texture_tamura': 0.10,
                'texture_gabor': 0.10,
                'texture_lbp': 0.05,
                'shape_hu': 0.15,
                'shape_hog': 0.15,
                'shape_contour': 0.10
            }
        }
        
        # Return class-specific weights or default balanced weights
        return weight_profiles.get(class_name, {
            'color': 0.25,
            'texture_tamura': 0.15,
            'texture_gabor': 0.15,
            'texture_lbp': 0.10,
            'shape_hu': 0.10,
            'shape_hog': 0.15,
            'shape_contour': 0.10
        })

    def _get_class_threshold(self, class_name):
        """
        Get minimum similarity threshold for a class
        Some classes need stricter matching than others
        """
        class_name = class_name.lower()
        
        thresholds = {
            # Vehicles - need distinct shape matching
            'car': 0.30,
            'bicycle': 0.30,
            'airplane': 0.30,
            'boat': 0.30,
            
            # Animals - can be more varied
            'dog': 0.25,
            'cat': 0.25,
            'horse': 0.25,
            'bird': 0.20,
            
            # People - highly varied
            'person': 0.20,
            
            # Food - texture/color focused
            'pizza': 0.25,
            'apple': 0.30,
            
            # Objects
            'bottle': 0.25,
            'laptop': 0.30,
            'umbrella': 0.25,
            'traffic_light': 0.35  # Very specific shape
        }
        
        return thresholds.get(class_name, 0.25)

    def _normalize_similarity_scores(self, similarities):
        """
        Normalize similarity scores for better distribution
        Uses min-max scaling with adaptive range compression
        """
        if len(similarities) <= 1:
            return similarities
        
        # Extract scores
        scores = np.array([s['similarity'] for s in similarities])
        
        # Find min and max
        min_score = scores.min()
        max_score = scores.max()
        
        # Avoid division by zero
        if max_score - min_score < 0.01:
            return similarities
        
        # Apply min-max normalization with range expansion
        # Map [min, max] to [0.3, 1.0] for better visual distinction
        normalized_scores = 0.3 + 0.7 * (scores - min_score) / (max_score - min_score)
        
        # Apply gentle power transformation to spread middle values
        # y = x^0.8 makes middle values more distinct
        normalized_scores = np.power(normalized_scores, 0.8)
        
        # Update similarities with normalized scores
        for i, similarity in enumerate(similarities):
            similarity['similarity'] = float(normalized_scores[i])
            # Keep visual_similarity as the original unmodified score
        
        return similarities
        
    def _compute_similarity(self, features1, features2, weights):
        """
        Compute weighted similarity between two feature sets
        Gracefully handles missing features by adjusting weights dynamically
        """
        total_similarity = 0.0
        total_weight = 0.0
        
        # Color similarity
        if 'color' in features1 and 'color' in features2:
            sim = self._color_similarity(features1['color'], features2['color'])
            w = weights.get('color', 0.3)
            total_similarity += sim * w
            total_weight += w
        
        # Tamura texture similarity
        if 'texture_tamura' in features1 and 'texture_tamura' in features2:
            sim = self._tamura_similarity(features1['texture_tamura'], features2['texture_tamura'])
            w = weights.get('texture_tamura', 0.2)
            total_similarity += sim * w
            total_weight += w
        
        # Gabor texture similarity
        if 'texture_gabor' in features1 and 'texture_gabor' in features2:
            sim = self._gabor_similarity(features1['texture_gabor'], features2['texture_gabor'])
            w = weights.get('texture_gabor', 0.2)
            total_similarity += sim * w
            total_weight += w
        
        # LBP texture similarity
        if 'texture_lbp' in features1 and 'texture_lbp' in features2:
            sim = self._lbp_similarity(features1['texture_lbp'], features2['texture_lbp'])
            w = weights.get('texture_lbp', 0.10)
            total_similarity += sim * w
            total_weight += w
        
        # Hu moments similarity
        if 'shape_hu' in features1 and 'shape_hu' in features2:
            sim = self._hu_similarity(features1['shape_hu'], features2['shape_hu'])
            w = weights.get('shape_hu', 0.15)
            total_similarity += sim * w
            total_weight += w
        
        # HOG similarity
        if 'shape_hog' in features1 and 'shape_hog' in features2:
            sim = self._hog_similarity(features1['shape_hog'], features2['shape_hog'])
            w = weights.get('shape_hog', 0.15)
            total_similarity += sim * w
            total_weight += w
        
        # Contour orientation similarity
        if 'shape_contour' in features1 and 'shape_contour' in features2:
            sim = self._contour_similarity(features1['shape_contour'], features2['shape_contour'])
            w = weights.get('shape_contour', 0.10)
            total_similarity += sim * w
            total_weight += w
        
        # Normalize by actual weight sum (handles missing features)
        if total_weight > 0:
            return total_similarity / total_weight
        
        # Fallback: if no features matched, return 0
        return 0.0
    
    def _color_similarity(self, color1, color2):
        """
        Compute color similarity using multiple metrics
        Combines histogram comparison + dominant color matching
        """
        # 1. RGB Histogram comparison (Chi-Square distance)
        hist1 = np.array(color1.get('hist_rgb', []))
        hist2 = np.array(color2.get('hist_rgb', []))
        
        hist_sim = 0.0
        if len(hist1) > 0 and len(hist2) > 0:
            # Chi-Square distance (more discriminative than intersection)
            epsilon = 1e-10
            chi_square = 0.5 * np.sum(
                ((hist1 - hist2) ** 2) / (hist1 + hist2 + epsilon)
            )
            # Convert distance to similarity (0=identical, higher=different)
            hist_sim = 1.0 / (1.0 + chi_square)
        
        # 2. Dominant color comparison
        dom_sim = self._dominant_color_similarity(color1, color2)
        
        # 3. Mean color distance
        mean1 = np.array(color1.get('mean_rgb', [0, 0, 0]))
        mean2 = np.array(color2.get('mean_rgb', [0, 0, 0]))
        mean_distance = np.linalg.norm(mean1 - mean2) / 441.67  # Normalize by max RGB distance (sqrt(255^2 * 3))
        mean_sim = 1.0 - mean_distance
        
        # Combine (70% histogram, 20% dominant colors, 10% mean)
        final_sim = 0.7 * hist_sim + 0.2 * dom_sim + 0.1 * mean_sim
        
        return float(max(0.0, min(1.0, final_sim)))

    def _dominant_color_similarity(self, color1, color2):
        """
        Compare dominant colors using weighted color distance
        Considers both color values and their percentages
        """
        dom1 = color1.get('dominant_colors', [])
        dom2 = color2.get('dominant_colors', [])
        
        if not dom1 or not dom2:
            return 0.0
        
        # Extract top 5 dominant colors
        colors1 = []
        weights1 = []
        for c in dom1[:5]:
            colors1.append(c['rgb'])
            weights1.append(c['percentage'] / 100.0)
        
        colors2 = []
        weights2 = []
        for c in dom2[:5]:
            colors2.append(c['rgb'])
            weights2.append(c['percentage'] / 100.0)
        
        colors1 = np.array(colors1)
        weights1 = np.array(weights1)
        colors2 = np.array(colors2)
        weights2 = np.array(weights2)
        
        # Compute weighted minimum color distance
        total_distance = 0.0
        for i, (c1, w1) in enumerate(zip(colors1, weights1)):
            # Find closest color in second set
            min_dist = float('inf')
            for c2 in colors2:
                # Euclidean distance in RGB space
                dist = np.linalg.norm(c1 - c2) / 441.67  # Normalize
                min_dist = min(min_dist, dist)
            total_distance += w1 * min_dist
        
        # Reverse: do the same from colors2 to colors1
        for i, (c2, w2) in enumerate(zip(colors2, weights2)):
            min_dist = float('inf')
            for c1 in colors1:
                dist = np.linalg.norm(c2 - c1) / 441.67
                min_dist = min(min_dist, dist)
            total_distance += w2 * min_dist
        
        # Average bidirectional distance
        total_distance /= 2.0
        
        similarity = 1.0 - total_distance
        return float(max(0.0, similarity))
    
    def _tamura_similarity(self, tamura1, tamura2):
        """
        Compute Tamura feature similarity with normalization
        Tamura features have different scales, so normalize them
        """
        # Extract features
        coarse1 = tamura1.get('coarseness', 0)
        contrast1 = tamura1.get('contrast', 0)
        directionality1 = tamura1.get('directionality', 0)
        
        coarse2 = tamura2.get('coarseness', 0)
        contrast2 = tamura2.get('contrast', 0)
        directionality2 = tamura2.get('directionality', 0)
        
        # Compute individual similarities with appropriate scaling
        # Coarseness: typically 0-10
        coarse_sim = 1.0 - min(abs(coarse1 - coarse2) / 10.0, 1.0)
        
        # Contrast: typically 0-100
        contrast_sim = 1.0 - min(abs(contrast1 - contrast2) / 100.0, 1.0)
        
        # Directionality: typically 0-5 (entropy-based)
        dir_sim = 1.0 - min(abs(directionality1 - directionality2) / 5.0, 1.0)
        
        # Weighted combination (coarseness is most important)
        final_sim = 0.5 * coarse_sim + 0.3 * contrast_sim + 0.2 * dir_sim
        
        return float(max(0.0, min(1.0, final_sim)))
    
    def _gabor_similarity(self, gabor1, gabor2):
        """Compute Gabor feature similarity"""
        features1 = np.array(gabor1.get('gabor_responses', []))
        features2 = np.array(gabor2.get('gabor_responses', []))
        
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Cosine similarity
        if np.linalg.norm(features1) > 0 and np.linalg.norm(features2) > 0:
            features1 = features1 / np.linalg.norm(features1)
            features2 = features2 / np.linalg.norm(features2)
            similarity = np.dot(features1, features2)
            return float(max(0, similarity))
        return 0.0
    
    def _hu_similarity(self, hu1, hu2):
        """
        Compute Hu moments similarity with improved distance metric
        Hu moments are log-transformed, so use absolute difference
        """
        moments1 = np.array(hu1.get('hu_moments', []))
        moments2 = np.array(hu2.get('hu_moments', []))
        
        if len(moments1) == 0 or len(moments2) == 0:
            return 0.0
        
        # Hu moments can have very different scales, so normalize each moment
        # Use weighted distance where earlier moments (more significant) have higher weight
        weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])  # 7 Hu moments
        
        # Absolute difference (since Hu moments are log-transformed)
        abs_diff = np.abs(moments1 - moments2)
        
        # Weighted Euclidean distance
        weighted_distance = np.sqrt(np.sum(weights * (abs_diff ** 2)))
        
        # Convert to similarity with adaptive scaling
        # Scale factor based on empirical observation (Hu moments typically range 0-20 after log transform)
        similarity = 1.0 / (1.0 + weighted_distance / 10.0)
        
        return float(max(0.0, min(1.0, similarity)))
    
    def _hog_similarity(self, hog1, hog2):
        """Compute HOG feature similarity"""
        features1 = np.array(hog1.get('hog', []))
        features2 = np.array(hog2.get('hog', []))
        
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Cosine similarity
        if np.linalg.norm(features1) > 0 and np.linalg.norm(features2) > 0:
            features1 = features1 / np.linalg.norm(features1)
            features2 = features2 / np.linalg.norm(features2)
            similarity = np.dot(features1, features2)
            return float(max(0, similarity))
        return 0.0
    
    def _lbp_similarity(self, lbp1, lbp2):
        """
        Compute LBP texture similarity using Chi-Square distance
        More discriminative than simple histogram intersection
        """
        hist1 = np.array(lbp1.get('lbp_hist', []))
        hist2 = np.array(lbp2.get('lbp_hist', []))
        
        if len(hist1) == 0 or len(hist2) == 0:
            return 0.0
        
        # Chi-Square distance for texture patterns
        epsilon = 1e-10
        chi_square = 0.5 * np.sum(
            ((hist1 - hist2) ** 2) / (hist1 + hist2 + epsilon)
        )
        
        # Convert to similarity
        similarity = 1.0 / (1.0 + chi_square)
        
        # Also consider mean and std for texture roughness
        mean1 = lbp1.get('lbp_mean', 0)
        mean2 = lbp2.get('lbp_mean', 0)
        std1 = lbp1.get('lbp_std', 0)
        std2 = lbp2.get('lbp_std', 0)
        
        # Normalize mean and std differences
        mean_diff = abs(mean1 - mean2) / 255.0  # Normalize by max LBP value
        std_diff = abs(std1 - std2) / 100.0  # Approximate normalization
        
        mean_std_sim = 1.0 - (mean_diff + std_diff) / 2.0
        
        # Combine: 80% histogram, 20% mean/std
        final_sim = 0.8 * similarity + 0.2 * mean_std_sim
        
        return float(max(0.0, min(1.0, final_sim)))

    def _contour_similarity(self, contour1, contour2):
        """
        Compute contour orientation similarity
        Uses both histogram and orientation statistics
        """
        hist1 = np.array(contour1.get('orientation_hist', []))
        hist2 = np.array(contour2.get('orientation_hist', []))
        
        if len(hist1) == 0 or len(hist2) == 0:
            return 0.0
        
        # Chi-Square distance for orientation histogram
        epsilon = 1e-10
        chi_square = 0.5 * np.sum(
            ((hist1 - hist2) ** 2) / (hist1 + hist2 + epsilon)
        )
        hist_sim = 1.0 / (1.0 + chi_square)
        
        # Compare main orientation angles
        main1 = contour1.get('main_orientation', 0)
        main2 = contour2.get('main_orientation', 0)
        
        # Angular difference (wrap around at 180 degrees)
        angle_diff = abs(main1 - main2)
        if angle_diff > 90:  # Wrap around
            angle_diff = 180 - angle_diff
        
        angle_sim = 1.0 - (angle_diff / 90.0)  # Normalize to [0, 1]
        
        # Compare orientation variance (how spread out the orientations are)
        var1 = contour1.get('orientation_variance', 0)
        var2 = contour2.get('orientation_variance', 0)
        var_diff = abs(var1 - var2)
        var_sim = 1.0 / (1.0 + var_diff)
        
        # Combine: 60% histogram, 25% main angle, 15% variance
        final_sim = 0.6 * hist_sim + 0.25 * angle_sim + 0.15 * var_sim
        
        return float(max(0.0, min(1.0, final_sim)))

    def delete_image_data(self, image_id):
        """Delete all data for an image from the database"""
        if image_id in self.database['images']:
            del self.database['images'][image_id]
            self._save_database()
            return True
        return False

    def cleanup_missing_images(self, existing_image_ids):
        """
        Remove database entries for images that no longer exist physically
        
        Args:
            existing_image_ids: Set of image IDs that still have physical files
            
        Returns:
            List of cleaned up image IDs
        """
        orphaned_ids = []
        for image_id in list(self.database['images'].keys()):
            if image_id not in existing_image_ids:
                orphaned_ids.append(image_id)
        
        # Remove orphaned entries
        for image_id in orphaned_ids:
            del self.database['images'][image_id]
        
        if orphaned_ids:
            self._save_database()
        
        return orphaned_ids
    
    def get_statistics(self):
        """Get database statistics"""
        total_images = len(self.database['images'])
        total_objects = sum(
            len(data.get('detections', []))
            for data in self.database['images'].values()
        )
        total_features = sum(
            len([f for f in data.get('features', []) if f is not None])
            for data in self.database['images'].values()
        )
        
        # Class distribution
        class_counts = {}
        for data in self.database['images'].values():
            for detection in data.get('detections', []):
                class_name = detection.get('class', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_images': total_images,
            'total_objects': total_objects,
            'total_features_extracted': total_features,
            'class_distribution': class_counts
        }