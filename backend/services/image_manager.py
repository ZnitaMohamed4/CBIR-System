# /home/muhammed/Documents/SmartGallery/backend/services/image_manager.py

import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from datetime import datetime

class ImageManager:
    """Service for managing image files"""
    
    def __init__(self, upload_folder):
        self.upload_folder = Path(upload_folder)
        self.upload_folder.mkdir(parents=True, exist_ok=True)

    def _compute_image_hash(self, image_path):
        """Compute perceptual hash of image for duplicate detection"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Resize to small size for quick comparison
        img_small = cv2.resize(img, (8, 8))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Compute average hash
        avg = img_gray.mean()
        hash_bits = (img_gray > avg).flatten()
        
        # Convert to hex string
        hash_bytes = np.packbits(hash_bits)
        import hashlib
        return hashlib.md5(hash_bytes.tobytes()).hexdigest()

    def _find_duplicate(self, new_image_path):
        """Check if image already exists in database"""
        new_hash = self._compute_image_hash(new_image_path)
        if not new_hash:
            return None
        
        # Check all existing images
        for filepath in self.upload_folder.glob('*'):
            if filepath.is_file() and filepath != Path(new_image_path):
                existing_hash = self._compute_image_hash(filepath)
                if existing_hash == new_hash:
                    # Found duplicate
                    return {
                        'image_id': filepath.stem,
                        'filename': filepath.name
                    }
        return None
    
    def save_image(self, file):
        """Save uploaded image file (with duplicate detection)"""
        filename = secure_filename(file.filename)
        image_id = str(uuid.uuid4())
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
        new_filename = f"{image_id}.{ext}"
        filepath = self.upload_folder / new_filename
        
        # Save temporarily
        file.save(str(filepath))
        
        # ✅ CHECK FOR DUPLICATES
        duplicate = self._find_duplicate(filepath)
        if duplicate:
            # Remove the temporary file
            filepath.unlink()
            
            # Return existing image info with duplicate flag
            existing_img = cv2.imread(str(self.upload_folder / duplicate['filename']))
            height, width = existing_img.shape[:2] if existing_img is not None else (0, 0)
            
            return {
                'image_id': duplicate['image_id'],
                'filename': duplicate['filename'],
                'original_filename': filename,
                'path': str(self.upload_folder / duplicate['filename']),
                'width': int(width),
                'height': int(height),
                'uploaded_at': datetime.now().isoformat(),
                'duplicate': True,  # ← Flag indicating it's a duplicate
                'message': 'This image already exists in the database'
            }
        
        # Not a duplicate, keep the new file
        img = cv2.imread(str(filepath))
        height, width = img.shape[:2] if img is not None else (0, 0)
        
        return {
            'image_id': image_id,
            'filename': new_filename,
            'original_filename': filename,
            'path': str(filepath),
            'width': int(width),
            'height': int(height),
            'uploaded_at': datetime.now().isoformat(),
            'duplicate': False
        }
    
    def get_all_images(self):
        """Get list of all images"""
        images = []
        for filepath in self.upload_folder.glob('*'):
            if filepath.is_file() and filepath.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                image_id = filepath.stem
                img = cv2.imread(str(filepath))
                height, width = img.shape[:2] if img is not None else (0, 0)
                
                images.append({
                    'image_id': image_id,
                    'filename': filepath.name,
                    'width': int(width),
                    'height': int(height),
                    'url': f'/api/images/file/{filepath.name}'
                })
        return images
    
    def get_image(self, image_id):
        """Get single image info"""
        for filepath in self.upload_folder.glob(f'{image_id}.*'):
            if filepath.is_file():
                img = cv2.imread(str(filepath))
                height, width = img.shape[:2] if img is not None else (0, 0)
                
                return {
                    'image_id': image_id,
                    'filename': filepath.name,
                    'width': int(width),
                    'height': int(height),
                    'url': f'/api/images/file/{filepath.name}'
                }
        return None
    
    def get_image_path(self, image_id):
        """Get full path to image file"""
        for filepath in self.upload_folder.glob(f'{image_id}.*'):
            if filepath.is_file():
                return str(filepath)
        return None
    
    def delete_image(self, image_id):
        """Delete an image and its database entries"""
        from services.similarity_search import SimilaritySearchService
        from pathlib import Path
        
        # Delete physical file
        deleted = False
        for filepath in self.upload_folder.glob(f'{image_id}.*'):
            if filepath.is_file():
                filepath.unlink()
                deleted = True
        
        # Delete from features database
        if deleted:
            db_path = Path(__file__).parent.parent / 'database' / 'features.json'
            similarity_service = SimilaritySearchService(str(db_path))
            similarity_service.delete_image_data(image_id)
        
        return deleted
    
    def delete_images(self, image_ids):
        """Delete multiple images"""
        results = []
        for image_id in image_ids:
            success = self.delete_image(image_id)
            results.append({'image_id': image_id, 'deleted': success})
        return results
    
    def transform_image(self, image_id, transform_type, params):
        """
        Apply transformation to create new image
        
        Args:
            image_id: Source image ID
            transform_type: 'crop', 'resize', 'rotate', 'flip'
            params: Transform parameters
            
        Returns:
            New image info
        """
        source_path = self.get_image_path(image_id)
        if not source_path:
            return {'error': 'Source image not found'}
        
        img = cv2.imread(source_path)
        if img is None:
            return {'error': 'Failed to load image'}
        
        # Apply transformation
        if transform_type == 'crop':
            x, y, w, h = params.get('x', 0), params.get('y', 0), params.get('width'), params.get('height')
            if w and h:
                img = img[y:y+h, x:x+w]
            else:
                return {'error': 'Crop parameters (x, y, width, height) required'}
        
        elif transform_type == 'resize':
            width, height = params.get('width'), params.get('height')
            if width and height:
                img = cv2.resize(img, (width, height))
            else:
                return {'error': 'Resize parameters (width, height) required'}
        
        elif transform_type == 'scale':
            # Scale with aspect ratio preservation
            scale_factor = params.get('scale', 1.0)
            if scale_factor <= 0:
                return {'error': 'Scale factor must be positive'}
            h, w = img.shape[:2]
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        elif transform_type == 'resize_keep_aspect':
            # Resize keeping aspect ratio (fit within max dimensions)
            max_width = params.get('max_width')
            max_height = params.get('max_height')
            if not max_width and not max_height:
                return {'error': 'max_width or max_height required'}
            
            h, w = img.shape[:2]
            aspect = w / h
            
            if max_width and max_height:
                # Fit within both constraints
                if w / max_width > h / max_height:
                    new_w = max_width
                    new_h = int(max_width / aspect)
                else:
                    new_h = max_height
                    new_w = int(max_height * aspect)
            elif max_width:
                new_w = max_width
                new_h = int(max_width / aspect)
            else:
                new_h = max_height
                new_w = int(max_height * aspect)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        elif transform_type == 'rotate':
            angle = params.get('angle', 0)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h))
        
        elif transform_type == 'flip':
            flip_code = params.get('direction', 1)  # 1=horizontal, 0=vertical, -1=both
            img = cv2.flip(img, flip_code)
        
        else:
            return {'error': f'Unknown transform type: {transform_type}'}
        
        # Save new image
        new_id = str(uuid.uuid4())
        ext = Path(source_path).suffix
        new_filename = f"{new_id}{ext}"
        new_path = self.upload_folder / new_filename
        
        cv2.imwrite(str(new_path), img)
        
        height, width = img.shape[:2]
        
        return {
            'image_id': new_id,
            'filename': new_filename,
            'source_image_id': image_id,
            'transform': transform_type,
            'width': int(width),
            'height': int(height),
            'created_at': datetime.now().isoformat()
        }