# test_segmentation.py
"""
Quick test script for YOLO segmentation model
Usage: python test_segmentation.py <image_path>
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

def test_segmentation(image_path):
    print(f"📸 Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Load segmentation model
    print("🤖 Loading YOLO segmentation model...")
    try:
        model = YOLO('yolov8n-seg.pt')  # Will auto-download if not exists
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run segmentation
    print("🔍 Running segmentation...")
    results = model(image, verbose=False)
    
    if len(results) == 0 or results[0].masks is None:
        print("⚠️ No objects detected with masks")
        return
    
    result = results[0]
    print(f"✅ Detected {len(result.boxes)} objects")
    
    # Display results
    for i, (box, mask) in enumerate(zip(result.boxes, result.masks.data)):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = result.names[cls]
        
        print(f"  [{i}] {class_name} - {conf*100:.1f}% confidence")
    
    # Create visualization
    print("🎨 Creating visualization...")
    
    # Original image
    vis_original = image.copy()
    
    # Segmented image (background removed)
    mask_combined = result.masks.data[0].cpu().numpy()  # First object
    mask_resized = cv2.resize(mask_combined, (image.shape[1], image.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    
    vis_segmented = image.copy()
    vis_segmented[mask_binary == 0] = 0  # Set background to black
    
    # Overlay visualization
    vis_overlay = result.plot()  # YOLO's built-in visualization
    
    # Save outputs
    cv2.imwrite('output_original.jpg', vis_original)
    cv2.imwrite('output_segmented.jpg', vis_segmented)
    cv2.imwrite('output_overlay.jpg', vis_overlay)
    
    print("\n✅ Results saved:")
    print("  📁 output_original.jpg - Original image")
    print("  📁 output_segmented.jpg - Background removed (first object)")
    print("  📁 output_overlay.jpg - Segmentation overlay")
    
    # Show preview (optional - comment out if running headless)
    try:
        cv2.imshow('Original', cv2.resize(vis_original, (600, 400)))
        cv2.imshow('Segmented', cv2.resize(vis_segmented, (600, 400)))
        cv2.imshow('Overlay', cv2.resize(vis_overlay, (600, 400)))
        print("\n👁️ Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("⚠️ Display not available (running headless)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_segmentation.py <image_path>")
        print("Example: python test_segmentation.py test_image.jpg")
        sys.exit(1)
    
    test_segmentation(sys.argv[1])