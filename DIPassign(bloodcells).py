import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMAGE_PATH = r"C:\Users\User\Documents\Uni\SEMESTER 5\DIGITAL IMAGE PROCESSING\final assignment\image20.jpg"
# Separation factor: 0.4 provides better sensitivity for small/touching cells
SEPARATION_FACTOR = 0.4  

def process_blood_cell_analysis(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return

    # 2. Image Processing Pipeline (Complexity 20%)
    # Convert to grayscale and invert colors
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    
    # Global thresholding using Otsu's method
    ret, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological Opening to remove background noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Watershed Segmentation to separate overlapping cells
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, SEPARATION_FACTOR * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Identify unknown regions for watershed markers
    unknown = cv2.subtract(cv2.dilate(opening, kernel, iterations=3), sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # 3. Morphological Measurement & Labeling
    cell_areas = []
    result_img = img.copy()
    unique_markers = np.unique(markers)
    
    for marker in unique_markers:
        if marker <= 1: continue # Skip background and boundaries
        
        # Isolate individual cell to calculate area
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == marker] = 255
        
        # Area calculation in square pixels
        area = np.sum(mask == 255)
        cell_areas.append(area)
        
        # Calculate centroid for text placement
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Apply offset to prevent text clipping at edges
            t_x = cX + 10 if cX < 20 else cX - 10
            t_y = cY + 10 if cY < 20 else cY
            cv2.putText(result_img, str(len(cell_areas)), (t_x, t_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 4. Result Presentation (Presentation 20%)
    plt.figure(figsize=(20, 10))
    
    # Panel 1: Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("1. Input Original Image")
    plt.axis('off')

    # Panel 2: Binary Mask
    plt.subplot(2, 2, 2)
    plt.imshow(opening, cmap='gray')
    plt.title("2. Segmentation Mask (Otsu + Morphology)")
    plt.axis('off')

    # Panel 3: Numbered Results
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"3. Detection Results: {len(cell_areas)} Cells Found")
    plt.axis('off')

    # Panel 4: Diagnostic Histogram (Relatability 40%)
    plt.subplot(2, 2, 4)
    n, bins, patches = plt.hist(cell_areas, bins=15, color='salmon', edgecolor='black', alpha=0.7)
    
    # Visual Diagnostic Thresholds
    plt.axvline(450, color='red', linestyle='dashed', linewidth=2, label='Microcytic Threshold')
    plt.axvline(950, color='blue', linestyle='dashed', linewidth=2, label='Macrocytic Threshold')
    
    # Labeling diagnostic zones
    plt.text(200, plt.ylim()[1]*0.8, 'Iron Def.\n(Micro)', color='red', fontweight='bold')
    plt.text(1050, plt.ylim()[1]*0.8, 'B12 Def.\n(Macro)', color='blue', fontweight='bold')
    plt.text(600, plt.ylim()[1]*0.9, 'Normal Range', color='green', fontweight='bold')

    plt.title("4. Size Distribution & Diagnostic Analysis")
    plt.xlabel("Cell Area (Pixels)")
    plt.ylabel("Frequency (Count)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_blood_cell_analysis(IMAGE_PATH)