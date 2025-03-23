import cv2
import numpy as np
import os

def remove_shadows(image):
    """Removes shadows and slightly brightens the image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Slightly increase brightness
    l = cv2.add(l, 20)
    l = np.clip(l, 0, 255)
    
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result

def enhance_receipt(image):
    """Enhances the receipt by slightly increasing contrast and sharpness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Slightly increase contrast
    alpha = 1.2  # Contrast control
    beta = 15    # Brightness control
    contrast_enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Sharpen image
    sharpen_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)
    
    return sharpened

def kmeans_segmentation(image):
    """Uses K-means clustering to isolate the receipt from the background."""
    reshaped = image.reshape((-1, 3))
    reshaped = np.float32(reshaped)
 
    k = 2
    _, labels, centers = cv2.kmeans(reshaped, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(image.shape)
 
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    return binary

def find_largest_bright_contour(binary, original):
    """Finds the largest bright contour, which should be the receipt."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000 or area > original.shape[0] * original.shape[1] * 0.9:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx
        
        hull = cv2.convexHull(cnt)
        hull_approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        if len(hull_approx) == 4:
            return hull_approx
 
    return None

def crop_receipt(img, receipt_contour):
    """Performs perspective transformation to properly align and crop the receipt."""
    pts = np.array([point[0] for point in receipt_contour], dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    width = int(np.linalg.norm(rect[1] - rect[0]))
    height = int(np.linalg.norm(rect[2] - rect[1]))
    
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    cropped_receipt = cv2.warpPerspective(img, M, (width, height))
    
    return cropped_receipt

def process_images(input_folder, output_folder):
    """Processes all images in the input folder and saves enhanced results in the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Error: Image {filename} not found!")
                continue
            
            img = remove_shadows(img)
            binary_kmeans = kmeans_segmentation(img)
            receipt_contour = find_largest_bright_contour(binary_kmeans, img)
            
            if receipt_contour is None:
                print(f"Error: Unable to detect a proper receipt contour in {filename}.")
                continue
            
            cropped_receipt = crop_receipt(img, receipt_contour)
            enhanced_receipt = enhance_receipt(cropped_receipt)
            
            cv2.imwrite(os.path.join(output_folder, f"{filename}_enhanced.jpg"), enhanced_receipt)
            
            print(f"Processed: {filename}")

# Set input and output folders
input_folder = "/Users/eloise/Downloads/large-receipt-image-dataset-SRD" 
output_folder = "/Users/eloise/Downloads/processed_receipts"
process_images(input_folder, output_folder)
