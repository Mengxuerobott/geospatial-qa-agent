import rasterio
import numpy as np
import cv2
import pandas as pd

def extract_image_features(tiff_path: str, tile_id: str) -> pd.DataFrame:
    """
    Reads an RGB TIFF and extracts image statistics to understand 
    environmental conditions that might cause model failures.
    """
    with rasterio.open(tiff_path) as src:
        # Read the image bands (assuming RGB)
        # rasterio reads as (channels, height, width)
        image_array = src.read()
        
        # Handle nodata values if they exist (ignore them in calculations)
        nodata = src.nodata
        if nodata is not None:
            valid_mask = image_array != nodata
            valid_pixels = image_array[valid_mask]
        else:
            valid_pixels = image_array

        # 1. Brightness: Mean pixel value (Are failures happening in dark shadows?)
        brightness = np.mean(valid_pixels)
        
        # 2. Contrast: Standard deviation of pixels (Are failures happening in washed-out areas?)
        contrast = np.std(valid_pixels)
        
        # 3. Edge Complexity / Texture: How "busy" is the image? 
        # Dense vegetation has high edge complexity; flat dirt has low complexity.
        # Convert to grayscale for edge detection (assuming band 1=R, 2=G, 3=B)
        if image_array.shape[0] >= 3:
            # Transpose to (height, width, channels) for OpenCV
            img_cv = np.transpose(image_array[:3, :, :], (1, 2, 0))
            # Convert to uint8 if it's not already (common for drone imagery)
            if img_cv.dtype != np.uint8:
                # Normalize to 0-255 safely
                img_cv = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            # Use Laplacian variance as a measure of texture/sharpness
            edge_complexity = cv2.Laplacian(gray, cv2.CV_64F).var()
        else:
            edge_complexity = 0.0 # Fallback if not RGB

    # Store results
    results = {
        "tile_id": tile_id,
        "mean_brightness": round(float(brightness), 2),
        "image_contrast": round(float(contrast), 2),
        "edge_complexity": round(float(edge_complexity), 2)
    }
    
    return pd.DataFrame([results])

# --- Test the function ---
if __name__ == "__main__":
    # Replace with an actual path to a drone TIFF on your machine
    # tiff_file = "../../data/tiffs/tile_001.tif"
    # df_features = extract_image_features(tiff_file, tile_id="tile_001")
    # print(df_features)
    print("Image extractor ready. Awaiting data paths to test.")