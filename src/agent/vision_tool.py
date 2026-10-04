import os
import base64
import rasterio
import cv2
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

# --- Robust Dotenv Loading ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
env_path = os.path.join(root_dir, ".env")
load_dotenv(dotenv_path=env_path)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(f"CRITICAL: OPENAI_API_KEY not found. Checked path: {env_path}")

def _summarize_b64(outputs) -> dict:
    """
    Keep the base64 payload out of the trace; log only its size.
    LangSmith hands this the raw return value (a str) for single-value returns,
    but a dict when the traced function returns one -- handle both.
    """
    b64 = outputs.get("output", "") if isinstance(outputs, dict) else (outputs or "")
    return {"base64_chars": len(b64)}

@traceable(run_type="tool", name="encode_and_resize_tiff", process_outputs=_summarize_b64)
def encode_and_resize_tiff(tiff_path: str, max_size: int = 1024) -> str:
    """
    Reads a massive drone TIFF, extracts RGB, resizes it to a safe dimension, 
    compresses it to JPEG, and returns a base64 string for the LLM.
    """
    with rasterio.open(tiff_path) as src:
        # Read the first 3 bands (Assuming RGB)
        # rasterio reads as (Channels, Height, Width)
        img_array = src.read([1, 2, 3])
        
        # Transpose to (Height, Width, Channels) for OpenCV
        img_array = np.transpose(img_array, (1, 2, 0))
        
        # Normalize to 8-bit (0-255) if it is 16-bit drone imagery
        if img_array.dtype != np.uint8:
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
        # Convert RGB to BGR for OpenCV encoding
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize if the image is too large
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # Encode to JPEG in memory (no temp files saved to disk!)
        success, buffer = cv2.imencode('.jpg', img_bgr)
        if not success:
            raise ValueError("Could not compress image to JPEG.")
            
        # Convert the buffer to a base64 string
        return base64.b64encode(buffer).decode('utf-8')

# Without this the model gets a bare question and an image, and tends to answer that it
# "cannot analyze the image directly" instead of describing what is in it.
VISION_SYSTEM_PROMPT = """You are an expert geospatial imagery annotator. The attached image is
an aerial drone tile, downscaled from the original TIFF. A segmentation model was run on it to
detect animal trails. You are not told how well the model did, so do not assume it did badly:
the tile may have scored very well.

Describe what you actually see in this image: land cover (forest, shrub, grass, bare ground,
water, snow) and lighting (shadows, glare, washed-out or very dark areas). Say where in the
tile things are (e.g. "upper left", "along the right edge"). Mention something that would make
a thin trail hard to see only if it is clearly present. If the image is evenly lit and clear,
say so plainly; do not go looking for problems or speculate about what "might" or "could"
cause difficulty.

Answer the question you are asked directly and concisely. Report only what is visible; if
something is not visible or you cannot tell at this resolution, say so rather than guessing.
Do not give generic advice about how to inspect an image."""

@traceable(run_type="chain", name="analyze_image_visually")
def analyze_image_visually(image_path: str, user_prompt: str) -> str:
    """
    Sends a resized image and a text prompt to GPT-4o-mini for visual analysis.
    """
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"

    print(f"👁️  Resizing and encoding massive TIFF: {os.path.basename(image_path)}...")
    base64_image = encode_and_resize_tiff(image_path, max_size=1024)
    
    # Initialize the Vision LLM
    vision_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500, temperature=0)
    
    # Construct the Multimodal Message
    message = HumanMessage(
        content=[
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high" 
                }
            }
        ]
    )
    
    print("🧠 Sending resized image to GPT-4o-mini for visual analysis...")
    response = vision_llm.invoke([SystemMessage(content=VISION_SYSTEM_PROMPT), message])
    
    return response.content

# --- Test the Vision Tool ---
if __name__ == "__main__":
    # Update this to exactly match your tile name
    test_tile = "ALL-2-81-13-W6M" 
    test_image_path = os.path.join(root_dir, "data", "tiffs", f"{test_tile}.tif")
    
    prompt = """
    1. Do you see dense vegetation, forests, or bare dirt?
    2. Are there any visible shadows or washed-out areas that might confuse a computer vision model trying to detect animal trails?
    Be concise.
    """
    
    if os.path.exists(test_image_path):
        result = analyze_image_visually(test_image_path, prompt)
        print("\n=== 🤖 VISION AI ANALYSIS ===")
        print(result)
        print("=============================")
    else:
        print(f"❌ Test image not found at: {test_image_path}")