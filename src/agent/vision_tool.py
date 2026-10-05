import logging
import os
import sys
import base64
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
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

# Run as a script, sys.path[0] is src/agent, so the project root has to be added
sys.path.insert(0, root_dir)
from src.agent.verdict import MATCH_TOLERANCE_M  # noqa: E402
from src.metrics.shapefiles import trails_within  # noqa: E402

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(f"CRITICAL: OPENAI_API_KEY not found. Checked path: {env_path}")

logger = logging.getLogger(__name__)


def _summarize_b64(outputs) -> dict:
    """
    Keep the base64 payload out of the trace; log only its size.
    LangSmith hands this the raw return value (a str) for single-value returns,
    but a dict when the traced function returns one -- handle both.
    """
    b64 = outputs.get("output", "") if isinstance(outputs, dict) else (outputs or "")
    return {"base64_chars": len(b64)}

def _read_bgr(tiff_path: str, max_size: int, bounds=None):
    """
    The tile, or the part of it inside bounds, as an 8-bit BGR image no larger than
    max_size on its long edge. Also returns a function taking map coordinates to pixel
    coordinates in that image, for drawing on it.
    """
    with rasterio.open(tiff_path) as src:
        window = Window(0, 0, src.width, src.height)
        if bounds is not None:
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))

        # Read at the size that will be sent. Reading every pixel of a 300 MB TIFF to
        # shrink it afterwards took the memory of the whole tile for each question asked.
        h, w = int(window.height), int(window.width)
        scale = min(1.0, max_size / max(h, w))
        out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))

        # Read the first 3 bands (Assuming RGB)
        # rasterio reads as (Channels, Height, Width)
        img_array = src.read([1, 2, 3], window=window, out_shape=(3, out_h, out_w),
                             resampling=Resampling.average)
        to_full_res = ~src.transform

    # Transpose to (Height, Width, Channels) for OpenCV
    img_array = np.transpose(img_array, (1, 2, 0))

    # Normalize to 8-bit (0-255) if it is 16-bit drone imagery
    if img_array.dtype != np.uint8:
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    def to_pixel(x, y):
        col, row = to_full_res * (x, y)
        return (col - window.col_off) * scale, (row - window.row_off) * scale

    return img_bgr, to_pixel


def _to_base64_jpeg(img_bgr) -> str:
    # Encode to JPEG in memory (no temp files saved to disk!)
    success, buffer = cv2.imencode('.jpg', img_bgr)
    if not success:
        raise ValueError("Could not compress image to JPEG.")
    return base64.b64encode(buffer).decode('utf-8')


@traceable(run_type="tool", name="encode_and_resize_tiff", process_outputs=_summarize_b64)
def encode_and_resize_tiff(tiff_path: str, max_size: int = 1024, bounds=None) -> str:
    """
    Reads a massive drone TIFF, extracts RGB, resizes it to a safe dimension, 
    compresses it to JPEG, and returns a base64 string for the LLM.

    bounds is (minx, miny, maxx, maxy) in the raster's CRS. When given, only that part of
    the tile is read, so a small area keeps the detail the whole tile loses on resizing.
    """
    img_bgr, _ = _read_bgr(tiff_path, max_size, bounds)
    return _to_base64_jpeg(img_bgr)


# Colours for the lines drawn on a crop, as (name in the prompt, BGR). Cyan and magenta
# because neither occurs in vegetation, soil or snow; the viewer's green would vanish.
ANNOTATED = ("cyan", (255, 255, 0))
PREDICTED = ("magenta", (255, 0, 255))


def _draw_trail(img_bgr, geometry, to_pixel, colour) -> None:
    """Draw every line of a geometry, or the outline of every polygon, on the image."""
    if geometry is None or geometry.is_empty:
        return
    for part in getattr(geometry, "geoms", [geometry]):
        if hasattr(part, "geoms"):
            _draw_trail(img_bgr, part, to_pixel, colour)
            continue
        line = part.exterior if part.geom_type == "Polygon" else part
        if line.geom_type not in ("LineString", "LinearRing"):
            continue
        points = np.array([to_pixel(x, y) for x, y, *_ in line.coords]).round().astype(np.int32)
        cv2.polylines(img_bgr, [points], False, colour, thickness=2, lineType=cv2.LINE_AA)


def _summarize_pair(outputs) -> dict:
    """Keep both base64 payloads out of the trace; log only their sizes."""
    pair = outputs.get("output", outputs) if isinstance(outputs, dict) else outputs
    try:
        return {"base64_chars": [len(b64) for b64 in pair]}
    except TypeError:
        return {"base64_chars": None}


@traceable(run_type="tool", name="encode_cell_with_trails", process_outputs=_summarize_pair)
def encode_cell_with_trails(tiff_path: str, bounds, annotated, predicted,
                            max_size: int = 1024) -> tuple:
    """
    One part of a tile twice: as it is, and with the annotated and predicted trails drawn
    on it. Returns the two as base64 JPEGs.

    Two images, because a line drawn over a trail a few pixels wide hides the trail. The
    first shows what is on the ground; the second shows where each line says a trail is.
    """
    img_bgr, to_pixel = _read_bgr(tiff_path, max_size, bounds)
    marked = img_bgr.copy()
    _draw_trail(marked, annotated, to_pixel, ANNOTATED[1])
    _draw_trail(marked, predicted, to_pixel, PREDICTED[1])
    return _to_base64_jpeg(img_bgr), _to_base64_jpeg(marked)


# Without this the model gets a bare question and an image, and tends to answer that it
# "cannot analyze the image directly" instead of describing what is in it.
# What the image is, for the whole tile and for one grid cell of it
WHOLE_TILE = "an aerial drone tile, downscaled from the original TIFF"
ONE_CELL = ("one small square cut from the {location} of an aerial drone tile, at close to "
            "full resolution. It shows that part of the tile only")

VISION_SYSTEM_PROMPT = """You are an expert geospatial imagery annotator. The attached image is
{image}. A segmentation model was run on it to
detect animal trails. You are not told how well the model did, so do not assume it did badly:
the tile may have scored very well.

Describe what you actually see in this image: land cover (forest, shrub, grass, bare ground,
water, snow) and lighting (shadows, glare, washed-out or very dark areas). Say where in the
image things are (e.g. "upper left", "along the right edge"). Mention something that would make
a thin trail hard to see only if it is clearly present. If the image is evenly lit and clear,
say so plainly; do not go looking for problems or speculate about what "might" or "could"
cause difficulty.

Answer the question you are asked directly and concisely. Report only what is visible; if
something is not visible or you cannot tell at this resolution, say so rather than guessing.
Do not give generic advice about how to inspect an image."""

# Added when the square comes with a second copy that has the two sets of lines drawn on it
TRAILS_PROMPT = """

You are given two images of the same square, {width:.0f} metres across. The first is the
imagery as it is. The second is the same imagery with lines drawn on it:
- {Annotated} lines are trails drawn by a human annotator. They sit near the trail they stand
  for, not exactly on it.
- {Predicted} lines are the trails the model predicted.
A {predicted} line and a {annotated} line within {tolerance:g} metres of each other stand for the same
trail; do not report the gap between them as a disagreement. Either colour may be absent.

Find where the two disagree: a stretch of one colour with no line of the other colour near
it. For each such place, look at the same spot in the first image, where no line covers the
ground, and say which of these holds:
- a trail is visible there, so the line is right and the other set lacks it;
- no trail is visible there although the ground is clear enough to show one;
- you cannot tell, because of shadow, canopy, blur or resolution.
A thin trail is often not visible from above. "Cannot tell" is the right answer whenever you
are not sure, and it is more useful than a guess. Neither the annotator nor the model is
assumed to be right."""

def _trails_in(image_path: str, bounds, trail_zips):
    """
    (annotated, predicted) geometries inside bounds, or None when neither layer has
    anything there or the files cannot be read. A crop without lines is still worth
    sending, so a failure here is not an error.
    """
    try:
        with rasterio.open(image_path) as src:
            crs = src.crs
        annotated, predicted = (trails_within(path, crs, bounds) for path in trail_zips)
    except Exception as e:
        logger.warning("Could not read the trails for this crop: %s", e)
        return None
    return None if annotated is None and predicted is None else (annotated, predicted)


def _image_part(base64_image: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"},
    }


@traceable(run_type="chain", name="analyze_image_visually")
def analyze_image_visually(image_path: str, user_prompt: str, bounds=None,
                           location: str = "", trail_zips=None) -> str:
    """
    Sends a resized image and a text prompt to GPT-4o-mini for visual analysis.

    With bounds, (minx, miny, maxx, maxy) in the raster's CRS, it sends that part of the
    tile alone; location says where in the tile that is, e.g. "north-east". With
    trail_zips as well, the (ground truth, prediction) zipped shapefiles, it also sends a
    second copy with both sets of lines drawn on it and asks where they disagree.
    """
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"

    logger.info("Encoding %s%s", os.path.basename(image_path),
                "" if bounds is None else f" ({location or 'one'} cell)")
    image = WHOLE_TILE if bounds is None else ONE_CELL.format(location=location or "middle")
    system_prompt = VISION_SYSTEM_PROMPT.format(image=image)

    trails = _trails_in(image_path, bounds, trail_zips) if bounds is not None and trail_zips else None
    if trails is None:
        images = [encode_and_resize_tiff(image_path, max_size=1024, bounds=bounds)]
    else:
        images = encode_cell_with_trails(image_path, bounds, *trails, max_size=1024)
        system_prompt += TRAILS_PROMPT.format(
            width=bounds[2] - bounds[0], tolerance=MATCH_TOLERANCE_M,
            annotated=ANNOTATED[0], predicted=PREDICTED[0],
            Annotated=ANNOTATED[0].capitalize(), Predicted=PREDICTED[0].capitalize())

    # Initialize the Vision LLM
    vision_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500, temperature=0)
    
    # Construct the Multimodal Message
    message = HumanMessage(
        content=[{"type": "text", "text": user_prompt}, *(_image_part(b64) for b64 in images)]
    )
    
    logger.info("Sending %d image(s) for visual analysis", len(images))
    response = vision_llm.invoke([SystemMessage(content=system_prompt), message])
    
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