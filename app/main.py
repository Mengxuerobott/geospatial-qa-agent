import streamlit as st
import os
import sys
import uuid
import duckdb
import requests

# Add the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# The agent itself runs behind the FastAPI backend; this app only talks to it over HTTP
from src.metrics.visualizer import plot_tile_results
from src.agent.verdict import FAIL_IOU_THRESHOLD, MATCH_TOLERANCE_M

# --- Page Config ---
st.set_page_config(page_title="Geospatial QA Agent", layout="wide")
st.title("🌍 Explainable AI: Geospatial QA Agent")

# One conversation per browser session; the API keys the agent's memory on this
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

GREETING = {"role": "assistant", "content": "Hello! I am your Geospatial QA agent. Ask me to check the metrics for a tile, or ask me to physically look at the drone imagery to explain a failure."}

if "messages" not in st.session_state:
    st.session_state.messages = [GREETING]

# --- Layout: Two Columns ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗺️ Map & Imagery Viewer")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    tiff_dir = os.path.join(base_dir, 'tiffs')
    db_path = os.path.join(base_dir, 'metrics.duckdb')
    
    if os.path.exists(tiff_dir):
        available_tiles = [f.replace('.tif', '') for f in os.listdir(tiff_dir) if f.endswith('.tif')]
    else:
        available_tiles = []
        
    if not available_tiles:
        st.warning("No TIFFs found in the data/tiffs/ folder.")
    else:
        selected_tile = st.selectbox("Select a Tile to Inspect:", available_tiles)
        
        tiff_path = os.path.join(tiff_dir, f'{selected_tile}.tif')
        gt_path = os.path.join(base_dir, 'ground_truth', f'{selected_tile}.zip')
        pred_path = os.path.join(base_dir, 'predictions', f'{selected_tile}.zip')
        
        # The tile's grid cells, if the database has them. A database built before the
        # pipeline scored cells has no such table; the map is then drawn without them.
        cells = None
        if os.path.exists(db_path):
            try:
                with duckdb.connect(db_path, read_only=True) as conn:
                    cells = conn.execute(
                        "SELECT * FROM cell_metrics WHERE tile_id = ?", [selected_tile]).df()
            except duckdb.Error:
                cells = None
        has_cells = cells is not None and not cells.empty
        show_cells = has_cells and st.checkbox(
            "Show where prediction and annotation disagree", value=True,
            help="Shades each grid cell by how much of its trail is annotated but not "
                 "predicted, or predicted but not annotated. Lines within "
                 f"{MATCH_TOLERANCE_M:g} m of each other count as the same trail. "
                 "Cells below the pass threshold are outlined and named; "
                 "use the name in the chat, e.g. \"look at r1c5\".")

        with st.spinner(f"Loading map for {selected_tile}..."):
            try:
                fig = plot_tile_results(tiff_path, gt_path, pred_path, selected_tile,
                                        cells=cells if show_cells else None)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not load map: {e}")

        # --- DISPLAY DUCKDB METRICS ---
        st.divider()
        st.subheader("📊 Tile Metrics (From DuckDB)")
        if os.path.exists(db_path):
            try:
                with duckdb.connect(db_path) as conn:
                    query = "SELECT iou, brightness, contrast FROM tile_metrics WHERE tile_id = ?"
                    df_metrics = conn.execute(query, [selected_tile]).df()
                    
                if not df_metrics.empty:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("IoU Score", f"{df_metrics['iou'].iloc[0]:.4f}")
                    m2.metric("Brightness", f"{df_metrics['brightness'].iloc[0]:.1f}")
                    m3.metric("Contrast", f"{df_metrics['contrast'].iloc[0]:.1f}")
                    if has_cells:
                        scored = cells[cells["iou"].notna()]
                        failing = int((scored["iou"] < FAIL_IOU_THRESHOLD).sum())
                        m4.metric("Cells failing", f"{failing} of {len(scored)}",
                                  help="Grid cells with trail in them that scored below "
                                       "the pass threshold.")
                else:
                    st.info("No metrics found in database for this tile. Did you run the pipeline?")
            except Exception as e:
                st.error(f"Database error: {e}")
        else:
            st.warning("Database not found. Run pipeline.py first.")


with col2:
    st.subheader("💬 Chat with the QA Agent")
    
    # Use environment variable for Docker, default to localhost for local testing
    API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
    
    # A new thread_id gives the agent a blank memory; the old thread is simply abandoned.
    # This runs before the history is drawn, so the cleared chat shows on this same rerun.
    if st.button("🆕 New conversation", help="Clear the chat and the agent's memory of it"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = [GREETING]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input(f"E.g., Why did {selected_tile if available_tiles else 'this tile'} fail? Look at the image."):
        
        # 1. Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Call the FastAPI Backend
        with st.chat_message("assistant"):
            with st.spinner("Agent is querying the metrics and vision tools via API..."):
                try:
                    # Send HTTP POST request to FastAPI
                    response = requests.post(
                        API_URL,
                        json={
                            "message": prompt,
                            "thread_id": st.session_state.thread_id,
                            "selected_tile": selected_tile if available_tiles else None,
                        },
                    )
                    
                    if response.status_code == 200:
                        answer = response.json()["reply"]
                        st.markdown(answer)
                        # Save assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the Backend API. Is FastAPI running?")