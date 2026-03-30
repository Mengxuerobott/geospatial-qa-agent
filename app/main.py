import streamlit as st
import pandas as pd
import os
import sys
import duckdb
from langchain_core.messages import HumanMessage
import requests

# Add the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- NEW: Import the LangGraph Agent instead of the old one ---
from src.agent.graph_agent import create_graph_agent
from src.metrics.visualizer import plot_tile_results

# --- Page Config ---
st.set_page_config(page_title="Geospatial QA Agent", layout="wide")
st.title("🌍 Explainable AI: Multi-Agent Geospatial QA")

# --- Initialize LangGraph Agent in Session State ---
if "agent" not in st.session_state:
    st.session_state.agent = create_graph_agent()
    
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Multi-Agent QA Supervisor. Ask me to check the metrics for a tile, or ask me to physically look at the drone imagery to explain a failure."}
    ]

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
        
        with st.spinner(f"Loading map for {selected_tile}..."):
            try:
                fig = plot_tile_results(tiff_path, gt_path, pred_path, selected_tile)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not load map: {e}")

        # --- DISPLAY DUCKDB METRICS ---
        st.divider()
        st.subheader("📊 Tile Metrics (From DuckDB)")
        if os.path.exists(db_path):
            try:
                with duckdb.connect(db_path) as conn:
                    query = f"SELECT iou, brightness, contrast FROM tile_metrics WHERE tile_id = '{selected_tile}'"
                    df_metrics = conn.execute(query).df()
                    
                if not df_metrics.empty:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("IoU Score", f"{df_metrics['iou'].iloc[0]:.4f}")
                    m2.metric("Brightness", f"{df_metrics['brightness'].iloc[0]:.1f}")
                    m3.metric("Contrast", f"{df_metrics['contrast'].iloc[0]:.1f}")
                else:
                    st.info("No metrics found in database for this tile. Did you run the pipeline?")
            except Exception as e:
                st.error(f"Database error: {e}")
        else:
            st.warning("Database not found. Run pipeline.py first.")


with col2:
    st.subheader("💬 Chat with Multi-Agent Supervisor")
    
    # Use environment variable for Docker, default to localhost for local testing
    API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
    
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
            with st.spinner("Supervisor is coordinating Data & Vision Agents via API..."):
                try:
                    # Send HTTP POST request to FastAPI
                    response = requests.post(API_URL, json={"message": prompt})
                    
                    if response.status_code == 200:
                        answer = response.json()["reply"]
                        st.markdown(answer)
                        # Save assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the Backend API. Is FastAPI running?")