# 🌍 Geospatial AI QA Agent (Multi-Agent System)

An explainable AI (XAI) assistant and automated data pipeline for post-inference Quality Assurance (QA) of geospatial computer vision models. 

This project bridges the gap between raw model outputs (Shapefiles/TIFFs) and actionable human insights. It uses a **LangGraph Multi-Agent architecture** to combine mathematical Explainable AI (SHAP) with Multimodal Vision (GPT-4o), allowing reviewers to chat with an AI Supervisor to triage model predictions.

## 🚀 Enterprise-Grade Architecture
This system is built using modern, decoupled AI engineering standards:
* **The Brain (LangGraph):** A Multi-Agent router that delegates tasks to a 'Data Agent' (DuckDB tools) and a 'Vision Agent' (Multimodal image analysis).
* **Multimodal Vision:** Uses OpenCV and Rasterio to compress massive drone TIFFs in-memory and passes them to `GPT-4o-mini` for visual verification of terrain anomalies.
* **The Heavy Lifting (Spatial Math):** `GeoPandas` and `Rasterio` for complex geometry manipulation, CRS alignment, and Tolerance-based Buffer IoU.
* **The Explainability Engine (XAI):** `XGBoost` and `SHAP` to extract mathematical feature importance from image metadata.
* **The Analytical Database:** `DuckDB` for fast, serverless storage of spatial metrics.
* **The Frontend & Deployment:** `Streamlit` for the visual mapping dashboard, fully containerized with `Docker`.

## ✨ Key Features

1. **Multi-Agent Routing (LangGraph)**
   * A Supervisor LLM automatically routes user questions. It queries DuckDB for hard metrics, then physically looks at the map images using Vision AI to confirm findings before synthesizing a final answer.
   
2. **Smart Spatial Math Engine**
   * **Dynamic Geometry Handling:** Automatically detects feature types. Applies exact-area math for Polygons (Cultivated Areas) and industry-standard **Tolerance-based Buffer IoU** (2-meter buffer) for LineStrings (Animal Trails).
   
3. **Explainable AI (XAI) Pipeline**
   * Extracts environmental metadata from RGB drone TIFFs (Brightness, Contrast, Edge Complexity).
   * Trains a localized XGBoost meta-model on spatial error rates to generate SHAP values, explaining exactly *which* environmental factor caused a failure.

4. **Visual Map Overlay Dashboard**
   * Renders the raw Drone TIFF overlaid with Ground Truth (Green) and Model Predictions (Red) directly in the UI using Matplotlib and Streamlit.

## 📂 Project Structure
```text
geo_qa_agent/
├── data/                   # Local data volume (TIFFs, Ground Truth ZIPs, Prediction ZIPs)
├── src/                    
│   ├── metrics/            
│   │   ├── pipeline.py     # Automated ETL: Math -> XGBoost -> SHAP -> DuckDB
│   │   └── visualizer.py   # Rasterio/Matplotlib rendering engine
│   └── agent/              
│       ├── graph_agent.py  # LangGraph Supervisor & Multi-Agent logic
│       └── vision_tool.py  # TIFF compression and GPT-4o Multimodal Vision tool
├── app/                    
│   └── main.py             # Streamlit Dashboard & Chat UI
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration
└── requirements.txt        # Python dependencies