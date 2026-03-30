# 🤖 Multi-Agent AI QA System (LangGraph + FastAPI)

An enterprise-grade, multi-agent AI platform for automated Quality Assurance (QA) and Explainable AI (XAI) analysis of computer vision models. 

This project bridges the gap between raw model outputs (predictions vs. ground truth) and actionable human insights. It utilizes a **decoupled microservices architecture**, combining **LangGraph** multi-agent routing, mathematical anomaly detection (SHAP), and **Multimodal Vision** (GPT-4o) to allow human reviewers to chat with an AI Supervisor to triage model failures.

## 🚀 Enterprise Architecture
This system is built using modern, production-ready AI engineering standards:
* **The Brain (LangGraph):** A Multi-Agent router that delegates tasks to a **'Data Agent'** (queries DuckDB for statistical metrics) and a **'Vision Agent'** (analyzes compressed images via Multimodal LLMs).
* **The Backend (FastAPI):** A highly scalable REST API that securely wraps the LangGraph agents and handles all LLM network traffic.
* **The Heavy Lifting (Spatial/Math Engine):** `GeoPandas`, `Rasterio`, and `Shapely` for complex geometry manipulation, CRS alignment, and Tolerance-based Buffer IoU.
* **Explainable AI (XAI):** `XGBoost` and `SHAP` extract mathematical feature importance from image metadata to explain *why* a model failed.
* **Analytical Database:** `DuckDB` for fast, serverless storage of evaluation metrics.
* **The Frontend (Streamlit):** An interactive chat and visual mapping dashboard.
* **DevOps (Docker Compose):** Fully containerized multi-container orchestration (Frontend + Backend).

## ✨ Key Features

1. **Multi-Agent Orchestration (LangGraph)**
   * A Supervisor LLM automatically routes user questions. It queries DuckDB for hard metrics, then physically looks at the map images using Vision AI to confirm findings before synthesizing a final answer.
   
2. **Multimodal Vision Integration**
   * Uses OpenCV and Rasterio to compress massive, high-resolution TIFFs in-memory and passes them to `GPT-4o-mini` for visual verification of terrain anomalies, shadows, and false positives.

3. **Smart Math & Evaluation Engine**
   * **Dynamic Geometry Handling:** Automatically detects feature types. Applies exact-area math for Polygons and industry-standard **Tolerance-based Buffer IoU** (2-meter buffer) for LineStrings.
   
4. **Explainable AI (XAI) Pipeline**
   * Extracts environmental metadata from images (Brightness, Contrast, Edge Complexity).
   * Trains a localized XGBoost meta-model on spatial error rates to generate SHAP values, providing the LLM with mathematical proof of failure causes.

## 📂 Project Structure
```text
geo_qa_agent/
├── data/                   # Local data volume (TIFFs, Ground Truth ZIPs, Prediction ZIPs)
├── src/                    
│   ├── api/
│   │   └── server.py       # FastAPI Backend Server
│   ├── agent/              
│   │   ├── graph_agent.py  # LangGraph Supervisor & Multi-Agent logic
│   │   └── vision_tool.py  # Image compression and Multimodal Vision tool
│   └── metrics/            
│       ├── pipeline.py     # Automated ETL: Math -> XGBoost -> SHAP -> DuckDB
│       └── visualizer.py   # Rasterio/Matplotlib rendering engine
├── app/                    
│   └── main.py             # Streamlit Dashboard & Chat UI (Frontend)
├── Dockerfile              # Container definition
├── docker-compose.yml      # Microservices orchestration
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variables template