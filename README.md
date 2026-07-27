# Geospatial QA Agent

A chat interface for triaging computer vision failures on drone imagery.

You point it at a tile and ask why the model did badly. It looks up the tile's IoU and SHAP
values in DuckDB, then sends the actual image to a vision model to check whether the numbers
match what's visible — shadows, dense vegetation, washed-out ground. The answer combines both.

The routing between those two steps is a LangGraph ReAct agent, so the LLM decides which tool
to call and when, rather than following a fixed script.

## How it fits together

```
Streamlit UI  ──HTTP──>  FastAPI  ──>  LangGraph supervisor (gpt-4o-mini)
                                            ├── get_duckdb_metrics   → DuckDB
                                            └── run_vision_analysis  → TIFF → JPEG → gpt-4o-mini
```

The metrics in DuckDB are produced ahead of time by a separate batch pipeline
(`src/metrics/pipeline.py`); the agent only reads them.

## Pieces

**Evaluation pipeline** (`src/metrics/pipeline.py`)
Walks `data/tiffs/`, pairs each TIFF with its ground-truth and prediction shapefile zips, and
computes:

- brightness and contrast, straight from the raster bands
- IoU between ground truth and prediction, after reprojecting both to the raster's CRS

The ground truth here is animal trails — LineStrings, which have no area, so IoU would always
be zero. The pipeline buffers any line geometry by 5 metres first and does the overlap on the
resulting polygons. Polygons pass through unbuffered.

It then fits an XGBoost regressor predicting error (`1 - IoU`) from brightness and contrast,
and runs SHAP over it. The per-tile SHAP values are what let the agent say *brightness pushed
this tile's error up* instead of just *this tile is bad*. Everything lands in
`data/metrics.duckdb`.

**Vision tool** (`src/agent/vision_tool.py`)
Drone TIFFs are far too large to send to an LLM. This reads the first three bands, normalises
16-bit to 8-bit if needed, downscales the long edge to 1024px, JPEG-encodes it in memory, and
base64s it. Nothing is written to disk.

**Agent** (`src/agent/graph_agent.py`)
`create_react_agent` with two tools and a system prompt telling it to get the numbers first,
then confirm visually. `src/agent/qa_agent.py` is the earlier single-agent AgentExecutor
version, kept for reference — it isn't wired into the app.

**Frontend** (`app/main.py`)
Left column renders the tile with ground truth in green and predictions in red over the RGB
raster, plus the DuckDB metrics. Right column is the chat, which posts to the API rather than
holding an agent in Streamlit session state.

## Running it

Needs an OpenAI key.

```bash
cp .env.example .env
```

```bash
pip install -r requirements.txt
```

Build the metrics database first, or the agent will have nothing to query:

```bash
python src/metrics/pipeline.py
```

Then the two services:

```bash
uvicorn src.api.server:app --reload
```

```bash
streamlit run app/main.py
```

Or with Docker, which runs both:

```bash
docker compose up --build
```

The compose setup points Streamlit at `http://api:8000/chat` via `API_URL`; running locally it
falls back to localhost.

### On Windows

`pipeline.py` prints emoji. In a GBK console that raises `UnicodeEncodeError` before it does
any work:

```bash
set PYTHONIOENCODING=utf-8
```

## LangSmith tracing

Off unless you turn it on. With `LANGCHAIN_TRACING_V2` unset or `false`, the `@traceable`
decorators do nothing — no network calls, no change in behaviour.

Fill in `LANGCHAIN_API_KEY` in `.env` (from smith.langchain.com → Settings → API Keys). The
project named in `LANGCHAIN_PROJECT` is created on the first trace; you don't need to make it
in the UI first.

To check the key works — note the `load_dotenv()`, since a bare `python -c` won't read `.env`
and `Client()` will claim the key is missing even when it isn't:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from langsmith import Client; print(list(Client().list_projects(limit=1)))"
```

LangChain and LangGraph instrument themselves. The parts that aren't LangChain are decorated
by hand:

| Function | Why |
| --- | --- |
| `analyze_image_visually` | parent span, so the encode step and the vision call sit in one subtree |
| `encode_and_resize_tiff` | records the base64 *length* only — the payload is ~320 KB per call and would otherwise be uploaded every time |
| `run_pipeline`, `train_and_explain` | the XGBoost + SHAP batch job, which involves no LLM at all |

Run the pipeline and you should get a `run_pipeline` trace with `train_and_explain` nested
under it. Ask the agent something that needs the image and the tree looks like:

```
LangGraph
├── agent → ChatOpenAI
├── tools → get_duckdb_metrics
├── agent → ChatOpenAI
├── tools → run_vision_analysis
│              └── analyze_image_visually
│                   ├── encode_and_resize_tiff
│                   └── ChatOpenAI          (vision)
└── agent → ChatOpenAI
```

In Docker the `api` service already picks `.env` up through `env_file`. The frontend makes no
LLM calls, so it needs nothing.

To rip it out: delete the decorators and the `from langsmith import traceable` imports.

## Evals

Prompt changes to an agent are untestable by eye — you fix one routing case and silently break
another. `evals/` turns that into a measurement.

```bash
python evals/dataset.py     # push the dataset; re-run after editing it
python evals/run_evals.py    # run the agent against it
```

The dataset lives in `evals/dataset.py` rather than only in the LangSmith UI, so it's diffable
and reviewable. Reference IoU values are read out of DuckDB at sync time instead of being
hardcoded, so re-running the pipeline on different imagery doesn't silently invalidate them.

Seven examples covering three things:

- **routing** — metrics-only questions must not open a 300 MB TIFF; visual questions must;
  diagnostic questions need both
- **grounding** — the answer has to quote the IoU that's actually in DuckDB, not a plausible
  nearby number
- **limits** — a tile that doesn't exist, and a question (weather) no tool can answer

The evaluators are code, not LLM-as-judge. The failures worth catching are wrong tool calls and
invented numbers, both checkable against the database — a judge model would be slower, cost
money per run, and be worse at exactly that.

Every run creates a new experiment, so you can compare two prompts side by side in LangSmith.

### Testing the tests

The first full run scored 7/7, which proves nothing on its own: a suite that has never failed
may just be incapable of failing. `tests/test_evaluators.py` feeds the evaluators hand-written
outputs containing the exact failure modes they exist to catch — a hallucinated IoU, a
fabricated score for a nonexistent tile, an unnecessary vision call — and asserts they score
those **0**.

```bash
pytest tests/ -q
```

(The first attempt at validation was a deliberately sabotaged agent whose prompt forbade tool
use. It called the tools anyway and passed everything, which made it useless as a control —
hence testing the evaluators directly.)

## Layout

```text
data/                      TIFFs, ground truth zips, prediction zips, metrics.duckdb
src/
  api/server.py            FastAPI, holds one agent instance for the process
  agent/
    graph_agent.py         LangGraph supervisor + the two tools
    vision_tool.py         TIFF → resized JPEG → base64, and the vision call
    qa_agent.py            earlier single-agent version, unused
  metrics/
    pipeline.py            IoU → XGBoost → SHAP → DuckDB
    visualizer.py          the matplotlib overlay Streamlit renders
    image_extractor.py     standalone image feature extraction
    spatial_calculator.py  standalone polygon error helpers
  xai_engine.py            QATriageEngine — classifier variant, not wired in
app/main.py                Streamlit dashboard and chat
evals/
  dataset.py               eval cases, versioned in git
  evaluators.py            code-based scorers
  run_evals.py             runs the agent against the dataset
tests/
  test_evaluators.py       proves the scorers can actually fail
```

## Notes

- SQL in the tools is built with f-strings on `tile_id`. Fine for a local single-user tool,
  not fine if this is ever exposed.
- `qa_agent.py`, `xai_engine.py`, `image_extractor.py`, and `spatial_calculator.py` are
  earlier or parallel implementations that nothing in the running app imports.
- Pinned to the LangChain 0.2 line. `langchain-core` has to be `>=0.2.27` because every
  `langgraph` 0.2.x requires it; pinning core to 0.2.11 makes the requirements unsolvable.
