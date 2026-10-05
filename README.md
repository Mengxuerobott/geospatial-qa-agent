# Geospatial QA Agent

A chat interface for working out why a segmentation model did badly on a drone image tile.

It is a single LangGraph ReAct agent with two tools:

| Tool | What it does |
| --- | --- |
| `get_duckdb_metrics` | Reads the tile's IoU, pass/fail verdict, SHAP values and weak areas from DuckDB |
| `run_vision_analysis` | Sends the tile image, or one grid cell of it at full resolution, to a vision model and reports what is visible |

Ask *"why did tile SE-31-18-03-W fail?"* and the agent looks up the numbers, then looks at the
image to check whether they match what is there — shadows, dense vegetation, washed-out
ground — and answers from both. The LLM chooses which tool to call and when; there is no
fixed script, no supervisor and no sub-agents.

It holds a conversation, so you can follow up with *"why? look at the image"*, or select a
tile in the viewer and ask *"why did this one do badly?"* without typing its ID.

## Architecture

```
Streamlit UI  ──HTTP──>  FastAPI  ──>  LangGraph ReAct agent (gpt-4o-mini)
                                            ├── get_duckdb_metrics   → DuckDB
                                            └── run_vision_analysis  → TIFF → JPEG → gpt-4o-mini
```

The metrics in DuckDB are produced ahead of time by a batch pipeline
(`src/metrics/pipeline.py`). The agent only reads them.

## Quick start

You need Python and an OpenAI API key.

1. Create `.env` and put your key in it:

   ```bash
   cp .env.example .env
   ```

2. Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv .venv
   ```

   ```bash
   source .venv/bin/activate
   ```

   ```bash
   pip install -r requirements.txt
   ```

   On macOS, XGBoost also needs the OpenMP runtime, or the pipeline stops at `import
   xgboost` with "libomp.dylib could not be loaded":

   ```bash
   brew install libomp
   ```

3. Put your data under `data/`, one set of files per tile, all named after the tile ID:

   ```text
   data/tiffs/<tile_id>.tif
   data/ground_truth/<tile_id>.zip
   data/predictions/<tile_id>.zip
   ```

4. Build the metrics database. Without it the agent has nothing to query:

   ```bash
   python src/metrics/pipeline.py
   ```

5. Start the API (port 8000):

   ```bash
   uvicorn src.api.server:app --reload
   ```

6. In a second terminal, start the UI (port 8501):

   ```bash
   streamlit run app/main.py
   ```

### With Docker

Steps 1, 3 and 4 still apply. Then one command runs both services:

```bash
docker compose up --build
```

Compose points Streamlit at `http://api:8000/chat` through `API_URL`; run locally, it falls
back to `http://localhost:8000/chat`.

`.dockerignore` keeps `.env` and `data/` out of the images. The `api` service reads its keys
at run time through `env_file`, and both services get the data through the `./data` volume
mount. Images built before `.dockerignore` existed contain your `.env`: rebuild them, and
rotate the keys if those images were ever pushed anywhere.

### On Windows

`pipeline.py` prints emoji, which raises `UnicodeEncodeError` in a GBK console before any
work is done. Set this first:

```bash
set PYTHONIOENCODING=utf-8
```

## How it works

### The agent (`src/agent/graph_agent.py`)

`create_react_agent` with the two tools and a system prompt. The prompt tells the agent to:

- get the metrics first, then use the image to confirm them
- use the metrics tool alone for a question about a score, and add the vision tool when the
  user asks why, asks for a diagnosis, or asks about the image
- open every answer with whether the tile passed or failed and its IoU, and contradict the
  user in the first sentence when the data disagrees with their question

A tile fails QA below an IoU of 0.75 (`FAIL_IOU_THRESHOLD` in `src/agent/verdict.py`). The
metrics tool returns the verdict next to the IoU, so the LLM never decides for itself what
counts as a failure.

### The metrics tool and the pipeline (`src/metrics/pipeline.py`)

The pipeline walks `data/tiffs/` and pairs each TIFF with its ground-truth and prediction
shapefile zips. It measures every tile twice: once whole, and once on a grid of 50 m cells
(`CELL_SIZE_M`).

**Per tile**, it computes the IoU between ground truth and prediction, after reprojecting
both to the raster's CRS. This is the number the pass/fail verdict uses.

The ground truth is animal trails, which are LineStrings. Lines have no area, so their IoU
would always be zero. The pipeline buffers line geometry by 5 metres and measures overlap on
the resulting polygons. Polygons pass through unbuffered.

**Per cell**, it computes the IoU inside the cell and five image attributes, all on valid
pixels only (the alpha band and the no-data padding are left out):

| Attribute | What it measures |
| --- | --- |
| `brightness` | Mean of the colour bands |
| `contrast` | Standard deviation of the colour bands |
| `shadow_fraction` | Share of pixels whose brightest band is under 25% of full scale |
| `greenness` | Excess Green index, `(2G − R − B) / (R + G + B)`: vegetation cover |
| `sharpness` | Variance of the Laplacian: low when the imagery is blurred |

The cells are windows read out of the TIFF; the image is never cut up. Each cell keeps its
row, column and map bounds, so it can be drawn back onto the whole tile. Three rules keep
the cell scores honest:

- Trails are buffered on the whole tile and clipped to the cell afterwards, so a trail near
  a cell edge is shared between the cells on both sides.
- A cell with no trail in either layer gets no IoU and is not trained on. There is nothing
  in it to get right or wrong.
- A cell with trail in one layer only is a miss or a false positive, and scores 0.

The pipeline then fits one XGBoost regressor across the cells of every tile, predicting
error (`1 - IoU`) from the five attributes, and runs SHAP over it. The SHAP values let the
agent say *shadow pushed the error up here* rather than only *this tile is bad*.

Everything is written to `data/metrics.duckdb`, in two tables:

| Table | One row per | Holds |
| --- | --- | --- |
| `tile_metrics` | tile | whole-tile IoU, brightness and contrast, how many cells were scored and how many failed, and the mean SHAP value of its cells for each attribute |
| `cell_metrics` | grid cell | position and map bounds, the five attributes, the cell's IoU, and its own SHAP values |

`get_duckdb_metrics` reads both. Its reply is the tile's IoU, verdict and SHAP values,
followed by the tile's weak areas: how many cells scored below the threshold, which part of
the tile they are in, and the worst five by name, for example:

```text
Weak areas: 3 of 40 grid cells with trail in them scored below the threshold (3 in the
north-east). Worst first: cell r1c6 (north-east): trail missed entirely, main driver
shadow_fraction (0.82); ...
```

A cell is named by its row from the top and its column from the left. Cell scores are given
as percentages and never called an IoU, so neither the LLM nor the `iou_grounded` scorer can
mistake one for the tile's IoU. Weak areas never change the verdict: a tile that passed
with weak areas still passed, and the agent reports them after the verdict.

The viewer draws the cells over the map; see [The frontend](#the-frontend-appmainpy).

### The vision tool (`src/agent/vision_tool.py`)

Drone TIFFs are too large to send to an LLM. The tool reads the first three bands,
normalises 16-bit to 8-bit if needed, downscales the long edge to 1024 px, JPEG-encodes the
result in memory and base64s it. Nothing is written to disk.

The image goes to the vision model with a system prompt saying what it is (a drone tile run
through an animal-trail segmentation model) and asking what is visible. The prompt does not
say the tile did badly; [What the evals caught](#what-the-evals-caught) explains why.

**Looking at one cell.** A whole tile shrunk to 1024 px shows land cover and lighting, but a
trail a few pixels wide disappears. So the tool takes an optional `cell`, a name from the
metrics tool's weak areas such as `r1c5`. It looks up that cell's map bounds in
`cell_metrics`, reads only that window of the TIFF, and sends it at close to full
resolution. The prompt then says the image is one square from, say, the north-east of the
tile. It still does not say the model did badly there.

The agent is told to look at the worst one or two cells, since each vision call is
token-heavy, and never to pass a cell name the metrics tool did not give it. A name that is
not shaped like `r1c5` is refused before it reaches the database.

### Conversation memory

Passing a checkpointer to `create_graph_agent()` gives the agent memory. Each call carries a
`thread_id`, and the earlier turns on that thread are replayed to the LLM. Without a
checkpointer, every call is a fresh single-turn conversation.

- **Only the last six turns are sent to the LLM** (`MAX_HISTORY_TURNS` in
  `src/agent/history.py`). The cut always lands on a user message, so a tool result is never
  separated from the call that requested it. When turns are dropped, a marker tells the
  model how many, so it says it no longer has them.
- **Finding the tile when the question has no ID.** The system prompt sets the order: an ID
  typed in the message, then the tile open in the viewer, then the tile most recently
  discussed. If none of those gives a tile, the agent asks.
- **Memory lives in the server process.** The API uses LangGraph's `MemorySaver`, so history
  is lost on restart, is not shared between uvicorn workers, and is never evicted. A SQLite
  or Postgres checkpointer is the fix if it needs to survive restarts or run on more than
  one worker.

### The API (`src/api/server.py`)

`POST /chat` takes:

| Field | Required | Meaning |
| --- | --- | --- |
| `message` | yes | The user's question |
| `thread_id` | no | Identifies the conversation; omit it for a one-off question with no memory |
| `selected_tile` | no | The tile open in the viewer, passed to the agent as a `[Viewer: tile … is open]` prefix on the message |

The response is `reply` and the `thread_id` that was used. A `selected_tile` that does not
look like a tile ID gets a 422. `GET /` is a health check.

### The frontend (`app/main.py`)

The left column draws the whole tile with ground truth in green and predictions in red over
the RGB raster, plus the DuckDB metrics, including how many cells failed.

With **Show where the model did badly** ticked, each scored grid cell is shaded blue by how
much of its trail the model missed or wrongly predicted: the darker the cell, the worse.
Cells below the pass threshold are also outlined in white and labelled with their name, so
a failure is never marked by shade alone. The name is the one the agent uses, so you can
type *"look at r1c5"* in the chat. Cells with no trail in them were never scored and are
left clear. When more than 20 cells fail, all are outlined and the worst 20 are named. The right column is the chat, which posts to the API. It
sends one `thread_id` per browser session and the selected tile with every message; **New
conversation** starts a fresh thread.

## LangSmith tracing

Tracing is off unless you turn it on. With `LANGCHAIN_TRACING_V2` unset or `false`, the
`@traceable` decorators do nothing.

To turn it on, fill in `LANGCHAIN_API_KEY` in `.env` (from smith.langchain.com → Settings →
API Keys). The project named in `LANGCHAIN_PROJECT` is created on the first trace.

To check the key works:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from langsmith import Client; print(list(Client().list_projects(limit=1)))"
```

The `load_dotenv()` matters: a bare `python -c` does not read `.env`, and `Client()` then
reports the key as missing.

LangChain and LangGraph instrument themselves. The parts that are not LangChain are
decorated by hand:

| Function | Why |
| --- | --- |
| `analyze_image_visually` | Parent span, so the encode step and the vision call sit in one subtree |
| `encode_and_resize_tiff` | Records the base64 *length* only; the payload is about 320 KB per call |
| `run_pipeline`, `train_and_explain` | The XGBoost and SHAP batch job, which involves no LLM |

A question that needs the image produces this trace:

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

In Docker the `api` service picks `.env` up through `env_file`. The frontend makes no LLM
calls, so it needs nothing.

To remove tracing, delete the decorators and the `from langsmith import traceable` imports.

## Evals

A prompt change that fixes one case can silently break another. `evals/` measures that
instead of leaving it to trying the chat by hand. The evals need LangSmith.

```bash
python evals/dataset.py
```

```bash
python evals/run_evals.py
```

The first command pushes the dataset; re-run it after editing the dataset. The second runs
the agent against it and creates a new LangSmith experiment each time, so two prompts can be
compared side by side. `--prefix` sets the experiment name prefix (default `react-agent`).

The dataset lives in `evals/dataset.py`, so it is diffable and reviewable. Reference IoU
values are read from DuckDB when the dataset is pushed, so re-running the pipeline on
different imagery does not invalidate them.

### What is tested

Nineteen examples covering six things:

| Area | What must hold |
| --- | --- |
| **Routing** | A metrics-only question does not open a 300 MB TIFF; a visual question does; a diagnostic question uses both tools |
| **Grounding** | The answer quotes the IoU that is in DuckDB, not a plausible nearby number |
| **Limits** | A tile that does not exist, and a question (weather) that no tool can answer, are admitted rather than answered |
| **False premises** | Four questions claim that a tile scoring 0.95 or more failed; the agent must contradict them |
| **Verdict** | A tile that failed is called failed and a tile that passed is called passed, on either side of the 0.75 threshold |
| **Conversation** | Six cases where the tile is not in the question and must come from earlier turns or the viewer |

In the conversation cases the earlier turns are replayed on one thread and only the final
turn is scored. `correct_tile` checks that every tool call in it named the right tile.

### How it is scored

Six of the eight scorers are code. Tool routing and quoted numbers have exact answers that
can be checked against the database, so an LLM judge would be slower, cost money and add
noise.

The other two are LLM judges (`judge_pushback` and `judge_verdict`), because whether an
answer accepted a false premise is a question of stance. Two string-matching versions of
that check scored sycophantic answers as passes: one missed "indeed experienced significant
issues", the next missed "did not perform well". Each fix only taught the scorer the wording
it had already seen. The judge is given the true IoU, so it grades against the database, and
it abstains when its reply cannot be parsed.

A rate-limited example is retried after 15, 30 and 60 seconds (`evals/retry.py`), replaying
the conversation on a fresh thread. If it is still rate limited after that,
`no_agent_error` abstains instead of scoring 0.

### What the evals caught

| Fault | Cause | Fix |
| --- | --- | --- |
| Asked whether a tile with an IoU of 0.9787 "failed badly", the agent agreed, then quoted the high IoU underneath | The system prompt said "when a user asks why a tile failed, first get the SHAP metrics", which presupposes the failure | The prompt now checks the IoU before accepting the framing. False-premise cases went from 1/3 to 4/4 |
| The answer led with image "issues" for a tile that passed | The vision tool's own prompt asked what "could explain its performance", so it listed problems for a tile scoring 0.98 | The vision prompt now says the tile may have scored well. 3/5 became 6/6 |
| A tile ID typed in the question lost to the tile open in the viewer | The prompt did not rank the two | The prompt sets the order. 5/8 became 8/8 |
| With no tile anywhere, the agent looked up a tile called "X" | The prompt's example was `[Viewer: tile X is open]`, and the model took the placeholder for a real ID | The placeholder is gone. Failing 1 run in 3 became 10/10 |
| Asked why the worst tile (IoU 0.5147) failed, the agent said it "did not fail" in 12 of 12 runs | The fixes above told it to push back on failure claims, and nothing told it what failing meant | The metrics tool returns the verdict, and `judge_verdict` scores both directions. 12 of 12 then gave the right verdict |

The numbers after the first row are repeated local runs of single cases, not full LangSmith
experiments. They show the fixes work, not that the cases can never fail.

The last row is the one to remember. The suite passed 17/17 while the agent was wrong,
because every false-premise case tested a user wrongly claiming failure and none tested a
user rightly claiming it.

### Testing the tests

A suite that has never failed may be unable to fail. `tests/test_evaluators.py` feeds the
code scorers hand-written outputs containing the failures they exist to catch — a
hallucinated IoU, a fabricated score for a nonexistent tile, an unnecessary vision call —
and asserts they score those 0. The judges' model call cannot be tested deterministically,
so the response parsing is split out and tested directly.

The unit tests need no API key:

```bash
pytest tests/ -q
```

## Layout

```text
data/                      TIFFs, ground truth zips, prediction zips, metrics.duckdb
src/
  api/server.py            FastAPI, holds one agent instance and its conversation memory
  agent/
    graph_agent.py         the LangGraph ReAct agent and its two tools
    history.py             the window of recent turns sent to the LLM
    verdict.py             the IoU threshold below which a tile fails
    cells.py               describes a tile's weak areas from its grid cells
    tiles.py               what a tile ID may look like
    vision_tool.py         TIFF → resized JPEG → base64, and the vision call
    qa_agent.py            earlier AgentExecutor version, unused
  metrics/
    pipeline.py            per-tile and per-cell metrics → XGBoost → SHAP → DuckDB
    shapefiles.py          finds the .shp inside a zipped export, for the pipeline and viewer
    visualizer.py          the matplotlib map Streamlit renders: trails and shaded cells
    image_extractor.py     standalone image feature extraction, unused
    spatial_calculator.py  standalone polygon error helpers, unused
  xai_engine.py            QATriageEngine, a classifier variant, unused
app/main.py                Streamlit dashboard and chat
evals/
  dataset.py               eval cases, versioned in git
  evaluators.py            six code scorers plus two LLM judges
  run_evals.py             runs the agent against the dataset
  retry.py                 waits out OpenAI rate limits
tests/
  test_cells.py            weak areas are counted, located and never called an IoU
  test_evaluators.py       proves the scorers can fail
  test_history.py          the history window never orphans a tool result
  test_pipeline.py         image metrics ignore padding; cells are scored and placed correctly
  test_retry.py            rate-limit retries
  test_shapefiles.py       zipped shapefiles are found at the root or in a folder
  test_tiles.py            paths and sentences are not tile IDs
  test_vision_crop.py      a cell is cut from the right place and keeps its detail
  test_verdict.py          the pass/fail cut-off and its scorer
  test_visualizer.py       failing cells are shaded, outlined and named on the map
```

## Limitations

- **The SHAP model has few tiles behind it.** It trains on grid cells, so it has far more
  rows than tiles, but cells from the same tile share lighting and ground cover and are not
  independent samples. Treat the SHAP output as a guide to where to look, not a calibrated
  model. More tiles is still the biggest improvement available.
- **Attributes that move together share the blame arbitrarily.** A shadowed cell is also a
  dark one, so SHAP may credit `brightness` for what `shadow_fraction` describes, or the
  reverse.
- **Rebuild the database after pulling a pipeline change.** A database built before the
  per-cell metrics has no `cell_metrics` table, and its SHAP values came from a model
  trained on whole tiles. The eval reference values are read from the database, so push
  the dataset again afterwards (`python evals/dataset.py`).
- **Unreadable tiles are skipped, not scored 0.** An IoU of 0.0 cannot be told apart from a
  prediction that missed. Zipped shapefiles are read whether the `.shp` is at the archive
  root or inside a folder named after the tile.
- **The API has no authentication.** It is a local single-user tool. Tile IDs are still
  treated as untrusted: the SQL binds them as parameters, and `run_vision_analysis` and
  `selected_tile` reject anything that is not letters, digits, hyphens and underscores
  (`src/agent/tiles.py`).
- **Pinned to the LangChain 0.2 line.** `langchain-core` must be `>=0.2.27` because every
  `langgraph` 0.2.x requires it; pinning core to 0.2.11 makes the requirements unsolvable.
