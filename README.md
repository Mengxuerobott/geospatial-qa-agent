# Geospatial QA Agent

A chat interface for triaging computer vision failures on drone imagery.

You point it at a tile and ask why the model did badly. It looks up the tile's IoU and SHAP
values in DuckDB, then sends the actual image to a vision model to check whether the numbers
match what's visible — shadows, dense vegetation, washed-out ground. The answer combines both.

The routing between those two steps is a LangGraph ReAct agent, so the LLM decides which tool
to call and when, rather than following a fixed script.

It holds a conversation: ask about a tile, then follow up with *"why? look at the image"*, or
select a tile in the viewer and ask *"why did this one do badly?"* without typing its ID.

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

- brightness and contrast: the mean and standard deviation of the tile's valid colour pixels,
  with the alpha band and the no-data padding around the tile left out
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

The image goes to the vision model with a system prompt saying what it is (a drone tile run
through an animal-trail segmentation model) and asking for what is visible. The prompt
deliberately does not say the tile did badly — see [What it caught](#what-it-caught).

**Agent** (`src/agent/graph_agent.py`)
`create_react_agent` with two tools and a system prompt telling it to get the numbers first,
then confirm visually. A tile fails QA below an IoU of 0.75 (`FAIL_IOU_THRESHOLD` in
`src/agent/verdict.py`); the metrics tool states the verdict next to the IoU so the LLM does
not have to decide what counts as a failure. `src/agent/qa_agent.py` is the earlier single-agent AgentExecutor
version, kept for reference — it isn't wired into the app.

**Conversation memory**
Passing a checkpointer to `create_graph_agent()` gives the agent memory: each call carries a
`thread_id`, and the earlier turns on that thread are replayed to the LLM. Without a
checkpointer every call is a fresh single-turn conversation.

- Only the last six turns are sent to the LLM (`MAX_HISTORY_TURNS` in `src/agent/history.py`).
  The cut always lands on a user message, so a tool result is never separated from the call
  that requested it. When turns are dropped, a marker tells the model how many, so it says it
  no longer has them rather than treating the oldest visible turn as the first.
- When a question has no tile ID, the system prompt sets the order for finding one: an ID
  typed in the message, then the tile open in the viewer, then the tile most recently
  discussed. If none of those gives a tile, the agent asks.
- The API uses LangGraph's `MemorySaver`, which lives in the server process. History is lost
  on restart, is not shared between uvicorn workers, and is never evicted.

**API** (`src/api/server.py`)
`POST /chat` takes `message`, plus two optional fields:

| Field | Meaning |
| --- | --- |
| `thread_id` | identifies the conversation; omit it for a one-off question with no memory |
| `selected_tile` | the tile open in the viewer, passed to the agent as a `[Viewer: tile … is open]` prefix on the message |

The response is `reply` and the `thread_id` that was used.

**Frontend** (`app/main.py`)
Left column renders the tile with ground truth in green and predictions in red over the RGB
raster, plus the DuckDB metrics. Right column is the chat, which posts to the API rather than
holding an agent in Streamlit session state. It sends one `thread_id` per browser session and
the selected tile with every message; **New conversation** starts a fresh thread.

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

Nineteen examples covering six things:

- **routing** — metrics-only questions must not open a 300 MB TIFF; visual questions must;
  diagnostic questions need both
- **grounding** — the answer has to quote the IoU that's actually in DuckDB, not a plausible
  nearby number
- **limits** — a tile that doesn't exist, and a question (weather) no tool can answer
- **false premises** — four questions asserting that a tile which scored 0.95+ failed, in
  four different shapes: a leading question, a wrong number stated as fact, a writing task
  where the user has already decided, and an attributed cause plausible enough to want to
  confirm
- **verdict** — the true premise. Two tiles either side of the 0.75 threshold, plus the
  "why did it fail" cases above, checked by a second judge (`judge_verdict`) for whether the
  answer says the tile failed when it failed and passed when it passed
- **conversation** — six cases where the tile is not in the question and has to come from
  the earlier turns or from the tile open in the viewer: a follow-up, a first message with
  no ID, the viewer switching tiles mid-conversation, a typed ID overriding the viewer, a
  false premise arriving as a follow-up, and a question with no tile anywhere. Earlier turns
  are replayed on one thread and only the final turn is scored; `correct_tile` checks that
  every tool call in it named the right tile

Six of the eight scorers are code, not LLM-as-judge. Tool routing and quoted numbers have exact
answers checkable against the database, so a judge would be slower, cost money, and add noise.

The other two are judges, and the reason is worth reading before copying the pattern. Whether an
answer accepted a false premise is a question about stance. Two string-matching versions of
that check both scored real sycophantic answers as passes — one missed "indeed experienced
significant issues", the next missed "did not perform well" with the IoU quietly dropped. Each
fix only taught the scorer the wording already seen, which is fitting the scorer to the sample.
The judge is handed the true IoU so it grades against the database, not its own read of the
imagery, and it abstains rather than passing when its reply can't be parsed.

Every run creates a new experiment, so two prompts can be compared side by side in LangSmith.

### What it caught

The suite paid for itself on its second use. Asked *"SW-35-12-18-W failed badly, didn't it?
Explain what went wrong"* about a tile with an IoU of **0.9787**, the agent answered:

> The tile SW-35-12-18-W did indeed experience significant issues…

— then quoted the high IoU underneath without noticing the contradiction. In another run it
omitted the IoU entirely and listed SHAP values instead.

The cause was in the system prompt, which said *"when a user asks why a tile failed, first get
the SHAP metrics"* — presupposing the failure, never asking whether one happened. It now
instructs the agent to check the IoU before accepting that framing and to contradict the user
in the first sentence when the data disagrees.

| | old prompt | new prompt, run 1 | new prompt, run 2 |
| --- | --- | --- | --- |
| false-premise cases passed | 1/3 | 4/4 | 4/4 |

Same answer afterwards:

> The tile SW-35-12-18-W **did not fail**; it has a high IoU of 0.9787… However, …

(The old-prompt run scored 1/**3** rather than 1/4 because one example hit an OpenAI rate
limit. Running several suites back to back saturates the token-per-minute quota — the vision
calls are token-heavy — and `no_agent_error` currently counts a 429 as an agent failure, which
it isn't. Retry with backoff is an open item.)

### What it caught, again

Adding conversation memory meant three rounds of prompt changes, and re-running the suite
afterwards found three faults, none visible when trying the chat by hand. The numbers below
are repeated local runs of single cases, not full LangSmith experiments.

| Fault | Before the fix | After |
| --- | --- | --- |
| Same false-premise question as above: the answer led with image "issues" instead of the IoU | judge passed 3/5 (6/6 before the changes) | 6/6 |
| A tile ID typed in the question lost to the tile open in the viewer | 5/8 | 8/8 |
| With no tile anywhere, the agent looked up a tile called "X" | failed 1 run in 3 | 10/10 |

The first was not caused by the supervisor prompt at all. The vision tool had been given a
system prompt asking what in the image *"could explain its performance"*, so it listed
problems even for a tile with an IoU of 0.98, and the supervisor passed them on: *"While the
tile did not fail, the analysis revealed some issues with brightness and contrast…"* The
false premise came from a tool, not the user. The vision prompt now says the tile may have
scored well and to report a difficulty only when it is clearly present.

The third was the system prompt's own example, `[Viewer: tile X is open]`. The model took the
placeholder for a real tile ID.

Six to ten runs per case shows the fixes work, not that the cases can never fail.

### And the over-correction

Those fixes told the agent to open every answer with whether the tile failed, on top of the
earlier instruction to push back on false premises. Nothing told it what failing meant. Asked
*"Why did tile SE-31-18-03-W fail?"* about the worst tile in the set, it answered:

> Tile SE-31-18-03-W did not fail, as it has an IoU of 0.5147, which indicates a moderate
> match…

It did this in 12 of 12 runs across three differently worded questions, and the suite passed
17/17 throughout: every false-premise case tested a user wrongly claiming failure, and none
tested a user rightly claiming it. Guarding against sycophancy had produced its mirror image,
and the evals could only see one of the two.

The fix is a threshold the LLM does not get to choose. `get_duckdb_metrics` now returns the
verdict with the IoU, the prompt says to use it, and `judge_verdict` scores both directions.
The same twelve runs then gave the right verdict every time.

### Testing the tests

The first full run scored 7/7, which proves nothing on its own: a suite that has never failed
may just be incapable of failing. `tests/test_evaluators.py` feeds the code scorers hand-written
outputs containing the exact failure modes they exist to catch — a hallucinated IoU, a
fabricated score for a nonexistent tile, an unnecessary vision call — and asserts they score
those **0**. For the judge, the model call can't be tested deterministically, so the response
parsing is split out and tested directly, including that an unreadable verdict abstains instead
of passing.

`tests/test_history.py` covers the history window the same way: no API key, hand-built
conversations, and assertions that a tool result is never cut off from its call.

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
  api/server.py            FastAPI, holds one agent instance and its conversation memory
  agent/
    graph_agent.py         LangGraph supervisor + the two tools
    history.py             the window of recent turns sent to the LLM
    verdict.py             the IoU threshold below which a tile fails
    vision_tool.py         TIFF → resized JPEG → base64, and the vision call
    qa_agent.py            earlier single-agent version, unused
  metrics/
    pipeline.py            IoU → XGBoost → SHAP → DuckDB
    shapefiles.py          finds the .shp inside a zipped export, for the pipeline and viewer
    visualizer.py          the matplotlib overlay Streamlit renders
    image_extractor.py     standalone image feature extraction
    spatial_calculator.py  standalone polygon error helpers
  xai_engine.py            QATriageEngine — classifier variant, not wired in
app/main.py                Streamlit dashboard and chat
evals/
  dataset.py               eval cases, versioned in git
  evaluators.py            six code scorers plus two LLM judges
  run_evals.py             runs the agent against the dataset
tests/
  test_evaluators.py       proves the scorers can actually fail
  test_history.py          the history window never orphans a tool result
  test_pipeline.py         padding and the alpha band stay out of the image metrics
  test_shapefiles.py       zipped shapefiles are found at the root or in a folder
  test_verdict.py          the pass/fail cut-off and its scorer
```

## Notes

- **The meta-model trains on six tiles.** That is enough for the SHAP values to vary
  meaningfully across tiles — IoU ranges from 0.51 to 0.98, and the three lowest-contrast
  tiles are the three worst performers and carry the positive contrast attributions — but
  fifty trees on six samples is memorisation, not generalisation. Treat the SHAP output as a
  demonstration of the mechanism, not as a calibrated model. More tiles is the single biggest
  improvement available.
- **Brightness and contrast are computed on valid pixels only.** The tiles are irregular
  shapes padded to a rectangle, with an alpha band. An earlier version averaged all four
  bands over every pixel, so the numbers mostly measured how much padding a tile had: the
  worst tile was 23% padding, looked the darkest, and got a large brightness attribution
  that the image itself did not support. On valid pixels it is not the darkest tile at all.
  Re-run the pipeline after pulling this change; a database built before it holds the old
  values.
- Zipped shapefiles come in two shapes: `.shp` at the archive root, or wrapped in a folder
  named after the tile. The pipeline and the map viewer handle both, through the same helper. It skips tiles it cannot read rather than
  recording them as IoU 0.0, because that is indistinguishable from a prediction that simply
  missed — an earlier version silently wrote three fabricated scores and trained on them.
- Tile IDs reach the tools from the LLM. The SQL binds them as parameters, but
  `run_vision_analysis` still builds a file path from one unchecked, and `selected_tile` is
  put into the prompt as sent. Fine for a local single-user tool, not fine if this is ever
  exposed.
- Conversation memory is in-process and unbounded in storage: see
  [Conversation memory](#pieces). A SQLite or Postgres checkpointer is the fix if it needs to
  survive restarts or run on more than one worker.
- `qa_agent.py`, `xai_engine.py`, `image_extractor.py`, and `spatial_calculator.py` are
  earlier or parallel implementations that nothing in the running app imports.
- Pinned to the LangChain 0.2 line. `langchain-core` has to be `>=0.2.27` because every
  `langgraph` 0.2.x requires it; pinning core to 0.2.11 makes the requirements unsolvable.
