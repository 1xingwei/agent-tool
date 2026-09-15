# Interview Hardening Design

Date: 2026-09-15
Status: Approved
Project: agent-service-toolkit (interview showcase)

## Goal

Close three weaknesses in the repo so it holds up as an interview showcase for LangGraph:
an architecture that is defensible when a senior interviewer probes past the default
demos. Each change is independently testable, small, and zero new runtime dependencies.

## 1. RAG Vector Ingestion Pipeline

**Problem:** `src/agents/tools.py:50-76` reads a pre-existing `./chroma_db` directory. An
ingestion script DOES exist (`scripts/create_chroma_db.py`) but it is naive: it deletes and
rebuilds the whole index each run (`shutil.rmtree`, line 24-26), supports only `.pdf`/`.docx`,
and hardcodes paths. There is no `md`/`txt` support and no incremental update, so the repo
can't claim a defensible ingestion story.

**Approach:** Upgrade the existing script (reuse, don't re-create) + configure the loader.

- Rework `scripts/create_chroma_db.py`:
  - Keep the existing chunk→embed→Chroma pipeline and `RecursiveCharacterTextSplitter`
  - Extend format support: `.md`/`.txt` via plain text reader, `.pdf` via `PyPDFLoader`,
    `.docx` via `Docx2txtLoader` (all loaders already imported or transitively available)
  - Make destructive `delete_chroma_db` opt-in (default off) instead of default-true
  - Paths/collection from `settings` (`CHROMA_DIR`), fall back to `./chroma_db`
  - (YAGNI: skip per-file hash manifest — full rebuild is fine for a demo corpus; note in
    code comment that hash-based incremental indexing is the upgrade path)
- Refactor `src/agents/tools.py`:
  - `load_chroma_db()` no longer hardcodes `./chroma_db`; reads `CHROMA_DIR` from
    `core.settings`, falls back to existing default

### Data flow

```
data/*.{md,pdf,txt,docx}
        │ scripts/create_chroma_db.py (upgraded)
        ▼
 RecursiveCharacterTextSplitter (2000/500, existing defaults)
        │ OpenAIEmbeddings
        ▼
 Chroma (CHROMA_DIR) ──► retriever (k=5) ◄── rag_assistant Database_Search tool
```

## 2. Long-Term Memory SQLite Persistence

**Problem:** `src/memory/sqlite.py:4,36-40` wraps LangGraph's `InMemoryStore` for the SQLite
database type — long-term memory is lost on restart. The file is named `sqlite.py` but
persists nothing.

**Approach:** Swap the `InMemoryStore` wrapper for LangGraph's official `AsyncSqliteStore`.

- `langgraph.store.sqlite.AsyncSqliteStore` ships INSIDE the installed `langgraph` package
  (verified: `langgraph/store/sqlite/__init__.py`) — no new dependency, no custom store to write.
  It implements the exact interface the codebase already uses: `from_conn_string(conn_string)`
  as an async context manager, `setup()`, `aget(namespace, key)`, `aput(namespace, key, value)`.
  Round-trip + persistence across close/reopen verified by a live run (value survives reopen).
- Rework `src/memory/sqlite.py`:
  - `get_sqlite_store()` becomes a thin `@asynccontextmanager` wrapping
    `AsyncSqliteStore.from_conn_string(settings.SQLITE_STORE_PATH)`, `await store.setup()`, yield
  - Delete the `AsyncInMemoryStore` wrapper and its `InMemoryStore` import
- Add `SQLITE_STORE_PATH: str = "memory_store.db"` to `core/settings.py` (distinct from the
  checkpointer's `SQLITE_DB_PATH = "checkpoints.db"`)
- `initialize_store()` (`src/memory/__init__.py:28-37`) and `service.py` lifespan: unchanged —
  they already `await store.setup()` and use `async with`

### Why this shape

- Same interface as `AsyncPostgresStore` → the lifespan wiring in `service.py` is untouched
- Zero new code for the store itself; the entire fix is the wrapper swap + one settings field
- The sqlite.py `checkpointer` provider (`get_sqlite_saver`) already uses
  `AsyncSqliteSaver.from_conn_string` — `get_sqlite_store` now mirrors it exactly, so the story
  is "short-term checkpointer and long-term store now both backed by the same SQLite file"
- Verified live: `AsyncSqliteStore` persists values across store close/reopen

## 3. Supervisor Real Search Tool

**Problem:** `src/agents/langgraph_supervisor_agent.py:21-30` — `web_search` returns hardcoded
FAANG headcounts. Reads as a demo fake; an interviewer asking "show me the tool" hits the end.

**Approach:** Replace the fake with the real DuckDuckGo search already used by
`research_assistant.py`.

- In `langgraph_supervisor_agent.py`: `web_search` becomes a thin wrapper over
  `DuckDuckGoSearchResults` (already imported in research_assistant.py)
- Keep `sub-agent-math_expert` (add/multiply) and `sub-agent-research_expert` structure
  and the `create_supervisor` handoff wiring unchanged — only the tool backend becomes real
- `langgraph_supervisor_hierarchy_agent.py` imports `web_search` from the flat agent
  (`:4`) — benefits automatically, no change there

## Out of scope (YAGNI)

- Multi-vector-DB abstraction (Chroma/FAISS/LanceDB) — no interview narrative gain
- git-based auto-sync for docs — overkill; manifest hash is enough
- Postgres migration of the existing path — orthogonal; both DB paths keep working
- Cross-agent shared state — not part of the three goals

## Testing

- `tests/test_ingest.py`: build a tiny Chroma in a temp dir from 2 fixture docs (one md, one
  txt), assert retrieval returns a hit for a known query; assert `delete_chroma_db=False` still
  rebuilds/upserts without error (idempotent re-run)
- `tests/test_sqlite_store.py`: `async with AsyncSqliteStore.from_conn_string(tmp)` → aput +
  aget roundtrip → close → reopen same file → aget returns the value (proves persistence).
  The test asserts the wiring in `get_sqlite_store`/`initialize_store`, not the SDK itself
- Supervisor: smoke check that `langgraph_supervisor_agent` compiles and its `web_search`
  tool invokes (mock-free; network required — mark `@pytest.mark` so CI default can skip)
- Run: `uv sync --frozen && pytest`

## Files touched

- `scripts/create_chroma_db.py` (rework, not new)
- `src/agents/tools.py` (configurable loader)
- `src/memory/sqlite.py` (swap InMemoryStore wrapper for official AsyncSqliteStore)
- `src/agents/langgraph_supervisor_agent.py` (real web_search)
- `src/core/settings.py` (CHROMA_DIR / SQLITE_STORE_PATH)
- `tests/test_ingest.py`, `tests/test_sqlite_store.py` (new)