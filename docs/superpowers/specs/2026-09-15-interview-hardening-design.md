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

**Approach:** Implement a real disk-backed store for the SQLite path.

- New `AsyncSqliteStore` in `src/memory/sqlite.py`:
  - Backed by `sqlite3` + `aiosqlite` (aiosqlite is already a core dependency)
  - Schema mirrors LangGraph's store table: `namespace`, `key`, `value` (JSON), timestamps
  - Implements the same async interface Postgres uses: `setup()`, `aget(namespace, key)`,
    `aput(namespace, key, value)`, `__aenter__`/`__aexit__`
  - Store file at `settings.SQLITE_DB_PATH.with_store_suffix` (e.g. `checkpoints_store.db`)
    or a configurable `SQLITE_STORE_PATH`
- `initialize_store()` (`src/memory/__init__.py:28-37`): SQLite branch returns
  `AsyncSqliteStore` instead of the `InMemoryStore` wrapper
- `service.py` lifespan already calls optional `setup()`/`__aenter__` — no change needed
- Delete the now-unused `AsyncInMemoryStore` wrapper

### Why this shape

- Same interface as `AsyncPostgresStore` → the lifespan wiring in `service.py` is untouched
- `sqlite3`/`aiosqlite` are stdlib/already-present; no new deps
- Values stored as JSON → survives restart; namespace isolation per user preserved
  (`interrupt_agent.py` reads/writes via the same `store.aget/aput` calls)

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
  txt), assert retrieval returns a hit for a known query; assert skip-when-unchanged and
  rebuild behavior
- `tests/test_sqlite_store.py`: create `AsyncSqliteStore` on temp file, put + get roundtrip,
  assert persistence across store close/reopen
- Supervisor: smoke check that `langgraph_supervisor_agent` compiles and its `web_search`
  tool invokes (mock-free; network required — mark `@pytest.mark` so CI default can skip)
- Run: `uv sync --frozen && pytest`

## Files touched

- `scripts/create_chroma_db.py` (rework, not new)
- `src/agents/tools.py` (configurable loader)
- `src/memory/sqlite.py` (replace InMemoryStore with AsyncSqliteStore)
- `src/agents/langgraph_supervisor_agent.py` (real web_search)
- `src/core/settings.py` (CHROMA_DIR / SQLITE_STORE_PATH)
- `tests/test_ingest.py`, `tests/test_sqlite_store.py` (new)