# Updated files — Cutis with the RAG add-on integrated and MySQL as the database

This is the **Cutis** project after two changes were applied on top of the original
`cutis.zip`:

1. **RAG add-on integrated** exactly the way `cutis_rag_addon/INTEGRATION.md` prescribes:
   retrieval lives as a router inside the existing FastAPI model service (behind the same shared
   secret), the guideline library lives in `rag/`, and Express gets two small modules
   (`services/rag.js`, `services/citations.js`).
2. **MySQL instead of SQLite.** The server's data layer (`server/src/db.js`) was rewritten from
   `node:sqlite` to `mysql2` (MySQL 8 or MariaDB 10.6+), with Docker Compose, configuration,
   scripts and tests updated to match.

No file was removed. **26 files were added, 28 were changed** (listed below).

---

## 1. What you get

`cutis-rag-mysql.zip` — the **complete, ready-to-run project** (all 106 files: every original
Cutis file, updated where needed, plus the 26 new RAG/MySQL files and this manifest). Unzip it
and you have everything; nothing else from the original `cutis.zip` is needed.

## 2. How to run

```bash
cp .env.example .env      # fill JWT_SECRET, ML_SERVICE_KEY, MYSQL_PASSWORD (+ GROQ_API_KEY if you have one)
docker compose up --build
docker compose run --rm rag-build     # once: fetches the guideline pages and builds rag/index
```

Without Docker you need a reachable MySQL server:

```sql
CREATE DATABASE cutis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'cutis'@'%' IDENTIFIED BY '...'; GRANT ALL PRIVILEGES ON cutis.* TO 'cutis'@'%';
```

then `npm install` in `server/` and `web/`, install `ml-service/requirements.txt` +
`requirements-rag.txt` plus torch, build the library once
(`python -m scripts.rag_fetch && python -m scripts.rag_build`), and run `./dev.sh`.
The tables are created automatically on first start. See `README.md` sections 3 and 11.

## 3. MySQL changes in detail

- **`server/src/db.js`** — rewritten on `mysql2/promise`: pool with `utf8mb4`, InnoDB tables
  (`users`, `scans`, `messages`), `JSON` columns for the JSON-shaped fields (`symptoms`,
  `prediction`, `xai`, `rag`, `agent_report`, …), foreign keys with `ON DELETE CASCADE`,
  `KEY` indexes for the list queries. All `Store` methods are now `async`. Timestamps are
  written/read as UTC literals (`dateStrings: true`, `SET time_zone = '+00:00'`) so no
  session timezone can shift a date. Duplicate-email registration races are caught through
  `ER_DUP_ENTRY`.
- **`server/src/config.js`** — new `db` block (`DB_HOST/PORT/USER/PASSWORD/NAME`, `DB_POOL`) and
  a `rag` block (`RAG_ENABLED`, `RAG_MAX_PASSAGES`, `RAG_TIMEOUT_MS`).
- **`server/src/index.js` / `app.js`** — the store is built from `config.db`, the server exits
  with a clear message if MySQL is unreachable, and `/api/health` reports `db: "ok" | "error: …"`.
- **`server/src/middleware/auth.js`, `routes/auth.js`, `routes/scans.js`, `services/context.js`**
  — call sites converted from synchronous to `await` (the store is async now). `routes/scans.js`
  additionally serialises per-scan read-modify-write sections with a small lock map, which the
  synchronous store used to give for free.
- **`server/package.json` / `package-lock.json`** — dependency `mysql2 ^3.15.3` added.
- **`docker-compose.yml`** — new `db` service (MySQL 8.4, healthcheck, `cutis-mysql` volume);
  the `server` service waits for it (`condition: service_healthy`) and points `DB_*` at it.
- **`.env.example`** — all `DB_*` / `MYSQL_*` variables documented, plus the `RAG_*` block.
- **`server/test/e2e.test.js`** — the suite is async, wipes/recreates the schema per run through
  `TEST_DB_*` (default `127.0.0.1:3306`, user `cutis`, database `cutis_test`), and gains a
  MySQL cascade-deletion test.

## 4. RAG integration in detail (per INTEGRATION.md)

**Added — Python side (`ml-service/`)**
- `app/rag/api.py` — FastAPI router: `POST /rag/retrieve`, `GET /rag/status`, `POST /rag/reload`,
  all behind the same `X-Service-Key` shared secret as `/predict`.
- `app/rag/{service,retriever,store,bm25,chunking,embedders,textclean,config}.py` — the retrieval
  stack: manifest validation, HTML/PDF → markdown cleaning, markdown-aware chunking, BM25 +
  embedding hybrid search with reciprocal-rank fusion, audience filter, per-source cap,
  relevance floor, hot reload of a rebuilt index, `fastembed` → `sentence-transformers` → `none`
  embedder fallback.
- `scripts/{rag_fetch,rag_build,rag_query,rag_eval}.py` — fetch the pages from `sources.yaml`,
  build `rag/index`, query it from the CLI, evaluate against the gold questions.
- `requirements-rag.txt` — numpy, pyyaml, httpx, beautifulsoup4, pypdf, fastembed.
- `tests/test_rag.py` — 32 tests (chunking, ranking, audience filter, floor, reload, API, scripts).

**Added — library (`rag/`)**
- `sources.yaml` (NCI PDQ + CDC pages with licence and audience), `README_RAG.md`,
  `eval/gold_questions.yaml`, `corpus/` and `index/` skeletons.

**Added — Node side (`server/`)**
- `src/services/rag.js` — builds the role-specific queries (patients: generic questions steered by
  symptoms only, never the predicted type or scores; clinicians: one query per report section with
  the predicted type steering management/medication) and the retrieval client that never throws.
- `src/services/citations.js` — the passages block for the prompt, the citation rules, and the
  checker that runs in code after the model answers: unknown markers removed, uncited medicine
  statements dropped, patient answers scrubbed of cancer-type names, sources listed with excerpt.
- `test/rag.test.js` — server-side RAG tests.

**Changed — wiring**
- `ml-service/app/main.py` — mounts the RAG router; `/health` reports the library state.
- `server/src/routes/scans.js` — retrieval starts the moment the prediction is stored (it needs
  nothing else) and runs **in parallel** with Grad-CAM/LIME/SHAP; the result is stored on the scan;
  chat does one retrieval per question and the answer is checked before being stored.
- `server/src/services/agent.js` — passages + citation rules appended to the prompt, the answer
  validated by `citations.js`, `sources[]` and `grounding{}` stored with the report; the template
  fallback and all fixed guarantees are untouched.
- `server/src/services/context.js` — the case context gains `rag{status, corpus_sha, built_at,
  n_passages}` (passages go in the prompt, not the JSON).
- `server/src/services/report.js` — the PDF gains a "Sources" section (linked titles, publisher,
  licence, excerpt, grounding line) and NCI/CDC credit; Cutis's own text is never labelled PDQ.
- `web/src/components/{AgentPanel,Bits,Chat}.jsx`, `styles.css` — `[P1]`/`[P2]` markers render as
  links into the source list, a grounding badge, an "not checked against guidelines" banner, and
  sources under chat answers.
- `web/src/__tests__/` — fixtures and page tests extended to cover sources, markers, grounding.

## 5. Complete file list

**Added (26)**
```
ml-service/app/rag/__init__.py          ml-service/app/rag/api.py
ml-service/app/rag/bm25.py              ml-service/app/rag/chunking.py
ml-service/app/rag/config.py            ml-service/app/rag/embedders.py
ml-service/app/rag/retriever.py         ml-service/app/rag/service.py
ml-service/app/rag/store.py             ml-service/app/rag/textclean.py
ml-service/requirements-rag.txt         ml-service/scripts/rag_build.py
ml-service/scripts/rag_eval.py          ml-service/scripts/rag_fetch.py
ml-service/scripts/rag_query.py         ml-service/tests/test_rag.py
rag/.gitignore                          rag/README_RAG.md
rag/corpus/manual/.gitkeep              rag/corpus/text/.gitkeep
rag/eval/gold_questions.yaml            rag/index/.gitkeep
rag/sources.yaml
server/src/services/citations.js        server/src/services/rag.js
server/test/rag.test.js
```

**Changed (28)**
```
.env.example                            README.md
dev.sh                                  docker-compose.yml
ml-service/Dockerfile                   ml-service/app/main.py
server/package-lock.json                server/package.json
server/src/app.js                       server/src/config.js
server/src/db.js                        server/src/index.js
server/src/middleware/auth.js           server/src/routes/auth.js
server/src/routes/scans.js              server/src/services/agent.js
server/src/services/context.js          server/src/services/report.js
server/test/e2e.test.js
web/src/__tests__/fixtures/clinician-list.json
web/src/__tests__/fixtures/clinician-scan.json
web/src/__tests__/fixtures/patient-list.json
web/src/__tests__/fixtures/patient-scan.json
web/src/__tests__/pages.test.jsx
web/src/components/AgentPanel.jsx       web/src/components/Bits.jsx
web/src/components/Chat.jsx             web/src/styles.css
```

## 6. Verified in this environment

- `server` — **50/50 tests** against a real MySQL-compatible server (MariaDB 11.4):
  auth/roles, upload, parallel explanations, agent with stubbed Groq, safety, history, PDF,
  MySQL cascade deletion, and RAG end-to-end (ok / empty / unavailable), citation check,
  dropped medicine line, chat sources.
- `ml-service` — **46/46 tests** (14 original + 32 RAG).
- `web` — **10/10 tests** (both roles' pages render sources, markers and grounding badges).
- Live smoke test: the Express API booted against MySQL, created the `users` / `scans` /
  `messages` schema automatically and reported `db: "ok"`; the Node retrieval client talked to
  the running Python service (`status: ok`, correct passages and corpus sha, patient audience
  filter applied, missing service key rejected with 401).

**Not verifiable here** (as documented in `README.md` section 9): real model weights, the live
Groq API, the real embedding model's retrieval quality (run `python -m scripts.rag_eval` after
building the library), downloading the actual NCI/CDC pages (network-restricted sandbox), Docker
builds and GPU runs.
