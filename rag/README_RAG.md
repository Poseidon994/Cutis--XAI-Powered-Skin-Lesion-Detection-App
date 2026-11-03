# RAG in Cutis: a guide from zero

> This add-on is **integrated** into Cutis now. The library lives in `rag/` at the project root, the retrieval
> code in `ml-service/app/rag/` (mounted at `/rag/retrieve` inside the model service, behind the same shared
> secret), the query builder and citation checker in `server/src/services/{rag,citations}.js`, and the citation
> markers plus source lists are rendered by `web/src/components/{AgentPanel,Chat,Bits}.jsx` and the PDF report.
> With Docker you do not need the manual steps below: `docker compose run --rm rag-build` fetches and indexes
> everything. Everything else in this guide still applies unchanged.

## 1. The idea in one minute

A language model writes fluent text from what it memorised, and it can state a drug, a margin or a follow-up
interval that is wrong with total confidence. **RAG (retrieval-augmented generation)** changes the job:

1. **Retrieve**: find the few passages in a trusted library that are relevant to this case.
2. **Generate**: give the model those passages and say "write the report using only these, and mark each
   statement with the passage that supports it".
3. **Check**: code (not the model) verifies the markers, drops uncited medicine statements, and shows the sources.

So the model stops being the *authority* and becomes the *writer*. The authority is the library, and every
claim can be traced to a page you can open.

## 2. What was built

```
 sources.yaml ──► rag_fetch ──► corpus/text/*.md ──► rag_build ──► index/
 (what to use)    (download,     (clean text)         (chunk, embed,  (chunks.jsonl, embeddings.npy, ...)
                   clean)                              keyword index)
                                                              │
   case (tier, type, symptoms) ──► queries ──► /rag/retrieve ─┘──► passages P1..P6 ──► Groq writes the report
                                                                                     │
                                                    checker: markers valid? medicines cited? sources shown
```

**Chunking**: each page is cut into pieces of about 220 words that follow the headings, because a model needs a
paragraph with its context, not a whole 30-page document. Reference lists and page boilerplate are dropped.

**Two ways of finding a passage, combined:**

| | finds | misses |
|---|---|---|
| **Keyword search (BM25)** | exact words: *imiquimod*, *Mohs*, *sentinel* | "spot that bleeds" vs "ulceration" |
| **Meaning search (embeddings)** | the same idea in different words | rare exact terms, numbers |

Each gives a ranking; **reciprocal rank fusion** adds `1/(60+rank)` from both lists, so a passage that both
searches like wins. This is called hybrid search and is the standard robust choice for medical text.

**Guards that matter for this app**
- Patients are only ever shown passages from `patient` or `both` sources; clinician-only text cannot reach them.
- Patient queries never contain the predicted type or any score.
- Patient reports have cancer-type names replaced by "skin cancer" in code, whatever the model wrote.
- A passage must clear a relevance floor, otherwise nothing is returned and the report says it is not grounded.
- If the library or the service is down, the app falls back to the standard text and says so. It never blocks a report.

## 3. First run on Windows (PowerShell)

You need the `ml-service` virtual environment from before. Activate it, then:

```powershell
cd C:\path\to\cutis\ml-service
.\.venv\Scripts\Activate.ps1
pip install -r requirements-rag.txt
```

**Step 1: download the library** (about 10 pages, takes under a minute):
```powershell
python -m scripts.rag_fetch
```
Every source prints its word count. A page under about 300 words was probably blocked or came out as menus: it is
flagged. Fix by saving the page from your browser (Print, Save as PDF) into `rag\corpus\manual\` and changing that
entry in `rag\sources.yaml` from `url:` to `file: name.pdf`. A page that errors with 404 simply has a different
address now: open it in the browser and correct the `url:`.

**Step 2: build the index** (the first run downloads a 70 MB embedding model):
```powershell
python -m scripts.rag_build
```
It prints chunks per source. If it says "keyword-only", the model could not be loaded; the warning above it says why.
The app still works in that mode, just with weaker matching of paraphrases.

**Step 3: play with it.** This is the best way to understand what the system does:
```powershell
python -m scripts.rag_query "excision margins for a thin melanoma"
python -m scripts.rag_query "a sore that does not heal" --audience patient
python -m scripts.rag_query            # interactive, empty line to quit
```
Each result shows `rrf` (combined score), `cos` (meaning similarity), `bm25 #n` and `dense #n` (rank in each search)
and `terms` (how many of your words it contains). Try the same idea in two phrasings and watch which search rescues it.

**Step 4: measure it.**
```powershell
python -m scripts.rag_eval
```
It runs the questions in `rag\eval\gold_questions.yaml` and prints:
- **hit@5**: for how many questions an answer-bearing passage is in the top 5. Aim for 0.85 or more. Misses mean a
  missing document, so add a source rather than tuning.
- **MRR**: how near the top the first good passage sits (1.0 is perfect).
- **clean on out-of-scope**: pizza, bicycles and diabetes must return nothing. This protects you from the
  library "answering" things it does not cover.
- **calibration**: the cosine range of in-scope vs out-of-scope questions, and a suggested `RAG_MIN_COSINE`.
  Set it in `.env` if the suggestion differs from the default 0.55.

## 4. Knobs (all optional, set in `.env`)

| variable | default | meaning |
|---|---|---|
| `RAG_ENABLED` | 1 | 0 switches the library off everywhere |
| `RAG_EMBEDDER` | auto | `none` = keyword only, `fastembed`, `st` (sentence-transformers) |
| `RAG_MIN_COSINE` | 0.55 | relevance floor for the meaning search (use the eval's suggestion) |
| `RAG_MIN_TERMS` | 3 | relevance floor for keyword matches |
| `RAG_MAX_PASSAGES` | 6 | passages sent to the model per report; more costs tokens and rate limit |
| `RAG_MAX_PER_SOURCE` | 2 | no single document may fill the answer |
| `RAG_DIR` | `rag/` | where the library lives |

After rebuilding the index you do not need to restart anything: the service notices the new `index\meta.json`.

## 5. Adding documents

1. Download the PDF or save the page into `rag\corpus\manual\`.
2. Add an entry to `rag\sources.yaml`: next free id (`S11`), `title`, `publisher`, `audience`, `topics`, `licence`,
   and `file: yourfile.pdf`.
3. `python -m scripts.rag_fetch --only S11`, then `python -m scripts.rag_build`, then `python -m scripts.rag_eval`.

Read the licence first. Free to reuse with credit: NCI and CDC text (what is used now). Not to be added:
NCCN guidelines, UpToDate, or any page marked "all rights reserved". Open-access journal guidelines often allow
non-commercial reuse with credit but forbid changes: fine for excerpts shown with a link, not for rewriting.

## 6. Limits you should know about

- The library is only as good as its documents. PDQ summaries are well maintained but US-centred and not a
  substitute for the current European or British guidelines. Adding those is the single best next improvement.
- The page text was not fetched in my test environment (no access to those sites), so the extraction rules were
  tested on sample pages only. Check the word counts in Step 1 and the passages in Step 3 with your own eyes once.
- Retrieval quality with the real embedding model is **unmeasured until you run Step 4**.
- A cited statement means "this passage supports it", not "this is clinically correct for this patient". The report
  stays a draft for a clinician to check.
- **Credit and links.** NCI asks for credit as the source and a link to the original page under its original title,
  so the Sources panel and the PDF must show the title as a link. The NCI logo and the PDQ mark may only appear on
  unchanged NCI products: never label Cutis's own generated text as "PDQ", and do not use either logo.
  Do not imply that NCI or CDC endorses Cutis.
- **The library is a snapshot.** These pages are updated over time. Re-run `rag_fetch --force` and `rag_build` every
  month or two, and keep the build date visible (the stored report records it).

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `ModuleNotFoundError: app` | run the commands from the `ml-service` folder |
| fetch: `403` or very few words | the site blocked the script: save the page by hand, use `file:` |
| build: "keyword-only index" | no model: `pip install fastembed`, check the internet, rerun `rag_build` |
| hf_hub symlink warning on Windows | harmless: files are copied instead of linked |
| query finds nothing | the relevance floor rejected everything: try `--all` to see scores, then adjust the knobs |
| `rag_eval` shows a MISS | open `rag_query` with that question; if the right text is simply absent, add a source |
