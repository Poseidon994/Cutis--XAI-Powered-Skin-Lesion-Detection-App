"""RAG layer tests. No model download and no network: a hashing embedder stands in for the real one,
so these prove the plumbing (chunking, fusion, filtering, reloads, API, scripts), not the search quality of
the real embedding model. Measure that with scripts/rag_eval.py on your own machine."""
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import api as rag_api  # noqa: E402
from app.rag.bm25 import BM25, stem, tokenize  # noqa: E402
from app.rag.chunking import chunk_document  # noqa: E402
from app.rag.config import RagConfig  # noqa: E402
from app.rag.embedders import get_embedder  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.rag.service import RagService  # noqa: E402
from app.rag.store import ManifestError, build_index, load_index, read_manifest  # noqa: E402
from app.rag.textclean import (  # noqa: E402
    _pdf_heading_level,
    clean_text,
    drop_repeated_lines,
    html_to_markdown,
    looks_like_reference,
)
from scripts import rag_build, rag_eval, rag_fetch, rag_query  # noqa: E402

quiet = lambda *_a, **_k: None  # noqa: E731


class HashingEmbedder:
    """Deterministic stand-in for a neural model: signed hashing of word stems into 128 dimensions."""

    name = "hash-test"
    dim = 128

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
        return v

    def embed_documents(self, texts):
        m = np.stack([self._vec(t) for t in texts])
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def embed_query(self, text):
        v = self._vec(text)
        return v / max(float(np.linalg.norm(v)), 1e-12)


MANIFEST = """
sources:
  - {id: S01, title: "Melanoma management", publisher: "Test Org", audience: clinician, topics: [melanoma], licence: "test", url: "https://example.org/s01"}
  - {id: S02, title: "Keratinocyte carcinoma", publisher: "Test Org", audience: clinician, topics: [bcc, scc], licence: "test", url: "https://example.org/s02"}
  - {id: S03, title: "Checking your skin", publisher: "Test Org", audience: patient, topics: [symptoms], licence: "test", url: "https://example.org/s03"}
  - {id: S04, title: "Sun protection", publisher: "Test Org", audience: both, topics: [prevention], licence: "test", url: "https://example.org/s04"}
"""

DOCS = {
    "S01": (
        "# Melanoma management\n\n## Surgical excision\n\n"
        "Wide local excision is the standard treatment for primary melanoma. Excision margins depend on tumour "
        "thickness, so thin lesions need narrower margins than thick lesions.\n\n"
        "## Sentinel lymph node biopsy\n\n"
        "Sentinel lymph node biopsy is offered to selected patients with intermediate thickness melanoma to stage "
        "the regional lymph nodes.\n\n"
        "## References\n\n1. Smith J, Lee K, et al. Margins in melanoma. J Surg 2015;12(3):45-52. doi:10.1/x PMID 1\n"
    ),
    "S02": (
        "# Keratinocyte carcinoma\n\n## Basal cell carcinoma\n\n"
        "Mohs micrographic surgery is used for high risk basal cell carcinoma on the face. Superficial lesions may "
        "be treated with topical imiquimod or fluorouracil cream.\n\n"
        "## Squamous cell carcinoma\n\n"
        "Cutaneous squamous cell carcinoma with perineural invasion or deep tumour depth is high risk and needs "
        "wider excision and sometimes radiotherapy.\n"
    ),
    "S03": (
        "# Checking your skin\n\n## When to see a doctor\n\n"
        "See a doctor if a mole changes in size, shape or colour, bleeds, itches or becomes a sore that does not "
        "heal.\n\n"
        "## The ABCDE guide\n\n"
        "Asymmetry, border, colour, diameter and evolving are the ABCDE signs to look for when you check a mole.\n"
    ),
    "S04": (
        "# Sun protection\n\n## Sunscreen and clothing\n\n"
        "Use broad spectrum sunscreen, wear a hat and cover up between ten and four to reduce ultraviolet "
        "exposure from the sun.\n"
    ),
}


def make_cfg(tmp_path, **kw):
    rag = tmp_path / "rag"
    (rag / "corpus" / "text").mkdir(parents=True)
    (rag / "sources.yaml").write_text(MANIFEST, encoding="utf-8")
    for sid, text in DOCS.items():
        (rag / "corpus" / "text" / f"{sid}.md").write_text(text, encoding="utf-8")
    # the stand-in embedder's noise on unrelated text reaches about 0.28 and a true match scores about 0.70,
    # so the floor sits between them; with the real model you pick it from scripts/rag_eval.py output
    return RagConfig(rag_dir=rag, embedder="none", min_cosine=0.5, **kw)


def built(tmp_path, dense=True, **kw):
    cfg = make_cfg(tmp_path, **kw)
    emb = HashingEmbedder() if dense else None
    build_index(cfg, emb, log=quiet)
    index, warnings = load_index(cfg, emb)
    assert warnings == []
    return cfg, Retriever(index, emb, cfg)


# ---- text handling -------------------------------------------------------------------------------------------

def test_stemming_and_tokenizing():
    assert stem("lesions") == "lesion"
    assert stem("biopsies") == "biopsy"
    assert stem("diagnosis") == "diagnosis"
    assert tokenize("The lesions and the melanoma") == ["lesion", "melanoma"]


def test_bm25_prefers_documents_with_the_rare_terms():
    docs = [tokenize(t) for t in ["wide excision margins melanoma", "sunscreen hat shade", "melanoma sunscreen"]]
    bm = BM25(docs)
    hits = bm.search(tokenize("excision margins"))
    assert [i for i, _ in hits] == [0]
    assert bm.matched(2, {"melanoma", "sunscreen"}) == 2
    assert bm.search(tokenize("pizza")) == []


def test_html_keeps_headings_and_lists_and_drops_page_chrome():
    html = (
        "<html><head><title>T</title><script>var x=1;</script></head><body><nav>Home | About</nav>"
        "<main><h1>Skin checks</h1><p>Look for <b>new</b> or changing spots.</p><h2>Warning signs</h2>"
        "<ul><li>A sore that does not heal</li><li>A mole that bleeds</li></ul></main>"
        "<footer>Copyright</footer></body></html>"
    )
    md = html_to_markdown(html)
    assert "# Skin checks" in md and "## Warning signs" in md
    assert "Look for new or changing spots." in md
    assert "- A sore that does not heal" in md
    assert "Home" not in md and "Copyright" not in md and "var x" not in md


def test_html_selector_picks_the_content_block():
    html = "<body><div id='menu'><p>menu text</p></div><div id='c'><p>the real content</p></div></body>"
    assert html_to_markdown(html, selector="#c").strip() == "the real content"


def test_reference_lines_are_recognised_but_normal_sentences_are_not():
    ref = "9. Balch CM, Gershenwald JE, Soong SJ, et al.: Final version of 2009 AJCC melanoma staging. J Clin Oncol 27 (36): 6199-206, 2009."
    assert looks_like_reference(ref)
    assert not looks_like_reference("A 2019 trial by Smith et al. reported longer survival with adjuvant therapy.")


def test_pdf_heading_detection_and_text_cleaning():
    assert _pdf_heading_level("3.2 Management of advanced disease") == 3
    assert _pdf_heading_level("TREATMENT OPTIONS") == 2
    assert _pdf_heading_level("This is an ordinary sentence in a paragraph.") == 0
    noisy = "Skip to content\n" * 5 + "Real sentence here.\n- Surgery\n- Surgery\n- Surgery\n- Surgery\n"
    cleaned = drop_repeated_lines(noisy)
    assert "Skip to content" not in cleaned and cleaned.count("- Surgery") == 4
    assert clean_text("a\u200b b\r\n\r\n\r\n\r\nc  d") == "a b\n\nc d\n"


# ---- chunking ------------------------------------------------------------------------------------------------

def test_chunking_uses_headings_and_skips_boilerplate_and_references():
    md = (
        "# Title\n\n## Treatment\n\n" + "Surgery is the main option for early disease. " * 5 + "\n\n"
        "## Changes to This Summary (03/14/2025)\n\nThis summary was reviewed and changed in many ways here today.\n\n"
        "## References\n\n1. Smith J, et al. Margins. J Surg 2015;12(3):45-52. doi:10.1/x PMID 1\n"
    )
    chunks = chunk_document("S01", md)
    assert [c["chunk_id"] for c in chunks] == ["S01-001"]
    assert chunks[0]["section"] == "Title > Treatment"


def test_long_sections_are_split_with_overlap():
    sentences = [f"Sentence number {i} describes one more detail of the management plan for this lesion." for i in range(60)]
    chunks = chunk_document("S02", "## Management\n\n" + " ".join(sentences), target_words=60, overlap_words=10)
    assert len(chunks) >= 8
    assert all(c["n_words"] <= 100 for c in chunks)
    tail = " ".join(chunks[0]["text"].split()[-10:])
    assert tail in chunks[1]["text"]
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_pdf_page_markers_become_page_numbers():
    md = "[[page 1]]\n## Intro\n\nFirst page text with enough words to be kept as a chunk by the splitter.\n\n[[page 2]]\n## Methods\n\nSecond page text with enough words to be kept as a chunk by the splitter."
    pages = {c["section"]: c["page"] for c in chunk_document("S09", md)}
    assert pages == {"Intro": 1, "Methods": 2}


# ---- manifest, index, retrieval --------------------------------------------------------------------------------

def test_manifest_validation(tmp_path):
    p = tmp_path / "m.yaml"
    row = "{id: S01, title: a, publisher: b, audience: both, licence: c, url: u}"
    p.write_text("sources:\n  - {id: S01, title: a, publisher: b, audience: nobody, licence: c, url: u}\n")
    with pytest.raises(ManifestError, match="audience"):
        read_manifest(p)
    p.write_text(f"sources:\n  - {row}\n  - {row}\n")
    with pytest.raises(ManifestError, match="duplicate"):
        read_manifest(p)
    p.write_text("sources:\n  - {id: X1, title: a, publisher: b, audience: both, licence: c, url: u}\n")
    with pytest.raises(ManifestError, match="id must"):
        read_manifest(p)
    p.write_text("sources:\n  - {id: S01, title: a, publisher: b, audience: both, licence: c}\n")
    with pytest.raises(ManifestError, match="url"):
        read_manifest(p)


def test_build_and_load_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    emb = HashingEmbedder()
    meta = build_index(cfg, emb, log=quiet)
    assert meta["n_sources"] == 4 and meta["n_chunks"] == 7
    assert meta["embedder"] == {"name": "hash-test", "dim": 128}
    index, warnings = load_index(cfg, emb)
    assert warnings == [] and index.emb.shape == (7, 128) and len(index.chunks) == 7
    assert not any("References" in c["section"] for c in index.chunks)


def test_hybrid_search_finds_the_answer_and_reports_ranks(tmp_path):
    _, r = built(tmp_path)
    hits = r.search("what are the excision margins for melanoma", "clinician", k=3)
    assert hits[0]["source_id"] == "S01" and "margins" in hits[0]["text"].lower()
    assert hits[0]["rank_bm25"] == 1 and hits[0]["rank_dense"] is not None
    assert hits[0]["cosine"] is not None and hits[0]["relevant"]
    assert hits[0]["publisher"] == "Test Org" and hits[0]["licence"] == "test"


def test_keyword_only_search_works_without_embeddings(tmp_path):
    _, r = built(tmp_path, dense=False)
    assert not r.dense_enabled
    hits = r.search("mohs surgery basal cell", "clinician", k=2)
    assert hits[0]["source_id"] == "S02" and hits[0]["cosine"] is None and hits[0]["rank_dense"] is None


def test_patients_never_receive_clinician_only_sources(tmp_path):
    _, r = built(tmp_path)
    for q in ["melanoma excision margins sentinel node", "mohs surgery imiquimod", "sunscreen hat"]:
        for h in r.search(q, "patient", k=8, include_irrelevant=True):
            assert h["source_id"] in {"S03", "S04"}
    clinician = {h["source_id"] for h in r.search("sunscreen hat", "clinician", k=8, include_irrelevant=True)}
    assert "S04" in clinician and "S03" not in clinician


def test_one_source_cannot_fill_the_whole_result(tmp_path):
    _, capped = built(tmp_path / "a", dense=False, max_per_source=1)
    _, open_ = built(tmp_path / "b", dense=False, max_per_source=2)
    assert len(capped.search("melanoma", "clinician", k=5)) == 1
    assert len(open_.search("melanoma", "clinician", k=5)) == 2


def test_topic_tags_nudge_the_ranking(tmp_path):
    _, r = built(tmp_path, dense=False)
    plain = r.search("excision surgery", "clinician", k=4)
    boosted = r.search("excision surgery", "clinician", k=4, topics=["bcc"])
    assert {h["source_id"] for h in plain} == {"S01", "S02"}
    assert boosted[0]["source_id"] == "S02"


@pytest.mark.parametrize("dense", [True, False])
def test_out_of_scope_questions_return_nothing(tmp_path, dense):
    _, r = built(tmp_path, dense=dense)
    for q in ["best recipe for homemade pizza dough", "how to repair a flat bicycle tire", "insulin dosing for type 2 diabetes"]:
        assert r.search(q, "clinician", k=5) == []
        assert r.search(q, "patient", k=5) == []


def test_dense_search_is_switched_off_when_the_model_is_missing_or_different(tmp_path):
    cfg = make_cfg(tmp_path)
    build_index(cfg, HashingEmbedder(), log=quiet)
    index, warnings = load_index(cfg, None)
    assert index.emb is None and "unavailable" in warnings[0]

    class Other(HashingEmbedder):
        name = "other-model"

    index, warnings = load_index(cfg, Other())
    assert index.emb is None and "other-model" in warnings[0]


def test_retrieve_labels_passages_round_robin_and_respects_the_cap(tmp_path):
    _, r = built(tmp_path)
    queries = [
        {"id": "management", "text": "melanoma excision biopsy"},
        {"id": "medication", "text": "basal cell carcinoma imiquimod fluorouracil"},
    ]
    out = r.retrieve(queries, "clinician", topics=["melanoma"], max_passages=3)
    assert out["status"] == "ok"
    assert [p["label"] for p in out["passages"]] == ["P1", "P2", "P3"]
    assert out["sections"]["management"][0] == "P1" and out["sections"]["medication"][0] == "P2"
    assert len(r.retrieve(queries, "clinician", max_passages=1)["passages"]) == 1
    empty = r.retrieve([{"id": "x", "text": "pizza dough recipe"}], "clinician")
    assert empty["status"] == "empty" and empty["passages"] == [] and empty["sections"] == {"x": []}


def test_service_loads_lazily_and_reloads_after_a_rebuild(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    monkeypatch.setenv("RAG_DIR", str(cfg.rag_dir))
    monkeypatch.setenv("RAG_EMBEDDER", "none")
    svc = RagService(embedder_factory=lambda c: (None, None))
    assert svc.status()["ready"] is False and "not built" in svc.status()["reason"]
    queries = [{"id": "a", "text": "melanoma excision margins"}]
    assert svc.retrieve("clinician", queries)["status"] == "unavailable"
    build_index(cfg, None, log=quiet)
    status = svc.status()
    assert status["ready"] and status["n_chunks"] == 7 and status["dense"] is False
    out = svc.retrieve("clinician", queries)
    assert out["status"] == "ok" and out["embedder"] is None and out["passages"][0]["label"] == "P1"


# ---- HTTP API --------------------------------------------------------------------------------------------------

def client_for(svc):
    app = FastAPI()
    app.include_router(rag_api.router)
    app.dependency_overrides[rag_api.get_service] = lambda: svc
    return TestClient(app)


BODY = {"audience": "clinician", "queries": [{"id": "a", "text": "melanoma excision margins"}]}


def test_api_requires_the_service_key(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DIR", str(tmp_path))
    client = client_for(RagService(embedder_factory=lambda c: (None, None)))
    monkeypatch.setenv("ML_SERVICE_KEY", "secret")
    assert client.post("/rag/retrieve", json=BODY).status_code == 401
    assert client.post("/rag/retrieve", json=BODY, headers={"X-Service-Key": "wrong"}).status_code == 401
    assert client.get("/rag/status").status_code == 401
    monkeypatch.delenv("ML_SERVICE_KEY")
    assert client.post("/rag/retrieve", json=BODY, headers={"X-Service-Key": "x"}).status_code == 503


def test_api_reports_unavailable_then_serves_passages(monkeypatch, tmp_path):
    cfg = make_cfg(tmp_path)
    monkeypatch.setenv("RAG_DIR", str(cfg.rag_dir))
    monkeypatch.setenv("RAG_EMBEDDER", "none")
    monkeypatch.setenv("ML_SERVICE_KEY", "secret")
    headers = {"X-Service-Key": "secret"}
    client = client_for(RagService(embedder_factory=lambda c: (None, None)))

    res = client.post("/rag/retrieve", json=BODY, headers=headers)
    assert res.status_code == 200 and res.json()["status"] == "unavailable" and res.json()["passages"] == []
    assert client.get("/rag/status", headers=headers).json()["ready"] is False

    build_index(cfg, None, log=quiet)
    assert client.post("/rag/reload", headers=headers).json()["ready"] is True
    data = client.post("/rag/retrieve", json=BODY, headers=headers).json()
    assert data["status"] == "ok" and data["passages"][0]["label"] == "P1"
    assert data["sections"] == {"a": ["P1"]}
    assert {"chunk_id", "title", "publisher", "url", "licence", "section", "text"} <= set(data["passages"][0])


def test_api_rejects_malformed_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_SERVICE_KEY", "secret")
    headers = {"X-Service-Key": "secret"}
    client = client_for(RagService(embedder_factory=lambda c: (None, None)))
    q = {"id": "a", "text": "melanoma margins"}
    assert client.post("/rag/retrieve", json={**BODY, "audience": "doctor"}, headers=headers).status_code == 422
    assert client.post("/rag/retrieve", json={**BODY, "queries": []}, headers=headers).status_code == 422
    assert client.post("/rag/retrieve", json={**BODY, "queries": [q, q]}, headers=headers).status_code == 422
    assert client.post("/rag/retrieve", json={**BODY, "queries": [{"id": "Bad Id", "text": "melanoma"}]}, headers=headers).status_code == 422
    assert client.post("/rag/retrieve", json={**BODY, "queries": [{"id": "a", "text": "x" * 500}]}, headers=headers).status_code == 422


# ---- scripts ---------------------------------------------------------------------------------------------------

PAGE = (
    "<html><body><nav>menu menu</nav><main><h1>Title</h1><p>"
    + "Skin checks matter for finding changes early. " * 45
    + "</p></main></body></html>"
)


def test_fetch_downloads_cleans_saves_and_then_uses_the_saved_copy(tmp_path):
    cfg = make_cfg(tmp_path)
    entry = read_manifest(cfg.manifest_path)[0]

    def handler(request):
        assert request.url.path == "/s01"
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=PAGE)

    res = rag_fetch.fetch_source(entry, cfg, httpx.Client(transport=httpx.MockTransport(handler)))
    assert res["status"] == "ok" and res["words"] > 300 and "warning" not in res
    text = (cfg.text_dir / "S01.md").read_text(encoding="utf-8")
    assert "# Title" in text and "menu" not in text
    assert (cfg.raw_dir / "S01.html").exists()

    def no_network(request):
        raise AssertionError("should have used the saved copy")

    again = rag_fetch.fetch_source(entry, cfg, httpx.Client(transport=httpx.MockTransport(no_network)))
    assert again["status"] == "cached"


def test_fetch_reports_errors_and_thin_pages_instead_of_crashing(tmp_path):
    cfg = make_cfg(tmp_path)
    entry = read_manifest(cfg.manifest_path)[0]
    gone = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    res = rag_fetch.fetch_source(entry, cfg, gone, force=True)
    assert res["status"] == "error" and "404" in res["error"]
    thin = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, text="<main><p>tiny</p></main>")))
    res = rag_fetch.fetch_source(entry, cfg, thin, force=True)
    assert res["status"] == "ok" and "very little text" in res["warning"]


def test_fetch_reads_files_saved_by_hand(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.manual_dir.mkdir(parents=True)
    (cfg.manual_dir / "note.md").write_text("# Guideline\n\n" + "A saved paragraph about management. " * 20, encoding="utf-8")
    entry = {"id": "S09", "title": "Saved", "file": "note.md"}
    res = rag_fetch.fetch_source(entry, cfg, httpx.Client())
    assert res["status"] == "ok" and (cfg.text_dir / "S09.md").exists()
    missing = rag_fetch.fetch_source({"id": "S10", "title": "Nope", "file": "nope.pdf"}, cfg, httpx.Client())
    assert missing["status"] == "error"


def test_eval_scores_hits_and_out_of_scope_questions(tmp_path):
    _, r = built(tmp_path)
    gold = [
        {"id": "a", "audience": "clinician", "q": "excision margins melanoma", "expect": ["margin"]},
        {"id": "b", "audience": "clinician", "q": "mohs surgery for basal cell carcinoma", "expect": ["mohs"]},
        {"id": "c", "audience": "patient", "q": "when to see a doctor about a mole", "expect": ["doctor"]},
        {"id": "d", "audience": "clinician", "oos": True, "q": "pizza dough recipe"},
        {"id": "e", "audience": "clinician", "q": "something the library never covers xylophone", "expect": ["xylophone"]},
    ]
    s = rag_eval.summarise(rag_eval.run_eval(r, gold, k=3))
    assert s["n_inscope"] == 4 and s["n_oos"] == 1
    assert s["hit_at_k"] == 0.75 and s["mrr"] > 0.5 and s["oos_clean"] == 1.0


def test_build_query_and_eval_scripts_run_end_to_end(tmp_path, monkeypatch, capsys):
    cfg = make_cfg(tmp_path)
    monkeypatch.setenv("RAG_DIR", str(cfg.rag_dir))
    assert rag_build.main(["--no-dense"]) == 0
    out = capsys.readouterr().out
    assert "chunks  : 7" in out and "keyword only" in out
    assert rag_query.main(["excision margins for melanoma", "--no-dense", "--audience", "clinician"]) == 0
    out = capsys.readouterr().out
    assert "S01-001" in out and "Melanoma management" in out
    assert rag_query.main(["pizza dough recipe", "--no-dense"]) == 0
    assert "nothing relevant found" in capsys.readouterr().out
    gold = cfg.rag_dir / "gold.yaml"
    gold.write_text("questions:\n  - {id: a, audience: clinician, q: 'excision margins melanoma', expect: [margin]}\n  - {id: o, audience: clinician, oos: true, q: 'pizza dough recipe'}\n", encoding="utf-8")
    assert rag_eval.main(["--no-dense", "--gold", str(gold)]) == 0
    out = capsys.readouterr().out
    assert "hit@5: 1.00" in out and "clean on out-of-scope: 1.00" in out


def test_build_fails_clearly_when_there_is_no_text(tmp_path, monkeypatch, capsys):
    rag = tmp_path / "rag"
    rag.mkdir()
    (rag / "sources.yaml").write_text(MANIFEST, encoding="utf-8")
    monkeypatch.setenv("RAG_DIR", str(rag))
    assert rag_build.main(["--no-dense"]) == 1
    assert "run rag_fetch first" in capsys.readouterr().out


def test_choosing_an_embedder_never_raises(tmp_path):
    cfg = RagConfig(rag_dir=tmp_path, embedder="none")
    assert get_embedder(cfg) == (None, None)
    emb, warning = get_embedder(replace(cfg, embedder="bogus"))
    assert emb is None and "unknown" in warning
    for mode in ("auto", "fastembed", "st"):
        emb, warning = get_embedder(replace(cfg, embedder=mode), offline=True)
        assert (emb is not None) or ("keyword search only" in warning)


REALISTIC_PAGE = """<html><head><title>Melanoma Treatment (PDQ)</title></head><body>
<header><a href="/">Home</a><nav><ul><li>Cancer Types</li><li>Research</li></ul></nav></header>
<div>Skip to main content</div>
<main><article>
<h1>Melanoma Treatment (PDQ)</h1>
<div><h2>On this page</h2><ul><li><a href="#a">General Information</a></li><li><a href="#b">Stage 0</a></li></ul></div>
<section id="a"><h2>General Information About Melanoma</h2>
<p>Melanoma is a malignant tumour of melanocytes. Most are found in the skin, and the diagnosis depends on examination
of the whole lesion by a pathologist after an excisional biopsy with a narrow margin of normal skin around it.</p>
<h3>Clinical Features</h3>
<p>Features that raise suspicion are asymmetry, an irregular border, colour variation, a diameter above six millimetres
and change over time, together with bleeding or ulceration of the surface of the lesion.</p>
<ul><li>Asymmetry of the shape of the lesion</li><li>Irregular or notched border</li></ul>
<h3>References</h3>
<ol><li>Smith J, Lee K, Park S, et al.: Margins in melanoma. J Surg 12 (3): 45-52, 2015. [PubMed: 1234567]</li>
<li>Jones A, Wu B, et al.: Staging revisited. Ann Oncol 2016;27(4):601-9. doi:10.1000/xyz</li></ol>
</section>
<section id="b"><h2>Treatment of Stage 0 Melanoma</h2>
<p>Surgical excision with a margin of 0.5 cm of normal skin is the standard treatment of melanoma in situ, and the
specimen should include subcutaneous fat down to the fascia so that the pathologist can assess the deep margin.</p>
<table><tr><th>Tumour thickness</th><th>Recommended margin</th></tr><tr><td>Less than 1 mm</td><td>1 cm</td></tr></table>
</section>
<section><h2>Changes to This Summary (03/14/2025)</h2><p>This summary is written and maintained by an editorial board and was updated in several places.</p></section>
<section><h2>About This PDQ Summary</h2><h3>Purpose of This Summary</h3><p>This summary gives comprehensive peer-reviewed information for health professionals about treatment.</p></section>
</article></main>
<footer><p>NCI footer links and policies</p></footer></body></html>"""


def test_a_realistic_guideline_page_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path)
    entry = read_manifest(cfg.manifest_path)[0]
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, text=REALISTIC_PAGE)))
    res = rag_fetch.fetch_source(entry, cfg, client, force=True)
    assert res["status"] == "ok"
    build_index(cfg, None, log=quiet)
    index, _ = load_index(cfg, None)
    sections = [c["section"] for c in index.chunks if c["source_id"] == "S01"]
    assert sections == [
        "Melanoma Treatment (PDQ) > General Information About Melanoma",
        "Melanoma Treatment (PDQ) > General Information About Melanoma > Clinical Features",
        "Melanoma Treatment (PDQ) > Treatment of Stage 0 Melanoma",
    ]
    text = " ".join(c["text"] for c in index.chunks if c["source_id"] == "S01")
    for noise in ("Skip to main content", "Cancer Types", "Margins in melanoma", "editorial board", "NCI footer", "Purpose of This Summary"):
        assert noise not in text
    r = Retriever(index, None, cfg)
    top = r.search("what excision margin is recommended for melanoma in situ", "clinician", k=2)
    assert top[0]["section"].endswith("Treatment of Stage 0 Melanoma") and "0.5 cm" in top[0]["text"]
    assert "Tumour thickness | Recommended margin" in top[0]["text"] and "Less than 1 mm | 1 cm" in top[0]["text"]
