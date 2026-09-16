import create_chroma_db
from langchain_core.embeddings import DeterministicFakeEmbedding

MD_TEXT = "Mission first. AcmeTech builds reliability into everything it ships.\n"
TXT_TEXT = "The employee pension plan vests after three years of service.\n"


def _make_corpus(tmp_path):
    (tmp_path / "guide.md").write_text(MD_TEXT, encoding="utf-8")
    (tmp_path / "notes.txt").write_text(TXT_TEXT, encoding="utf-8")
    return tmp_path


def _ingest(corpus, db_name, **kwargs):
    return create_chroma_db.create_chroma_db(str(corpus), db_name=db_name, **kwargs)


def _recall(chroma, query):
    hits = chroma.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return [h.page_content for h in hits]


def test_ingest_loads_md_and_txt_and_recalls(tmp_path):
    corpus = _make_corpus(tmp_path)
    kwargs = {"embeddings": DeterministicFakeEmbedding(size=32)}

    chroma = _ingest(corpus, db_name=str(tmp_path / "chroma_db"), **kwargs)

    assert MD_TEXT.strip() in [h.strip() for h in _recall(chroma, MD_TEXT)]
    assert TXT_TEXT.strip() in [h.strip() for h in _recall(chroma, TXT_TEXT)]


def test_ingest_is_idempotent_without_delete(tmp_path):
    corpus = _make_corpus(tmp_path)
    db_name = str(tmp_path / "chroma_db")
    kwargs = {"embeddings": DeterministicFakeEmbedding(size=32)}

    _ingest(corpus, db_name=db_name, **kwargs)
    chroma = _ingest(corpus, db_name=db_name, **kwargs)

    assert MD_TEXT.strip() in [h.strip() for h in _recall(chroma, MD_TEXT)]
