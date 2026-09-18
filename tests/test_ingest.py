from pathlib import Path

import create_chroma_db
from langchain_core.embeddings import DeterministicFakeEmbedding

MD_TEXT = "Mission first. AcmeTech builds reliability into everything it ships.\n"
TXT_TEXT = "The employee pension plan vests after three years of service.\n"


def _make_corpus(tmp_path):
    (tmp_path / "guide.md").write_text(MD_TEXT, encoding="utf-8")
    (tmp_path / "notes.txt").write_text(TXT_TEXT, encoding="utf-8")
    return tmp_path


def _ingest(corpus, db_name, **kwargs):
    """建库；**FTS 侧车库必须落到临时目录**。

    不显式传 `fts_db` 时，`create_chroma_db` 会退回 `settings.CHROMA_FTS_DB`
    —— 那是仓库根的真实索引（相对路径 `./var/chroma_fts.sqlite`），而
    `_build_fts_sidecar` 会先 `DROP TABLE` 再重建。后果是跑一次测试就把生产
    索引覆盖成测试语料（docs/19 R1）。因此这里默认按语料目录推导。
    """
    kwargs.setdefault("fts_db", str(Path(corpus) / "chroma_fts.sqlite"))
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


def test_ingest_does_not_write_production_fts(tmp_path, monkeypatch):
    """守卫 R1：建库必须写到调用方指定的 `fts_db`，不得碰生产索引。

    这条用例守的是一个真实发生过的静默污染：`create_chroma_db` 曾无条件写
    `settings.CHROMA_FTS_DB`（相对 CWD 的 `./var/chroma_fts.sqlite`），
    于是跑一次测试就把生产 RAG 的词法索引 `DROP TABLE` 后填成测试语料，
    而 Chroma 那侧因为 `db_name` 被隔离了、看不出异常。

    反例：把 `fts_db` 参数从 `create_chroma_db` 去掉（退回硬取 settings），
    本用例必须变红 —— 那时 `sentinel` 会被创建。
    """
    sentinel = tmp_path / "PROD_index_must_not_be_touched.sqlite"
    monkeypatch.setattr(create_chroma_db.settings, "CHROMA_FTS_DB", str(sentinel))

    corpus = _make_corpus(tmp_path)
    _ingest(
        corpus, db_name=str(tmp_path / "chroma_db"), embeddings=DeterministicFakeEmbedding(size=32)
    )

    assert not sentinel.exists(), "建库把 FTS 侧车间写进了生产路径（docs/19 R1 回归）"
    assert (tmp_path / "chroma_fts.sqlite").exists(), "FTS 侧车应落在调用方给的路径上"
