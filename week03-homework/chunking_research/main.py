import argparse
import csv
import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM

try:
    from llama_index.core.llms import MessageRole
except Exception:
    # 兼容旧版本
    class MessageRole:
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"


def _import_node_parsers():
    SentenceSplitter = TokenTextSplitter = SentenceWindowNodeParser = MarkdownNodeParser = None
    try:
        from llama_index.core.node_parser import SentenceSplitter as S1, TokenTextSplitter as T1
        SentenceSplitter, TokenTextSplitter = S1, T1
    except Exception:
        try:
            from llama_index.core.node_parser.text import SentenceSplitter as S2
            from llama_index.core.node_parser.text import TokenTextSplitter as T2
            SentenceSplitter, TokenTextSplitter = S2, T2
        except Exception:
            pass

    try:
        from llama_index.core.node_parser import SentenceWindowNodeParser as W1
        SentenceWindowNodeParser = W1
    except Exception:
        try:
            from llama_index.core.node_parser.text.sentence import SentenceWindowNodeParser as W2
            SentenceWindowNodeParser = W2
        except Exception:
            pass

    try:
        from llama_index.core.node_parser import MarkdownNodeParser as M1
        MarkdownNodeParser = M1
    except Exception:
        try:
            from llama_index.core.node_parser.markdown import MarkdownNodeParser as M2
            MarkdownNodeParser = M2
        except Exception:
            pass
    return SentenceSplitter, TokenTextSplitter, SentenceWindowNodeParser, MarkdownNodeParser


def _import_postprocessors():
    MetadataReplacementPostProcessor = None
    try:
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor as P1
        MetadataReplacementPostProcessor = P1
    except Exception:
        try:
            from llama_index.core.postprocessor.metadata_replacement import (
                MetadataReplacementPostProcessor as P2,
            )
            MetadataReplacementPostProcessor = P2
        except Exception:
            pass
    return MetadataReplacementPostProcessor


def configure_openai_models() -> None:
    """配置 OpenAI 的 LLM 与 Embedding。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE") or None
    llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，请在 .env 中配置或设置环境变量。")
    Settings.llm = OpenAILLM(model=llm_model, api_key=api_key, api_base=api_base, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=api_key, api_base=api_base)


# =================== 嵌入缓存（持久化） ===================
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore

EMBED_CACHE_PERSIST_PATH: Optional[str] = None


def _safe_model_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def setup_embedding_cache(cache_dir: str, reset: bool = False, disable: bool = False) -> Optional[SimpleKVStore]:
    """为当前 embedding 模型配置磁盘缓存。返回 KVStore 或 None（若禁用）。"""
    global EMBED_CACHE_PERSIST_PATH
    if disable:
        Settings.embed_model.embeddings_cache = None
        EMBED_CACHE_PERSIST_PATH = None
        print("[CACHE] 已禁用嵌入缓存")
        return None

    os.makedirs(cache_dir, exist_ok=True)
    model_name = getattr(Settings.embed_model, "model_name", "unknown")
    fname = f"embedding_cache_{_safe_model_name(str(model_name))}.json"
    path = os.path.join(cache_dir, fname)

    kv: Optional[SimpleKVStore] = None
    if (not reset) and os.path.exists(path):
        try:
            kv = SimpleKVStore.from_persist_path(path)
            print(f"[CACHE] 加载已有缓存: {path}")
        except Exception as e:
            print(f"[CACHE] 读取缓存失败，将重建: {e}")

    if kv is None:
        kv = SimpleKVStore()
        print(f"[CACHE] 新建缓存: {path}")

    Settings.embed_model.embeddings_cache = kv
    EMBED_CACHE_PERSIST_PATH = path

    try:
        n = len(kv.get_all(collection="embeddings"))
        print(f"[CACHE] 现有嵌入条目: {n}")
    except Exception:
        pass
    return kv


def print_model_banner():
    llm_model = getattr(Settings.llm, "model", type(Settings.llm).__name__)
    embed_model = getattr(Settings.embed_model, "model_name", type(Settings.embed_model).__name__)
    print("LLM model:", llm_model)
    print("Embedding model:", embed_model)


def load_documents(data_dir: str) -> List[Document]:
    reader = SimpleDirectoryReader(input_dir=data_dir, recursive=False)
    return reader.load_data()


def load_queries(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def concat_node_text(nodes: List[Any]) -> str:  # type: ignore[name-defined]
    parts = []
    for n in nodes or []:
        node_obj = getattr(n, "node", None) or n
        text = getattr(node_obj, "text", None)
        if text is None and hasattr(node_obj, "get_content"):
            try:
                text = node_obj.get_content()
            except Exception:
                text = None
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def seq_similarity(a: str, b: str) -> float:
    import difflib

    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, int(len(text or "")))


def shard_documents(docs: List[Document], max_tokens: int = 6000) -> List[Document]:
    """将超长 Document 按 tokens 粗略切分为多段，避免嵌入输入超长。
    仅在 sentence_window 策略中作为预处理使用。
    """
    out: List[Document] = []
    for d in docs:
        text = d.text or ""
        if count_tokens(text) <= max_tokens:
            out.append(d)
            continue
        # 按近似字符窗口切分（tokens≈chars）
        approx_chars = max_tokens
        start = 0
        idx = 0
        while start < len(text):
            end = min(len(text), start + approx_chars)
            # 尝试在窗口内向右寻找换行或句号边界
            boundary = text.rfind("\n", start, end)
            if boundary == -1:
                boundary = text.rfind("。", start, end)
            if boundary == -1 or boundary <= start:
                boundary = end
            chunk = text[start:boundary]
            meta = dict(d.metadata or {})
            meta["shard_index"] = idx
            out.append(Document(text=chunk, metadata=meta, excluded_embed_metadata_keys=d.excluded_embed_metadata_keys,
                                excluded_llm_metadata_keys=d.excluded_llm_metadata_keys))
            start = boundary
            idx += 1
    return out


def redundancy_score(context: str) -> Tuple[float, int]:
    text = context or ""
    n = 20
    if len(text) < n:
        return 0.0, 1
    shingles = [text[i: i + n] for i in range(0, len(text) - n + 1)]
    total = len(shingles)
    uniq = len(set(shingles))
    ratio = 1.0 - (uniq / total)
    if ratio < 0.05:
        score = 1
    elif ratio < 0.10:
        score = 2
    elif ratio < 0.20:
        score = 3
    elif ratio < 0.35:
        score = 4
    else:
        score = 5
    return ratio, score


from typing import Any  # placed after functions to avoid circular
from dataclasses import dataclass


@dataclass
class EvalConfig:
    method: str
    params: Dict[str, Any]
    top_k: int


def build_index_with_strategy(
        documents: List[Document], cfg: EvalConfig
):
    SentenceSplitter, TokenTextSplitter, SentenceWindowNodeParser, MarkdownNodeParser = _import_node_parsers()
    MetadataReplacementPostProcessor = _import_postprocessors()

    transformations = []
    node_postprocessors = []

    if cfg.method == "sentence":
        if not SentenceSplitter:
            raise RuntimeError("当前 LlamaIndex 版本缺少 SentenceSplitter")
        splitter = SentenceSplitter(**cfg.params)
        transformations = [splitter]
    elif cfg.method == "token":
        if not TokenTextSplitter:
            raise RuntimeError("当前 LlamaIndex 版本缺少 TokenTextSplitter")
        splitter = TokenTextSplitter(**cfg.params)
        transformations = [splitter]
    elif cfg.method == "sentence_window":
        if not SentenceWindowNodeParser:
            raise RuntimeError("当前 LlamaIndex 版本缺少 SentenceWindowNodeParser")
        window_parser = SentenceWindowNodeParser.from_defaults(**cfg.params)
        # 先对超长文档做预切，避免嵌入超长
        documents = shard_documents(documents, max_tokens=6000)
        transformations = [window_parser]
        if not MetadataReplacementPostProcessor:
            raise RuntimeError("缺少 MetadataReplacementPostProcessor")
        node_postprocessors = [
            MetadataReplacementPostProcessor(
                target_metadata_key=cfg.params.get("window_metadata_key", "window")
            )
        ]
    elif cfg.method == "markdown":
        if not MarkdownNodeParser:
            raise RuntimeError("当前 LlamaIndex 版本缺少 MarkdownNodeParser")
        md_parser = MarkdownNodeParser()
        transformations = [md_parser]
    else:
        raise ValueError(f"未知的切片方法: {cfg.method}")

    index = VectorStoreIndex.from_documents(documents, transformations=transformations)
    retriever = index.as_retriever(similarity_top_k=cfg.top_k)
    query_engine = index.as_query_engine(
        similarity_top_k=cfg.top_k, node_postprocessors=node_postprocessors
    )
    return index, retriever, query_engine


def build_index_components(documents: List[Document], method: str, params: Dict[str, Any]):
    """仅构建索引与必要的后处理器，便于跨不同 top_k 复用。"""
    SentenceSplitter, TokenTextSplitter, SentenceWindowNodeParser, MarkdownNodeParser = _import_node_parsers()
    MetadataReplacementPostProcessor = _import_postprocessors()

    transformations = []
    node_postprocessors: List[Any] = []  # type: ignore[name-defined]

    if method == "sentence":
        if not SentenceSplitter:
            raise RuntimeError("当前 LlamaIndex 版本缺少 SentenceSplitter")
        splitter = SentenceSplitter(**params)
        transformations = [splitter]
    elif method == "token":
        if not TokenTextSplitter:
            raise RuntimeError("当前 LlamaIndex 版本缺少 TokenTextSplitter")
        splitter = TokenTextSplitter(**params)
        transformations = [splitter]
    elif method == "sentence_window":
        if not SentenceWindowNodeParser:
            raise RuntimeError("当前 LlamaIndex 版本缺少 SentenceWindowNodeParser")
        window_parser = SentenceWindowNodeParser.from_defaults(**params)
        documents = shard_documents(documents, max_tokens=6000)
        transformations = [window_parser]
        if not MetadataReplacementPostProcessor:
            raise RuntimeError("缺少 MetadataReplacementPostProcessor")
        node_postprocessors = [
            MetadataReplacementPostProcessor(
                target_metadata_key=params.get("window_metadata_key", "window")
            )
        ]
    elif method == "markdown":
        if not MarkdownNodeParser:
            raise RuntimeError("当前 LlamaIndex 版本缺少 MarkdownNodeParser")
        md_parser = MarkdownNodeParser()
        transformations = [md_parser]
    else:
        raise ValueError(f"未知的切片方法: {method}")

    index = VectorStoreIndex.from_documents(documents, transformations=transformations)
    return index, node_postprocessors


def _chat_json(prompt_system: str, prompt_user: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
    """用设置的 LLM 执行一次严格 JSON 响应的对话，并解析为 dict。"""
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt_system),
        ChatMessage(role=MessageRole.USER, content=prompt_user),
    ]
    text = None
    for _ in range(max_retries + 1):
        try:
            resp = Settings.llm.chat(msgs)
            text = getattr(resp, "message", None)
            if text and hasattr(text, "content"):
                text = text.content
            if not isinstance(text, str):
                text = str(resp)
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def judge_answer(question: str, context: str, answer: str, ground_truth: str) -> Dict[str, Any]:
    """使用 LLM 做自动化评审，输出 0-1 区间评分与布尔裁决。"""
    system = (
        "你是一个严谨的评审助手。只输出 JSON，不要多余文字。\n"
        "给定提问、检索上下文、模型回答、标准答案，你需要评估回答是否正确、是否忠实于上下文、是否覆盖关键要点。"
    )
    user = (
        "请按如下 JSON 模式作答：{\n"
        "  \"correctness\": 0..1,\n"
        "  \"faithfulness\": 0..1,\n"
        "  \"coverage\": 0..1,\n"
        "  \"verdict\": true|false\n"
        "}\n\n"
        f"提问: {question}\n\n"
        f"标准答案: {ground_truth}\n\n"
        f"检索上下文: {context[:4000]}\n\n"
        f"模型回答: {answer}"
    )
    obj = _chat_json(system, user)
    out = {"judge_correctness": None, "judge_faithfulness": None, "judge_coverage": None, "judge_verdict": None}
    if isinstance(obj, dict):
        def val01(x):
            try:
                v = float(x)
                return max(0.0, min(1.0, v))
            except Exception:
                return None

        out["judge_correctness"] = val01(obj.get("correctness"))
        out["judge_faithfulness"] = val01(obj.get("faithfulness"))
        out["judge_coverage"] = val01(obj.get("coverage"))
        v = obj.get("verdict")
        out["judge_verdict"] = bool(v) if isinstance(v, bool) else (str(v).lower() in {"true", "1", "yes"})
    return out


EVAL_WITH_JUDGE = True
FUZZY_THRESHOLD = 0.6


def evaluate_once(retriever, query_engine, question: str, ground_truth: str) -> Dict[str, Any]:
    try:
        nodes = retriever.retrieve(question)
    except Exception as e:
        nodes = []
        retrieve_error = str(e)
    else:
        retrieve_error = None
    context = concat_node_text(nodes)

    t0 = time.time()
    try:
        response = query_engine.query(question)
        elapsed = time.time() - t0
    except Exception as e:
        response = None
        elapsed = time.time() - t0
        gen_error = str(e)
    else:
        gen_error = None

    def resp_text(r: Any) -> str:
        if r is None:
            return ""
        for attr in ("response", "text"):
            v = getattr(r, attr, None)
            if isinstance(v, str) and v.strip():
                return v
        try:
            return str(r)
        except Exception:
            return ""

    answer = resp_text(response)

    contains_strict_retrieval = (ground_truth or "") in (context or "")
    contains_strict_answer = (ground_truth or "") in (answer or "")
    fuzzy_retrieval = 0.0
    try:
        fuzzy_retrieval = max((seq_similarity(ground_truth, getattr(getattr(n, "node", n), "text", "")) for n in nodes),
                              default=0.0)
    except Exception:
        pass
    fuzzy_answer = seq_similarity(ground_truth, answer)
    answer_hit_fuzzy_bin = bool(fuzzy_answer >= FUZZY_THRESHOLD)
    retrieval_hit_fuzzy_bin = bool(fuzzy_retrieval >= FUZZY_THRESHOLD)
    context_tokens = count_tokens(context)
    redundancy_ratio, redundancy_grade = redundancy_score(context)
    # LLM 评审
    judge = judge_answer(question, context, answer, ground_truth) if EVAL_WITH_JUDGE else {
        "judge_correctness": None,
        "judge_faithfulness": None,
        "judge_coverage": None,
        "judge_verdict": None,
    }

    return {
        "retrieve_error": retrieve_error,
        "gen_error": gen_error,
        "retrieval_hit_strict": contains_strict_retrieval,
        "retrieval_hit_fuzzy": float(fuzzy_retrieval),
        "answer_hit_strict": contains_strict_answer,
        "answer_hit_fuzzy": float(fuzzy_answer),
        "answer_hit_fuzzy_bin": answer_hit_fuzzy_bin,
        "retrieval_hit_fuzzy_bin": retrieval_hit_fuzzy_bin,
        "context_tokens": int(context_tokens),
        "redundancy_ratio": float(redundancy_ratio),
        "redundancy_grade": int(redundancy_grade),
        **judge,
        "answer": answer,
        "context_preview": context[:300],
        "latency_sec": round(elapsed, 3),
    }


def eval_grid(documents: List[Document], queries: List[Dict[str, str]], configs: List[EvalConfig], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    summary_path = os.path.join(out_dir, "summary.csv")
    open(results_path, "w").close()

    # 复用索引：按 (method, params) 分组缓存
    index_cache: Dict[str, Tuple[Any, List[Any]]] = {}
    for cfg in configs:
        print(f"[RUN] method={cfg.method} top_k={cfg.top_k} params={cfg.params}")
        key = json.dumps({"method": cfg.method, "params": cfg.params}, sort_keys=True, ensure_ascii=False)
        if key not in index_cache:
            try:
                idx, post = build_index_components(documents, cfg.method, cfg.params)
                index_cache[key] = (idx, post)
            except Exception as e:
                print(f"  构建索引失败: {e}")
                continue

        idx, post = index_cache[key]
        retriever = idx.as_retriever(similarity_top_k=cfg.top_k)
        query_engine = idx.as_query_engine(similarity_top_k=cfg.top_k, node_postprocessors=post)
        for i, item in enumerate(queries, 1):
            q = item.get("question", "")
            gt = item.get("ground_truth", "")
            res = evaluate_once(retriever, query_engine, q, gt)
            record = {"method": cfg.method, "top_k": cfg.top_k, "params": cfg.params, "qid": i, "question": q,
                      "ground_truth": gt, **res}
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("  完成。")

    # 汇总
    agg: Dict[str, Dict[str, Any]] = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = json.dumps({"method": r["method"], "top_k": r["top_k"], "params": r["params"]}, ensure_ascii=False)
            a = agg.setdefault(key, {
                "method": r["method"],
                "top_k": r["top_k"],
                "params": r["params"],
                "n": 0,
                "retrieval_hit_strict_sum": 0,
                "answer_hit_strict_sum": 0,
                "retrieval_hit_fuzzy_avg": 0.0,
                "answer_hit_fuzzy_avg": 0.0,
                "retrieval_hit_fuzzy_sum_bin": 0,
                "answer_hit_fuzzy_sum_bin": 0,
                "context_tokens_avg": 0.0,
                "redundancy_ratio_avg": 0.0,
                "redundancy_grade_avg": 0.0,
            })
            a["n"] += 1
            a["retrieval_hit_strict_sum"] += 1 if r["retrieval_hit_strict"] else 0
            a["answer_hit_strict_sum"] += 1 if r["answer_hit_strict"] else 0
            if r.get("retrieval_hit_fuzzy_bin"):
                a["retrieval_hit_fuzzy_sum_bin"] += 1
            if r.get("answer_hit_fuzzy_bin"):
                a["answer_hit_fuzzy_sum_bin"] += 1
            for k, avg_k in [
                ("retrieval_hit_fuzzy", "retrieval_hit_fuzzy_avg"),
                ("answer_hit_fuzzy", "answer_hit_fuzzy_avg"),
                ("context_tokens", "context_tokens_avg"),
                ("redundancy_ratio", "redundancy_ratio_avg"),
                ("redundancy_grade", "redundancy_grade_avg"),
            ]:
                a[avg_k] = ((a["n"] - 1) * a[avg_k] + float(r[k])) / a["n"]

            # 汇总 judge 指标
            if r.get("judge_correctness") is not None:
                a["judge_correctness_sum"] = a.get("judge_correctness_sum", 0.0) + float(r["judge_correctness"]) if r[
                                                                                                                        "judge_correctness"] is not None else a.get(
                    "judge_correctness_sum", 0.0)
            if r.get("judge_faithfulness") is not None:
                a["judge_faithfulness_sum"] = a.get("judge_faithfulness_sum", 0.0) + float(r["judge_faithfulness"]) if \
                r["judge_faithfulness"] is not None else a.get("judge_faithfulness_sum", 0.0)
            if r.get("judge_coverage") is not None:
                a["judge_coverage_sum"] = a.get("judge_coverage_sum", 0.0) + float(r["judge_coverage"]) if r[
                                                                                                               "judge_coverage"] is not None else a.get(
                    "judge_coverage_sum", 0.0)
            if r.get("judge_verdict") is not None:
                a["judge_verdict_true"] = a.get("judge_verdict_true", 0) + (1 if bool(r["judge_verdict"]) else 0)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "top_k",
                "params",
                "n",
                "retrieval_hit_rate",
                "retrieval_hit_fuzzy_rate",
                "answer_hit_rate",
                "answer_hit_fuzzy_rate",
                "retrieval_hit_fuzzy_avg",
                "answer_hit_fuzzy_avg",
                "context_tokens_avg",
                "redundancy_ratio_avg",
                "redundancy_grade_avg",
                "judge_correctness_avg",
                "judge_faithfulness_avg",
                "judge_coverage_avg",
                "judge_verdict_rate",
            ],
        )
        writer.writeheader()
        for a in agg.values():
            # judge 聚合（简单均值/比例），若无字段则保持 0
            jc = a.get("judge_correctness_sum", 0.0) / max(1, a["n"]) if a.get(
                "judge_correctness_sum") is not None else 0.0
            jf = a.get("judge_faithfulness_sum", 0.0) / max(1, a["n"]) if a.get(
                "judge_faithfulness_sum") is not None else 0.0
            jcov = a.get("judge_coverage_sum", 0.0) / max(1, a["n"]) if a.get("judge_coverage_sum") is not None else 0.0
            jv = a.get("judge_verdict_true", 0) / max(1, a["n"]) if a.get("judge_verdict_true") is not None else 0.0

            writer.writerow({
                "method": a["method"],
                "top_k": a["top_k"],
                "params": json.dumps(a["params"], ensure_ascii=False),
                "n": a["n"],
                "retrieval_hit_rate": round(a["retrieval_hit_strict_sum"] / max(1, a["n"]), 3),
                "retrieval_hit_fuzzy_rate": round(a["retrieval_hit_fuzzy_sum_bin"] / max(1, a["n"]), 3),
                "answer_hit_rate": round(a["answer_hit_strict_sum"] / max(1, a["n"]), 3),
                "answer_hit_fuzzy_rate": round(a["answer_hit_fuzzy_sum_bin"] / max(1, a["n"]), 3),
                "retrieval_hit_fuzzy_avg": round(a["retrieval_hit_fuzzy_avg"], 3),
                "answer_hit_fuzzy_avg": round(a["answer_hit_fuzzy_avg"], 3),
                "context_tokens_avg": int(a["context_tokens_avg"]),
                "redundancy_ratio_avg": round(a["redundancy_ratio_avg"], 3),
                "redundancy_grade_avg": round(a["redundancy_grade_avg"], 2),
                "judge_correctness_avg": round(jc, 3),
                "judge_faithfulness_avg": round(jf, 3),
                "judge_coverage_avg": round(jcov, 3),
                "judge_verdict_rate": round(jv, 3),
            })

    return results_path, summary_path


def _gen_charts(summary_csv: str, out_dir: str) -> List[str]:
    import csv as _csv
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] 未安装或无法导入 matplotlib，跳过绘图: {e}")
        return []
    rows = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        for x in r:
            def fget(name):
                try:
                    return float(x.get(name) or 0.0)
                except Exception:
                    return 0.0

            x["answer_hit_rate"] = fget("answer_hit_rate")
            x["answer_hit_fuzzy_rate"] = fget("answer_hit_fuzzy_rate")
            x["retrieval_hit_rate"] = fget("retrieval_hit_rate")
            x["retrieval_hit_fuzzy_rate"] = fget("retrieval_hit_fuzzy_rate")
            x["judge_verdict_rate"] = fget("judge_verdict_rate")
            x["context_tokens_avg"] = fget("context_tokens_avg")
            rows.append(x)
    # 排序优先级：judge_verdict_rate > answer_hit_fuzzy_rate > answer_hit_rate > retrieval_hit_fuzzy_rate
    rows.sort(key=lambda x: (-x["judge_verdict_rate"], -x["answer_hit_fuzzy_rate"], -x["answer_hit_rate"],
                             -x["retrieval_hit_fuzzy_rate"], x["context_tokens_avg"]))
    top = rows[:10]

    labels = [f"{i['method']}/k={i['top_k']}" for i in top]
    # 优先使用 judge_verdict_rate 或 fuzzy_rate 作为可视化“准确度”
    acc = [i.get("judge_verdict_rate", 0.0) or i.get("answer_hit_fuzzy_rate", 0.0) or i.get("answer_hit_rate", 0.0) for
           i in top]
    rcl = [i.get("retrieval_hit_fuzzy_rate", 0.0) or i.get("retrieval_hit_rate", 0.0) for i in top]
    cost = [i["context_tokens_avg"] for i in top]

    paths: List[str] = []
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(acc)), acc, color="#4C78A8")
    plt.xticks(range(len(acc)), labels, rotation=30, ha='right')
    plt.ylabel('score (judge/answer_fuzzy)')
    plt.tight_layout()
    p1 = os.path.join(out_dir, "chart_answer_hit_rate.png")
    plt.savefig(p1, dpi=150)
    paths.append(p1)
    plt.close()

    plt.figure(figsize=(10, 4))
    w = 0.4
    x = list(range(len(acc)))
    plt.bar([xx - w / 2 for xx in x], rcl, width=w, label='retrieval', color="#72B7B2")
    plt.bar([xx + w / 2 for xx in x], acc, width=w, label='answer', color="#4C78A8")
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('hit_rate (retrieval_fuzzy vs answer)')
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(out_dir, "chart_hit_rates.png")
    plt.savefig(p2, dpi=150)
    paths.append(p2)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(cost)), cost, color="#F58518")
    plt.xticks(range(len(cost)), labels, rotation=30, ha='right')
    plt.ylabel('context_tokens_avg')
    plt.tight_layout()
    p3 = os.path.join(out_dir, "chart_cost.png")
    plt.savefig(p3, dpi=150)
    paths.append(p3)
    plt.close()
    return paths


def write_report(summary_csv: str, report_path: str):
    rows = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["answer_hit_rate"] = float(r["answer_hit_rate"]) if r["answer_hit_rate"] else 0.0
            r["retrieval_hit_rate"] = float(r["retrieval_hit_rate"]) if r["retrieval_hit_rate"] else 0.0
            r["context_tokens_avg"] = float(r["context_tokens_avg"]) if r["context_tokens_avg"] else 0.0
            rows.append(r)
    rows.sort(key=lambda x: (-x["answer_hit_rate"], -x["retrieval_hit_rate"], x["context_tokens_avg"]))
    top = rows[:5]

    lines = []
    lines.append("# 实验结果分析写在这里!")
    lines.append("")
    lines.append("## 自动生成的基线总结")
    lines.append("")
    lines.append("下表为当前参数网格的前五名配置（优先按回答命中率排序，其次检索命中率，最后平均上下文 tokens 成本）：")
    lines.append("")
    lines.append(
        "| method | top_k | params | judge_verdict_rate | answer_hit_fuzzy_rate | answer_hit_rate | retrieval_hit_fuzzy_rate | ctx_tokens_avg | redundancy(avg) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in top:
        lines.append(
            f"| {r['method']} | {r['top_k']} | {r['params']} | {float(r.get('judge_verdict_rate', 0.0)):.3f} | {float(r.get('answer_hit_fuzzy_rate', 0.0)):.3f} | {float(r.get('answer_hit_rate', 0.0)):.3f} | {float(r.get('retrieval_hit_fuzzy_rate', 0.0)):.3f} | {int(float(r.get('context_tokens_avg', 0.0)))} | {float(r.get('redundancy_ratio_avg', 0.0)):.3f} |"
        )
    lines.append("")
    lines.append("### 观察与建议（可在此基础上补充）")
    lines.append(
        "- chunk_size 与 chunk_overlap 的权衡：更大的 chunk_size 往往提升检索命中，但上下文冗余与 tokens 成本上升；overlap 过小会丢上下文，过大则重复严重。")
    lines.append(
        "- 句子窗口：在相近 top_k 下，通常更有助于生成质量，因为返回了紧邻上下文；需确保启用 MetadataReplacementPostProcessor。")
    lines.append("- 成本控制：监控 context_tokens_avg，避免过高导致延迟和费用增加。")

    # 生成图表并引用
    out_dir = os.path.join(os.path.dirname(summary_csv))
    charts = _gen_charts(summary_csv, out_dir)
    if charts:
        lines.append("")
        lines.append("## 指标可视化")
        for p in charts:
            rel = os.path.relpath(p, os.path.dirname(report_path))
            lines.append(f"![]({rel})")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _parse_list_str(val: Optional[str], cast=int) -> Optional[List[Any]]:
    if val is None:
        return None
    if isinstance(val, list):
        return val
    parts = [p.strip() for p in str(val).split(",") if p.strip()]
    try:
        return [cast(p) for p in parts]
    except Exception:
        return [p for p in parts]


def build_default_configs(
        include_markdown: bool = False,
        sample: bool = False,
        methods: Optional[List[str]] = None,
        topk_list: Optional[List[int]] = None,
        sentence_chunk_sizes: Optional[List[int]] = None,
        sentence_overlaps: Optional[List[int]] = None,
        token_chunk_sizes: Optional[List[int]] = None,
        token_overlaps: Optional[List[int]] = None,
        window_sizes: Optional[List[int]] = None,
) -> List[EvalConfig]:
    methods = [m.strip() for m in (methods or ["sentence", "token", "sentence_window"])]
    topks = topk_list or ([3, 5, 8] if not sample else [3])

    # 采样模式用较小组合
    if sample:
        sentence_chunk_sizes = sentence_chunk_sizes or [512]
        sentence_overlaps = sentence_overlaps or [50]
        token_chunk_sizes = token_chunk_sizes or [256]
        token_overlaps = token_overlaps or [32]
        window_sizes = window_sizes or [3]
    else:
        sentence_chunk_sizes = sentence_chunk_sizes or [256, 512, 1024]
        sentence_overlaps = sentence_overlaps or [0, 50, 100]
        token_chunk_sizes = token_chunk_sizes or [128, 256]
        token_overlaps = token_overlaps or [16, 32]
        window_sizes = window_sizes or [1, 3, 5]

    cfgs: List[EvalConfig] = []

    if "sentence" in methods:
        for cs in sentence_chunk_sizes:
            for ov in sentence_overlaps:
                for k in topks:
                    cfgs.append(
                        EvalConfig(
                            method="sentence",
                            params={"chunk_size": cs, "chunk_overlap": ov, "separator": "。！？；\n"},
                            top_k=k,
                        )
                    )

    if "token" in methods:
        for cs in token_chunk_sizes:
            for ov in token_overlaps:
                for k in topks:
                    cfgs.append(
                        EvalConfig(
                            method="token",
                            params={"chunk_size": cs, "chunk_overlap": ov, "separator": "\n"},
                            top_k=k,
                        )
                    )

    if "sentence_window" in methods:
        for ws in window_sizes:
            for k in topks:
                cfgs.append(
                    EvalConfig(
                        method="sentence_window",
                        params={"window_size": ws, "window_metadata_key": "window",
                                "original_text_metadata_key": "original_text"},
                        top_k=k,
                    )
                )

    if include_markdown and "markdown" in methods:
        for k in topks:
            cfgs.append(EvalConfig(method="markdown", params={}, top_k=k))

    return cfgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunking research eval runner")
    p.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "docs"))
    p.add_argument("--queries", default=os.path.join(os.path.dirname(__file__), "queries.json"))
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    p.add_argument("--cache_dir", default=os.path.join(os.path.dirname(__file__), "cache"), help="嵌入缓存目录")
    p.add_argument("--reset_cache", action="store_true", help="忽略已有缓存并重建")
    p.add_argument("--no_cache", action="store_true", help="禁用嵌入缓存")
    p.add_argument("--methods", default="sentence,token,sentence_window", help="要运行的方法，逗号分隔")
    p.add_argument("--topk", default="3,5,8", help="top_k 列表，逗号分隔")
    p.add_argument("--sentence_chunk_sizes", default=None, help="句子切片 chunk_size 列表，逗号分隔")
    p.add_argument("--sentence_overlaps", default=None, help="句子切片 chunk_overlap 列表，逗号分隔")
    p.add_argument("--token_chunk_sizes", default=None, help="Token 切片 chunk_size 列表，逗号分隔")
    p.add_argument("--token_overlaps", default=None, help="Token 切片 chunk_overlap 列表，逗号分隔")
    p.add_argument("--window_sizes", default=None, help="句子窗口 window_size 列表，逗号分隔")
    p.add_argument("--include_markdown", action="store_true")
    p.add_argument("--sample", action="store_true", help="仅跑少量参数组合以快速验证")
    p.add_argument("--no_judge", action="store_true", help="跳过 LLM 评审以加速运行")
    p.add_argument("--fuzzy_threshold", type=float, default=0.6, help="模糊匹配阈值，默认 0.6")
    # 两阶段评测：先全量无评审选 TopN，再对 TopN 开启评审
    p.add_argument("--two_stage", action="store_true", help="两阶段评测：先无评审筛选 TopN，再对 TopN 自动评审")
    p.add_argument("--top_n", type=int, default=5, help="两阶段评测中进入第二阶段的 TopN 数量")
    p.add_argument(
        "--rank_by",
        default="answer_hit_fuzzy_rate",
        help="两阶段评测第一阶段排序指标（answer_hit_fuzzy_rate/answer_hit_rate/retrieval_hit_fuzzy_rate/retrieval_hit_rate）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    global EVAL_WITH_JUDGE
    EVAL_WITH_JUDGE = not args.no_judge
    global FUZZY_THRESHOLD
    FUZZY_THRESHOLD = float(args.fuzzy_threshold or 0.6)
    configure_openai_models()
    print_model_banner()
    # 配置嵌入缓存
    kv = setup_embedding_cache(cache_dir=args.cache_dir, reset=args.reset_cache, disable=args.no_cache)

    documents = load_documents(args.data_dir)
    queries = load_queries(args.queries)
    methods = [m.strip() for m in (args.methods or "").split(",") if m.strip()]
    configs = build_default_configs(
        include_markdown=args.include_markdown,
        sample=args.sample,
        methods=methods,
        topk_list=_parse_list_str(args.topk, int),
        sentence_chunk_sizes=_parse_list_str(args.sentence_chunk_sizes, int),
        sentence_overlaps=_parse_list_str(args.sentence_overlaps, int),
        token_chunk_sizes=_parse_list_str(args.token_chunk_sizes, int),
        token_overlaps=_parse_list_str(args.token_overlaps, int),
        window_sizes=_parse_list_str(args.window_sizes, int),
    )

    if args.two_stage:
        # 第一阶段：无评审跑全量，输出到 stage1 目录
        stage1_dir = os.path.join(args.out_dir, "stage1")
        print("[Two-Stage] 阶段一：无评审基线跑全量，用于筛选 TopN")
        saved_eval_with_judge = EVAL_WITH_JUDGE
        EVAL_WITH_JUDGE = False
        _, stage1_summary = eval_grid(documents, queries, configs, stage1_dir)

        def _safe_float(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        # 读取阶段一 summary，选 TopN
        import csv as _csv
        rows = []
        with open(stage1_summary, "r", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for x in r:
                x["answer_hit_rate"] = _safe_float(x.get("answer_hit_rate"))
                x["answer_hit_fuzzy_rate"] = _safe_float(x.get("answer_hit_fuzzy_rate"))
                x["retrieval_hit_rate"] = _safe_float(x.get("retrieval_hit_rate"))
                x["retrieval_hit_fuzzy_rate"] = _safe_float(x.get("retrieval_hit_fuzzy_rate"))
                x["context_tokens_avg"] = _safe_float(x.get("context_tokens_avg"))
                rows.append(x)

        rank_by = (args.rank_by or "answer_hit_fuzzy_rate").strip()
        if rank_by not in {"answer_hit_fuzzy_rate", "answer_hit_rate", "retrieval_hit_fuzzy_rate",
                           "retrieval_hit_rate"}:
            rank_by = "answer_hit_fuzzy_rate"

        rows.sort(key=lambda x: (-x.get(rank_by, 0.0), x.get("context_tokens_avg", 0.0)))
        top_rows = rows[: max(1, int(args.top_n or 5))]

        # 将 TopN 行映射回 EvalConfig 列表
        top_cfgs: List[EvalConfig] = []
        for r in top_rows:
            try:
                params = json.loads(r.get("params") or "{}")
            except Exception:
                params = {}
            try:
                top_cfgs.append(
                    EvalConfig(method=r.get("method") or "sentence", params=params, top_k=int(r.get("top_k") or 3)))
            except Exception:
                continue

        print(f"[Two-Stage] 阶段一完成，选出 Top{len(top_cfgs)} 进入自动评审：")
        for c in top_cfgs:
            print(f"  - {c.method} k={c.top_k} params={c.params}")

        # 第二阶段：对 TopN 开启评审，输出至最终 out_dir
        EVAL_WITH_JUDGE = True  # 强制开启评审
        results_path, summary_path = eval_grid(documents, queries, top_cfgs, args.out_dir)
        print(f"结果明细: {results_path}")
        print(f"结果汇总: {summary_path}")
        report_path = os.path.join(os.path.dirname(__file__), "report.md")
        write_report(summary_path, report_path)
        print(f"报告已更新: {report_path}")
        # 还原原始评审开关状态
        EVAL_WITH_JUDGE = saved_eval_with_judge
    else:
        results_path, summary_path = eval_grid(documents, queries, configs, args.out_dir)
        print(f"结果明细: {results_path}")
        print(f"结果汇总: {summary_path}")
        report_path = os.path.join(os.path.dirname(__file__), "report.md")
        write_report(summary_path, report_path)
        print(f"报告已更新: {report_path}")

    # 持久化嵌入缓存
    from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore as _SKV
    global EMBED_CACHE_PERSIST_PATH
    if isinstance(kv, _SKV) and EMBED_CACHE_PERSIST_PATH:
        try:
            kv.persist(EMBED_CACHE_PERSIST_PATH)
            print(f"[CACHE] 已持久化到: {EMBED_CACHE_PERSIST_PATH}")
        except Exception as e:
            print(f"[CACHE] 持久化失败: {e}")


if __name__ == "__main__":
    main()
