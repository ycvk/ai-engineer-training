import argparse
import os
import sys
from typing import List

# 优先加载 .env（项目根目录/当前目录）
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=False)
except Exception:
    # 未安装或加载失败不致命，继续执行
    pass

from .image_ocr_reader import ImageOCRReader


def build_index_and_query(documents: List, query: str | None) -> None:
    if not documents:
        print("[INFO] 无可用 Document，跳过索引构建。")
        return
    try:
        from llama_index.core import VectorStoreIndex
    except Exception as e:
        print(f"[WARN] 无法导入 LlamaIndex（{e}），仅展示 OCR 文本，不做索引与查询。")
        for i, d in enumerate(documents):
            preview = (d.text or "").strip().replace("\n", " ")
            print(f"[DOC {i}] meta={d.metadata} text_preview={preview[:120]}")
        return

    # 如果没有 OpenAI Key，构建索引可能会失败（取决于默认嵌入/LLM配置）
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[WARN] 缺少 OPENAI_API_KEY，可能无法完成向量化/回答。尝试仅构建索引并跳过查询…")

    try:
        index = VectorStoreIndex.from_documents(documents)
        if query:
            qe = index.as_query_engine()
            resp = qe.query(query)
            print("[QUERY]", query)
            print("[RESPONSE]", resp)
        else:
            print("[INFO] 索引构建完成，未提供 query，跳过查询。")
    except Exception as e:
        print(f"[WARN] 构建索引/查询失败：{type(e).__name__}: {e}")
        for i, d in enumerate(documents):
            preview = (d.text or "").strip().replace("\n", " ")
            print(f"[DOC {i}] meta={d.metadata} text_preview={preview[:120]}")


def main():
    parser = argparse.ArgumentParser(
        description="作业二：基于 PaddleOCR 的 ImageOCRReader（CPU, PP-OCRv5）"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="图像路径或目录（可多个）。目录将自动枚举其中的图片文件。",
    )
    parser.add_argument("--lang", default="ch", help="OCR 语言，默认 ch")
    parser.add_argument(
        "--ocr_version", default="PP-OCRv5", help="OCR 版本，默认 PP-OCRv5"
    )
    parser.add_argument("--query", default=None, help="可选：构建索引后执行的查询")
    args = parser.parse_args()

    reader = ImageOCRReader(lang=args.lang, ocr_version=args.ocr_version, device="cpu")
    documents = reader.load_data(args.paths)

    # 打印 OCR 概览
    if not documents:
        print("[INFO] 未发现可用的图片文件。")
        sys.exit(0)

    for i, d in enumerate(documents):
        preview = (d.text or "").strip().replace("\n", " ")
        print(f"[DOC {i}] meta={d.metadata} text_preview={preview[:120]}")

    # 构建索引与可选查询
    build_index_and_query(documents, args.query)


if __name__ == "__main__":
    main()
