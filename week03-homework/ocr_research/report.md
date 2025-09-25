# 作业二 · OCR 图像文本加载器报告

## 声明
- 本实验在 macOS（CPU，Python 3.11）环境完成；默认优先调用 PP‑OCRv5，若遇到平台兼容性问题自动降级至 RapidOCR（ONNXRuntime）。
- 统一通过 LlamaIndex 将 OCR 文本封装为 `Document`，并使用向量索引完成简单问答验证。

## 一、实验配置
- 数据来源：`ocr_research/img`（三类图片：海报/标语/段落）
- OCR 方案：优先 `PaddleOCR (PP‑OCRv5, lang=ch, device=cpu)` → 失败时降级 `RapidOCR`
- LLM/Embedding：读取 `.env` 中的 OpenAI 配置（`OPENAI_API_KEY` 等），由 `llama-index-llms-openai` 与 `llama-index-embeddings-openai` 提供
- 评测指标：
  - 识别：`num_text_blocks`、`avg_confidence`
  - 检索/问答：是否能召回正确文本片段并生成一致回答（定性）

## 二、关键结论
- 可用性：三张图片均成功识别并生成 `Document`；向量索引与问答链路可通。
- 质量概况：
  - 大字标语（1.jpg）识别接近满分，块数少、置信度高。
  - 短标语（3.jpg）识别稳定。
  - 段落文本（2.jpg）按行分块，平均置信度约 0.98，拼接自然。
- 工程取舍：为避免本机 Paddle 3.x 兼容性陷阱，默认自动降级 RapidOCR；接口与元数据保持一致，便于后续切回 PP‑OCRv5。

## 三、结果总览（摘要）
| file | ocr_model | num_text_blocks | avg_confidence | preview |
|---|---|---:|---:|---|
| 2.jpg | RapidOCR | 5 | 0.9789 | 有时我们分开…你原本就不想只做朋友。 |
| 3.jpg | RapidOCR | 1 | 0.9704 | 文明驾驶 消除陋习 |
| 1.jpg | RapidOCR | 2 | 0.9981 | 宁可累死自己 也要卷死同学 |

> 说明：预览为文本前 20–30 个字；完整文本见运行日志或 `Document.text`。

## 四、代表性样例（问答）
- Query：图片中提到了什么？
- Answer（节选）：有时我们分开，喜欢一个人，就告诉他…你原本就不想只做朋友。
- 观察：答案与 2.jpg 段落语义一致，说明检索召回与生成链路可用。

## 五、方法说明（Reader 与封装）
- 架构示意：
```
        +------------------+
        |   Images (JPG)   |
        +---------+--------+
                  |
                  v
        +------------------+      init fail       +------------------+
        |   PaddleOCR v5   | -------------------> |     RapidOCR     |
        | (lang=ch, cpu)   |  fallback (optional) |   (ONNXRuntime)  |
        +---------+--------+                      +---------+--------+
                  |  blocks: [(text, conf, box)]            |
                  v                                         v
        +------------------+                      +------------------+
        |  Normalize/Sort  |                      |  Normalize/Sort  |
        | (top_y,left_x)   |                      | (top_y,left_x)   |
        +---------+--------+                      +---------+--------+
                  |  join with newlines                     |
                  +-------------------+---------------------+
                                      v
                           +-----------------------+
                           |  LlamaIndex Document  |
                           |  text + metadata      |
                           |  - image_path         |
                           |  - ocr_model          |
                           |  - language           |
                           |  - num_text_blocks    |
                           |  - avg_confidence     |
                           +-----------+-----------+
                                       |
                                       v
                          +--------------------------+
                          |  VectorStoreIndex (RAG)  |
                          +------------+-------------+
                                       |
                                       v
                               +---------------+
                               |  QueryEngine  |
                               +-------+-------+
                                       |
                                       v
                                    Answer
```
- Reader：`ImageOCRReader(BaseReader)`
  - 初始化：`lang='ch'`、`ocr_version='PP-OCRv5'`、`device='cpu'`，预留 `**kwargs` 透传 OCR 参数。
  - OCR 选择：先尝试 `PaddleOCR`，失败则降级 `RapidOCR`；对外统一返回 `Document`。
  - 解析与拼接：抽取 `(text, conf, top_y, left_x)`，按 `(y, x)` 排序后换行拼接，避免句子黏连。
  - 元数据：`image_path`、`ocr_model`、`language`、`num_text_blocks`、`avg_confidence`。
- 入口：自动 `load_dotenv()` 读取 `.env`，构建 `VectorStoreIndex`，可选 `--query` 直接问答。

## 六、参数与实现细节
- OCR 参数（默认）：`lang=ch, ocr_version=PP-OCRv5, device=cpu`
- 文本顺序：按 `top_y`（从上到下）→ `left_x`（从左到右）排序后拼接，保留换行。
- 稳健性：
  - 支持文件/目录批量；自动跳过不存在/非图片路径。
  - OCR 异常时返回空文本 `Document`，并在 metadata 标注错误原因。
- 关键文件：
  - Reader 实现：`ocr_research/image_ocr_reader.py`
  - CLI 入口：`ocr_research/main.py`

## 七、复现实验（最小命令）
- 安装依赖（推荐）：`uv sync`
- 仅 OCR：
```
python -m ocr_research.main ocr_research/img
```
- OCR + 索引 + 问答：
```
python -m ocr_research.main ocr_research/img --query "图片中提到了什么？"
```
- 环境变量：在 `.env` 中提供 `OPENAI_API_KEY`；如使用代理，建议同时设置 `OPENAI_BASE_URL`。

## 八、Document 封装讨论
- 拼接策略：按 `(y, x)` 排序换行，通用简单，对问答友好；在表格/多栏场景可考虑保留块级位置信息以提高可解释性。
- 元数据最小集合：`image_path/ocr_model/language/num_text_blocks/avg_confidence`，便于评测、过滤与调试。
