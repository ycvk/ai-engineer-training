from __future__ import annotations

import os
from typing import List, Union, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


def _is_image(path: str) -> bool:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return os.path.splitext(path)[1].lower() in exts


class ImageOCRReader(BaseReader):
    """使用 PaddleOCR (PP-OCRv5) 从图像提取文本并返回 LlamaIndex Document 列表。

    仅使用 CPU，默认中文模型，可通过 `lang` 与 `ocr_version` 覆盖。
    """

    def __init__(
        self,
        lang: str = "ch",
        ocr_version: str = "PP-OCRv5",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self.lang = lang
        self.ocr_version = ocr_version
        self.device = device or "cpu"
        self._ocr = None
        self._paddle_available: bool | None = None
        self._ocr_kwargs = dict(kwargs) if kwargs else {}

    def _ensure_ocr(self):
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception:
            self._paddle_available = False
            return None

        try:
            # 尽量只设置稳定可用的参数，避免不同版本不兼容
            self._ocr = PaddleOCR(
                lang=self.lang,
                ocr_version=self.ocr_version,
                device=self.device,
                **self._ocr_kwargs,
            )
            self._paddle_available = True
            return self._ocr
        except Exception:
            # 构造失败时降级
            self._paddle_available = False
            self._ocr = None
            return None

    @staticmethod
    def _parse_ocr_result(raw) -> List[tuple[str, float, float, float]]:
        """解析 PaddleOCR ocr() 的返回结果为 (text, conf, top_y, left_x) 列表。

        兼容形如：
        result = [ [ [box_points], (text, score) ], ... ]  # 单图
        或多图形式： [ result_img1, result_img2, ... ]
        """
        # 规范化为单图的行列表
        lines = raw
        if isinstance(raw, list) and raw and isinstance(raw[0], list) and raw and (
            len(raw) > 0 and raw and raw and isinstance(raw[0][0], (list, tuple))
        ) and not (
            isinstance(raw[0][0], (list, tuple)) and len(raw[0]) >= 1 and isinstance(raw[0][0], (list, tuple)) and isinstance(raw[0][0][0], (list, tuple))
        ):
            # 这种判断不可靠，统一在下方覆盖
            pass

        # PaddleOCR 通常返回形如 [ [ line1, line2, ... ] ]
        if isinstance(raw, list) and raw and isinstance(raw[0], list) and raw and (
            len(raw) > 0 and isinstance(raw[0], list)
        ) and raw and len(raw) == 1:
            lines = raw[0]

        out: List[tuple[str, float, float, float]] = []
        if not isinstance(lines, Iterable):
            return out
        for item in lines:
            try:
                box, recog = item  # box: 4点坐标; recog: (text, conf)
                text, conf = recog
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                top_y = min(ys)
                left_x = min(xs)
                out.append((str(text), float(conf), float(top_y), float(left_x)))
            except Exception:
                continue
        return out

    @staticmethod
    def _join_text(blocks: List[tuple[str, float, float, float]]) -> str:
        # 读取顺序：先按 y 再按 x
        blocks_sorted = sorted(blocks, key=lambda b: (b[2], b[3]))
        return "\n".join(b[0].strip() for b in blocks_sorted if b[0].strip())

    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        paths: List[str]
        if isinstance(file, str):
            paths = [file]
        else:
            paths = list(file)

        valid_paths: List[str] = []
        for p in paths:
            if not os.path.exists(p):
                continue
            if os.path.isdir(p):
                for name in os.listdir(p):
                    fp = os.path.join(p, name)
                    if os.path.isfile(fp) and _is_image(fp):
                        valid_paths.append(fp)
            elif os.path.isfile(p) and _is_image(p):
                valid_paths.append(p)

        if not valid_paths:
            return []

        ocr = self._ensure_ocr()
        documents: List[Document] = []

        for img_path in valid_paths:
            try:
                blocks: List[tuple[str, float, float, float]]
                if ocr is not None:
                    raw = ocr.ocr(img_path, cls=True)
                    blocks = self._parse_ocr_result(raw)
                else:
                    # Fallback: RapidOCR (ONNXRuntime)
                    try:
                        from rapidocr_onnxruntime import RapidOCR  # type: ignore
                    except Exception as e:
                        raise RuntimeError(
                            "PaddleOCR 初始化失败且未安装 RapidOCR 作为降级方案。"
                        ) from e
                    rocr = RapidOCR()
                    result, _ = rocr(img_path)
                    tmp: List[tuple[str, float, float, float]] = []
                    for item in result or []:
                        try:
                            box, text, conf = item
                            xs = [p[0] for p in box]
                            ys = [p[1] for p in box]
                            top_y, left_x = min(ys), min(xs)
                            tmp.append((str(text), float(conf), float(top_y), float(left_x)))
                        except Exception:
                            continue
                    blocks = tmp
                text = self._join_text(blocks)
                confs = [b[1] for b in blocks]
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                metadata = {
                    "image_path": os.path.abspath(img_path),
                    "ocr_model": self.ocr_version if ocr is not None else "RapidOCR",
                    "language": self.lang,
                    "num_text_blocks": len(blocks),
                    "avg_confidence": round(avg_conf, 4),
                }
                documents.append(Document(text=text, metadata=metadata))
            except Exception as e:
                err_meta = {
                    "image_path": os.path.abspath(img_path),
                    "ocr_model": self.ocr_version,
                    "language": self.lang,
                    "error": f"OCR 失败: {type(e).__name__}: {e}",
                }
                documents.append(Document(text="", metadata=err_meta))

        return documents
