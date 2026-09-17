"""Cognia embedding 层：本地 fastembed 语义向量。

背景（已核实）：
- 工蜂 Gateway 不支持 embedding（/v1/embeddings 返回 502，/v1/models 无 embedding 模型）。
- huggingface.co 与 hf-mirror.com 均可达，本地推理可行且零外部 API 成本。
- 磁盘约束：/data 分区空间紧张，torch（sentence-transformers 依赖）+ CUDA 依赖
  动辄数 GB 装不下（实测 No space left on device），故选用 fastembed
  （纯 ONNX Runtime CPU 推理，不依赖 torch，模型量化后轻量，约 200MB 内）。

选型：BAAI/bge-small-zh-v1.5（quantized ONNX，中文友好）。
懒加载单例 + 安全降级（加载失败返回 None，调用方退化为归一化 name 精确合并）。

模型 ID / 镜像可通过环境变量覆盖：
- EMBEDDING_MODEL（默认 BAAI/bge-small-zh-v1.5）
- EMBEDDING_HF_MIRROR（默认空；内网可设 https://hf-mirror.com 加速）
"""

import os

_model = None
_model_name = None


def _default_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def _ensure_hf_mirror() -> None:
    """若配置了 EMBEDDING_HF_MIRROR 且未显式配置 HF_ENDPOINT，则启用镜像加速下载。"""
    mirror = os.getenv("EMBEDDING_HF_MIRROR")
    if mirror and not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = mirror


def get_embedding_model():
    """懒加载单例。加载失败返回 None（调用方安全降级，不抛异常）。"""
    global _model, _model_name
    name = _default_model()
    if _model is not None and _model_name == name:
        return _model
    _model_name = name
    try:
        _ensure_hf_mirror()
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=name)
        return _model
    except Exception:
        # 依赖缺失 / 模型下载失败 / 内存不足等：返回 None，不阻断主流程
        _model = None
        return None


def embed_text(text: str) -> list[float] | None:
    """文本 → 归一化向量。模型不可用 / 文本为空返回 None。"""
    if not text or not text.strip():
        return None
    model = get_embedding_model()
    if model is None:
        return None
    try:
        embeddings = list(model.embed([text.strip()]))
        if not embeddings:
            return None
        vec = embeddings[0]
        # fastembed 返回 numpy 数组；归一化到单位向量便于 cosine 直接点积
        norm = float(sum(float(x) * float(x) for x in vec)) ** 0.5
        if norm < 1e-12:
            return [float(x) for x in vec]
        return [float(x) / norm for x in vec]
    except Exception:
        return None


def cosine(a: list[float], b: list[float]) -> float:
    """两向量的 cosine 相似度。任一为空 / 维度不一致 / 模近零时返回 0.0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)
