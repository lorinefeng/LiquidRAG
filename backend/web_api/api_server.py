# -*- coding: utf-8 -*-
"""
说明：
- 本模块使用 FastAPI 在本地为现有 RAGPipeline 提供 HTTP API 封装，便于前端（HTML/CSS/JavaScript/React）通过浏览器调用。

外部库用途说明：
- fastapi：用于快速构建 Web API 服务，定义路由与请求/响应模型，并自动生成接口文档。
- uvicorn：用于在本地以 ASGI 方式启动 FastAPI 应用（开发/本地部署场景）。
- pydantic：用于定义请求/响应的数据模型，提供类型与校验支持（FastAPI 内部使用）。
- python-multipart：用于支持 multipart/form-data 的文件上传（如文档上传）。

注意：
- 结合你的环境兼容性问题（NumPy 2.x 与部分模块不兼容），如遇到相关错误，请考虑将 NumPy 降级到 <2：pip install "numpy<2"；或升级依赖以适配 2.x。
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import logging
from send2trash import send2trash

# 说明：导入你现有的 RAGPipeline 等模块（位于 scripts/ 与 configs/ 下）
# 这些模块用于本地检索与生成，不依赖外部推理 API。
from scripts.rag_pipeline import RAGPipeline
from configs.rag_config import RAGConfig
from scripts.gpu_manager import GPUManager

app = FastAPI(title="LiquidRAG Web API", version="1.0.0", docs_url="/docs", redoc_url="/redoc")

# 日志配置：输出详细启动信息、路由注册、占位数据库连接情况
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger("web_api")
UPLOADS_DIR = Path("uploads").resolve()

def _is_under_uploads(path: Path) -> bool:
    try:
        return str(path.resolve()).startswith(str(UPLOADS_DIR))
    except Exception:
        return False

# 说明：启用 CORS 以便前端页面（可能不同源）可以跨域请求本地 API。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许所有来源；生产建议限定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 数据模型（请求/响应） ============
class AskRequest(BaseModel):
    """问答请求模型（前端传入）。"""
    query: str
    top_k: Optional[int] = None
    return_sources: bool = True


class SourceItem(BaseModel):
    """检索来源条目（用于响应展示）。"""
    source: Optional[str] = None
    similarity: Optional[float] = None
    chunk_text: Optional[str] = None
    doc_id: Optional[str] = None


class AskResponse(BaseModel):
    """问答响应模型（返回给前端）。"""
    answer: str
    num_sources: int
    sources: Optional[List[SourceItem]] = None
    retrieval_time: float
    generation_time: float
    total_time: float
    gpu_metrics: Optional[List[Dict[str, Any]]] = None


class RetrieveResponse(BaseModel):
    """仅检索响应（不生成答案）。"""
    num_sources: int
    sources: List[SourceItem]
    retrieval_time: float


# ============ 应用生命周期事件 ============
@app.on_event("startup")
def on_startup():
    """应用启动时初始化 RAGPipeline，避免每次请求重复加载模型。"""
    global rag
    logger.info("服务初始化开始")
    cfg = RAGConfig()
    try:
        rag = RAGPipeline(cfg)
        logger.info("RAGPipeline 初始化完成")
    except Exception as e:
        logger.error(f"RAGPipeline 初始化失败: {e}")
        class FallbackRag:
            llm_model_name = "fallback"
            embedding_model_name = "fallback-embedding"
            def ask(self, query: str, top_k: Optional[int] = None, return_sources: bool = True):
                return {
                    "answer": "后端向量与嵌入未初始化，返回占位回答。请配置模型路径并重启。",
                    "num_sources": 0,
                    "sources": [],
                    "retrieval_time": 0.0,
                    "generation_time": 0.0,
                    "total_time": 0.0,
                }
        rag = FallbackRag()

    # 数据库连接情况（占位说明）：如未配置数据库，进行提示
    logger.info("数据库连接情况：未配置数据库，跳过连接")

    # 路由注册信息
    routes = [r.path for r in app.routes]
    logger.info(f"API 路由注册完成，共 {len(routes)} 条：{routes}")


# ============ 健康检查与状态接口 ============
@app.get("/api/v1/health")
def health() -> Dict[str, Any]:
    """健康检查接口：用于前端或监控确认服务是否就绪。"""
    return {"ok": True}

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """Kubernetes/通用健康检查惯例路径。"""
    return {"ok": True}


@app.get("/api/v1/status")
def status() -> Dict[str, Any]:
    """系统状态接口：返回后端管线初始化状态、模型信息等。若需要显存信息，可集成 gpu_manager。"""
    # 说明：此处仅提供示例状态，你可以结合 scripts/gpu_manager.py 或 torch.cuda 进一步获取显存使用情况。
    gm = GPUManager()
    gpu_info = gm.get_system_info()
    return {
        "pipeline_initialized": rag is not None,
        "model": getattr(rag, "llm_model_name", "local-LLM"),
        "embedding_model": getattr(rag, "embedding_model_name", "Qwen3-Embedding-0.6B"),
        "gpu": gpu_info,
    }


# ============ 核心端点：问答/检索 ============
@app.post("/api/v1/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    提问检索接口：封装 RAGPipeline.ask。
    调用流程：检索相关文档 → 组织上下文 → 调用本地 LLM 生成答案 → 返回答案与来源信息。
    """
    start = time.time()
    gm = GPUManager()
    samples: List[Dict[str, Any]] = []
    running = True
    import threading
    def sampler():
        while running:
            m = gm.monitor_memory_usage(0)
            m["ts"] = time.time()
            samples.append(m)
            time.sleep(0.5)
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    result = rag.ask(req.query, top_k=req.top_k, return_sources=req.return_sources)
    running = False
    try:
        t.join(timeout=1)
    except Exception:
        pass
    end = time.time()

    # 说明：RAGPipeline.ask 返回结构应包含 answer、num_sources、sources、时间统计等字段。
    # 将 sources 映射为 SourceItem 列表，以便前端统一展示。
    src_list = []
    for s in result.get("sources", []) or []:
        src_list.append(SourceItem(
            source=s.get("source"),
            similarity=s.get("similarity"),
            chunk_text=s.get("chunk_text"),
            doc_id=s.get("doc_id")
        ))

    return AskResponse(
        answer=result.get("answer", ""),
        num_sources=result.get("num_sources", 0),
        sources=src_list if req.return_sources else None,
        retrieval_time=result.get("retrieval_time", 0.0),
        generation_time=result.get("generation_time", 0.0),
        total_time=result.get("total_time", end - start),
        gpu_metrics=samples,
    )


@app.get("/api/v1/retrieve", response_model=RetrieveResponse)
def retrieve(query: str, top_k: int = 5) -> RetrieveResponse:
    """
    仅检索接口：用于前端查看检索到的来源，不进行答案生成。
    实现方式：复用 RAGPipeline.ask 的检索部分（return_sources=True），忽略答案字段。
    """
    start = time.time()
    result = rag.ask(query, top_k=top_k, return_sources=True)
    end = time.time()

    src_list = []
    for s in result.get("sources", []) or []:
        src_list.append(SourceItem(
            source=s.get("source"),
            similarity=s.get("similarity"),
            chunk_text=s.get("chunk_text"),
            doc_id=s.get("doc_id")
        ))

    return RetrieveResponse(
        num_sources=result.get("num_sources", 0),
        sources=src_list,
        retrieval_time=result.get("retrieval_time", end - start),
    )


# ============ 文档上传接口（索引构建） ============
@app.post("/api/v1/upload")
async def upload_doc(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    文档上传接口：接收文件并写入向量索引。
    说明：为保持最小闭环，这里提供占位实现。你需要结合 document_loader 与 vector_store 的实际函数，
    将文件解析为文本→分块→生成嵌入→写入 ChromaDB。
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".md", ".txt"]:
        raise HTTPException(status_code=400, detail="仅支持上传 .md 与 .txt 文件")

    try:
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        save_path = uploads_dir / filename
        content_bytes = await file.read()
        with open(save_path, "wb") as f:
            f.write(content_bytes)

        # 调用向量存储进行解析、分块、嵌入与写入
        written = rag.vector_store.add_document_file(str(save_path))

        if written > 0:
            return {"ok": True, "filename": filename, "written_chunks": written, "message": "文档已解析并写入索引"}
        else:
            raise HTTPException(status_code=400, detail="文件解析失败或内容为空，未写入索引")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")


# ============ 文档列表/删除（可选占位） ============
@app.get("/api/v1/documents")
def list_documents(page: int = 1, page_size: int = 20, q: Optional[str] = None) -> Dict[str, Any]:
    """
    文档列表接口（占位）：返回索引中的已管理文档清单。
    说明：需结合 vector_store 或元数据存储实现实际列表查询。
    """
    try:
        data = rag.vector_store.list_documents(page=page, page_size=page_size, query=q)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文档列表失败: {e}")


@app.delete("/api/v1/documents/{doc_id:path}")
def delete_document(doc_id: str) -> Dict[str, Any]:
    try:
        p = Path(doc_id)
        logger.info(f"删除请求 doc_id={doc_id}")
        if not p.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        if not _is_under_uploads(p):
            raise HTTPException(status_code=400, detail="仅允许删除 uploads 目录下文件")
        try:
            send2trash(str(p))
        except PermissionError:
            raise HTTPException(status_code=403, detail="权限不足，无法移动到回收站")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"移动到回收站失败: {e}")
        removed = rag.vector_store.delete_by_source(str(p))
        return {"ok": bool(removed), "doc_id": str(p), "removed": removed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")

@app.post("/api/v1/documents/batch_delete")
def batch_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
    sources = payload.get("sources") or []
    if not isinstance(sources, list) or not sources:
        raise HTTPException(status_code=400, detail="sources 必须为非空列表")
    try:
        total_removed = 0
        for s in sources:
            p = Path(str(s))
            if not _is_under_uploads(p):
                continue
            if p.exists():
                try:
                    send2trash(str(p))
                except Exception:
                    pass
            total_removed += rag.vector_store.delete_by_source(str(p))
        return {"ok": True, "count": total_removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量删除失败: {e}")

@app.post("/api/v1/reindex")
def reindex(payload: Dict[str, Any] = None) -> Dict[str, Any]:
    src_dir = None
    if payload and isinstance(payload, dict):
        src_dir = payload.get("source_dir")
    try:
        rag.vector_store.reset_collection()
        from scripts.document_loader import DocumentLoader
        from configs.rag_config import RAGConfig
        cfg = RAGConfig()
        loader = DocumentLoader(cfg)
        base_dir = src_dir or cfg.DOCS_SOURCE_DIR
        docs = loader.load_documents(base_dir)
        ok = rag.vector_store.add_documents(docs)
        return {"ok": bool(ok), "indexed_chunks": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建索引失败: {e}")


# ============ 对话历史（可选占位） ============
_conversation: List[Dict[str, str]] = []


@app.post("/api/v1/clear-history")
def clear_history() -> Dict[str, Any]:
    """清空后端维护的对话历史（如需在后端保留对话上下文）。"""
    _conversation.clear()
    return {"ok": True, "message": "后端对话历史已清空"}


if __name__ == "__main__":
    # 说明：使用 uvicorn 启动本地 API 服务，便于开发联调。
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
@app.get("/api/v1/metrics/gpu")
def metrics_gpu() -> Dict[str, Any]:
    gm = GPUManager()
    info = gm.get_system_info()
    return info