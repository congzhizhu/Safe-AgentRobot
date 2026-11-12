import os
import sys
import json
import uuid
import shutil
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document

# 设置路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
# 让 add_laws_to_db 等模块可以被顶层导入（本文件由 `python -m uvicorn run_ingestion_api:app` 启动，
# 此时 scripts/ 不是包，`from scripts.add_laws_to_db import ...` 会报 No module named 'scripts'）
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent.workflow import LegalClauseRetriever, CaseTextRetriever
from retriever.clip_engine import CLIPCaseEngine
from data_processor.extractor import CaseExtractor

app = FastAPI(title="Safety Agent Data Ingestion API")

# 获取配置
# 注意：这里我们复用 AgentConfig 中的路径
LEGAL_DB_DIR = "data/data_processed/chroma_legal"
# 案例文本向量库：供纯文字/语音提问检索隐患案例（CLIP 只能以图搜图）
CASES_DB_DIR = "data/data_processed/chroma_cases"
CASES_METADATA_PATH = "data/data_processed/cases_metadata.json"


def _resolve_bge_model() -> str:
    # 优先使用项目本地模型目录（ModelScope 下载），其次环境变量，最后默认 HF 名称
    local_bge = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
    if (local_bge / "config.json").exists():
        return str(local_bge)
    return os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-zh-v1.5")


BGE_MODEL_NAME = _resolve_bge_model()


# -----------------
# 工具函数
# -----------------

def _metadata_path() -> Path:
    return PROJECT_ROOT / CASES_METADATA_PATH


def _load_cases() -> list[dict[str, Any]]:
    path = _metadata_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw.get("cases", []) if isinstance(raw, dict) else []
    except Exception:
        return []


def _save_cases(cases: list[dict[str, Any]]) -> None:
    path = _metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")


def _filter_cases(cases: list[dict[str, Any]], lib_type: str | None) -> list[dict[str, Any]]:
    if not lib_type:
        return cases
    if lib_type == "laws":
        return [c for c in cases if "legal_basis" in c]
    if lib_type == "hazards":
        return [c for c in cases if "issue_text" in c]
    return cases


def _legal_store():
    retriever = get_legal_retriever()
    return retriever.store


# -----------------
# 检索单例（重要：本进程是 Chroma 库的唯一所有者。
# Chroma 不支持多进程并发访问同一目录，Agent/WebUI 进程一律通过下方
# /search/laws、/search/image HTTP 接口检索，避免索引损坏。）
# -----------------

_legal_retriever_singleton: Optional[LegalClauseRetriever] = None
_clip_engine_singleton: Any = None


def get_legal_retriever() -> LegalClauseRetriever:
    global _legal_retriever_singleton
    if _legal_retriever_singleton is None:
        _legal_retriever_singleton = LegalClauseRetriever(
            project_root=PROJECT_ROOT,
            metadata_path=CASES_METADATA_PATH,
            persist_directory=LEGAL_DB_DIR,
            embedding_model_name=BGE_MODEL_NAME,
            local=True,
        )
    return _legal_retriever_singleton


def get_clip_engine():
    global _clip_engine_singleton
    if _clip_engine_singleton is None:
        from retriever.clip_engine import CLIPCaseEngine

        _clip_engine_singleton = CLIPCaseEngine(project_root=PROJECT_ROOT)
    return _clip_engine_singleton


_case_retriever_singleton: Optional[CaseTextRetriever] = None


def get_case_text_retriever() -> CaseTextRetriever:
    """案例文本检索器单例（本进程持有 chroma_cases）。"""
    global _case_retriever_singleton
    if _case_retriever_singleton is None:
        _case_retriever_singleton = CaseTextRetriever(
            project_root=PROJECT_ROOT,
            metadata_path=CASES_METADATA_PATH,
            persist_directory=CASES_DB_DIR,
            embedding_model_name=BGE_MODEL_NAME,
            local=True,
        )
    return _case_retriever_singleton


def _sync_case_add(case: dict[str, Any]) -> None:
    """知识库新增/更新案例后同步文本向量库。"""
    try:
        get_case_text_retriever().upsert_case(case)
    except Exception:
        pass


def _sync_case_delete(case_id: str) -> None:
    try:
        get_case_text_retriever().delete_case(case_id)
    except Exception:
        pass


class SearchLawsBody(BaseModel):
    query: str
    top_k: int = 6


@app.post("/search/laws")
def search_laws(body: SearchLawsBody):
    """法规向量检索（供 Agent 进程远程调用）"""
    try:
        retriever = get_legal_retriever()
        results = retriever.search(query=body.query, top_k=max(1, min(body.top_k, 20)))
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = Form(3)):
    """相似隐患案例图像检索（供 Agent 进程远程调用）"""
    tmp_path = Path(tempfile.gettempdir()) / f"search_{uuid.uuid4().hex}_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        engine = get_clip_engine()
        cases = engine.search_similar_cases(query_image=tmp_path, top_k=max(1, min(top_k, 10)))
        return {"status": "success", "results": [c.__dict__ for c in cases]}
    except FileNotFoundError:
        return {"status": "success", "results": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


class SearchCasesBody(BaseModel):
    query: str
    top_k: int = 3


@app.post("/search/cases")
def search_cases(body: SearchCasesBody):
    """隐患案例文本检索（供 Agent 进程远程调用）。

    案例库原本只支持以图搜图，纯文字/语音提问检索不到任何案例。
    这里提供基于案例元数据的语义检索，让无图提问同样能命中知识库。
    """
    try:
        retriever = get_case_text_retriever()
        cases = retriever.search(query=body.query, top_k=max(1, min(body.top_k, 20)))
        return {"status": "success", "results": [c.__dict__ for c in cases]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cases/reindex")
def reindex_cases():
    """按最新的 cases_metadata.json 重建案例文本索引（知识库大批量变更后调用）。"""
    try:
        count = get_case_text_retriever().reindex()
        return {"status": "success", "indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _sync_legal_add(case: dict[str, Any]) -> None:
    store = _legal_store()
    if store:
        store.add_documents(
            [Document(page_content=case.get("legal_basis", ""), metadata={"case_id": case.get("id", ""), "source_file": case.get("source_file", "")})],
            ids=[case.get("id", "")],
        )


def _sync_legal_delete(case_id: str) -> None:
    store = _legal_store()
    if store:
        # 按元数据 case_id 删除，兼容旧文档未设置 id 的场景
        try:
            store._collection.delete(where={"case_id": case_id})
        except Exception:
            try:
                store.delete(ids=[case_id])
            except Exception:
                pass

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    lib_type: str = Form(...), # "laws" 或 "hazards"
):
    """
    上传文件并自动处理入库
    lib_type: "laws" (法律法规库) 或 "hazards" (安全隐患库)
    """
    if lib_type not in ["laws", "hazards"]:
        raise HTTPException(status_code=400, detail="Invalid library type. Must be 'laws' or 'hazards'.")

    # 创建临时上传目录
    temp_dir = PROJECT_ROOT / "data" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if lib_type == "laws":
            return await process_laws(file_path)
        else:
            return await process_hazards(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件（清理失败不能让已成功的入库变成 500，留待下次清理）
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass

async def process_laws(file_path: Path):
    """处理法律法规文件并加入 ChromaDB"""
    try:
        from add_laws_to_db import extract_text_from_file, chunk_text
    except ImportError:
        from scripts.add_laws_to_db import extract_text_from_file, chunk_text
    
    # 1. 提取文本
    text = extract_text_from_file(file_path)
    if not text:
        return {"status": "error", "message": "Could not extract text from file."}
    
    # 2. 切分文本
    chunks = chunk_text(text)
    
    # 3. 加载现有的 metadata 
    metadata_path = PROJECT_ROOT / CASES_METADATA_PATH
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cases = data.get("cases", [])
    else:
        cases = []

    # 4. 构造新记录
    new_cases = []
    source_name = file_path.name.split('_', 1)[-1]
    for i, chunk in enumerate(chunks):
        new_case = {
            "id": f"law_{uuid.uuid4().hex[:8]}",
            "source_file": source_name,
            "legal_basis": chunk
        }
        new_cases.append(new_case)
        cases.append(new_case)

    # 5. 保存更新后的 metadata (LegalClauseRetriever 会读取这个文件来初始化/更新向量库)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=2)

    # 6. 强制刷新向量库（复用本进程单例，直接追加新文档）
    retriever = get_legal_retriever()

    # 直接手动添加新文档
    if retriever.store:
        docs = [Document(page_content=c["legal_basis"], metadata={"case_id": c["id"], "source_file": c["source_file"]}) for c in new_cases]
        retriever.store.add_documents(docs, ids=[c["id"] for c in new_cases])

    return {"status": "success", "message": f"Successfully processed {len(new_cases)} law clauses from {source_name}"}

async def process_hazards(file_path: Path):
    """处理安全隐患文件 (Word/PDF)"""
    # 1. 使用 CaseExtractor 提取
    extractor = CaseExtractor(project_root=PROJECT_ROOT)
    
    # 复制到临时目录因为 extract_all 扫描目录
    task_dir = file_path.parent / f"task_{uuid.uuid4().hex}"
    task_dir.mkdir()
    shutil.copy(file_path, task_dir / file_path.name.split('_', 1)[-1])
    
    try:
        new_cases = extractor.extract_all(task_dir)
        
        if not new_cases:
            return {"status": "error", "message": "No typical hazard cases found in the document."}
        
        # 2. 刷新图像向量库 (如果包含图片)
        engine = CLIPCaseEngine(project_root=PROJECT_ROOT)
        indexed_count = engine.index_images()

        # 3. 同步案例文本向量库，保证纯文字/语音提问也能检索到这批案例
        synced = 0
        for case in new_cases:
            if not case.get("id"):
                continue
            _sync_case_add(case)
            synced += 1
        if synced == 0 and new_cases:
            # 抽取结果缺 id（不应发生），退化为全量重建
            try:
                synced = get_case_text_retriever().reindex()
            except Exception:
                synced = 0

        return {
            "status": "success", 
            "message": f"Extracted {len(new_cases)} hazard cases. Indexed {indexed_count} images. Synced {synced} case texts."
        }
    finally:
        try:
            shutil.rmtree(task_dir, ignore_errors=True)
        except Exception:
            pass


# -----------------
# 知识库 CRUD 接口
# -----------------


class KbCreateBody(BaseModel):
    lib_type: str  # laws / hazards
    content: str
    source_file: str | None = "manual"


class KbUpdateBody(BaseModel):
    lib_type: str
    case_id: str
    content: str
    source_file: str | None = None


class KbDeleteBody(BaseModel):
    case_id: str


@app.get("/kb")
def list_kb(lib_type: str | None = None):
    cases = _filter_cases(_load_cases(), lib_type)
    return {"cases": cases}


@app.post("/kb/create")
def create_kb(body: KbCreateBody):
    if body.lib_type not in {"laws", "hazards"}:
        raise HTTPException(status_code=400, detail="lib_type must be laws or hazards")

    cases = _load_cases()
    if body.lib_type == "laws":
        new_case = {
            "id": f"law_{uuid.uuid4().hex[:8]}",
            "source_file": body.source_file or "manual",
            "legal_basis": body.content.strip(),
        }
    else:
        new_case = {
            "id": f"hazard_{uuid.uuid4().hex[:8]}",
            "source_file": body.source_file or "manual",
            "issue_text": body.content.strip(),
        }

    cases.append(new_case)
    _save_cases(cases)

    if body.lib_type == "laws":
        _sync_legal_add(new_case)
    else:
        _sync_case_add(new_case)

    return {"status": "success", "case": new_case}


@app.post("/kb/update")
def update_kb(body: KbUpdateBody):
    if body.lib_type not in {"laws", "hazards"}:
        raise HTTPException(status_code=400, detail="lib_type must be laws or hazards")

    cases = _load_cases()
    found = False
    for case in cases:
        if case.get("id") == body.case_id:
            if body.lib_type == "laws":
                case["legal_basis"] = body.content.strip()
            else:
                case["issue_text"] = body.content.strip()
            if body.source_file:
                case["source_file"] = body.source_file
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="case not found")

    _save_cases(cases)

    if body.lib_type == "laws":
        _sync_legal_delete(body.case_id)
        case = next(c for c in cases if c.get("id") == body.case_id)
        _sync_legal_add(case)
    else:
        case = next(c for c in cases if c.get("id") == body.case_id)
        _sync_case_add(case)

    return {"status": "success"}


@app.post("/kb/delete")
def delete_kb(body: KbDeleteBody):
    cases = _load_cases()
    new_cases = [c for c in cases if c.get("id") != body.case_id]
    if len(new_cases) == len(cases):
        raise HTTPException(status_code=404, detail="case not found")

    _save_cases(new_cases)

    # 如果是法律库条目则同步法规向量库；否则同步案例文本向量库
    if body.case_id.startswith("law_"):
        _sync_legal_delete(body.case_id)
    else:
        _sync_case_delete(body.case_id)

    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
