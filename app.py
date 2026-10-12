import os
import importlib.util
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from routes.auth import router as auth_router

# Import the rag_chain logic from search.py
from search import rag_chain

# ==============================================================================
# IMPORT DYNAMIC PIPELINE (from 'dynamic_pipeline.py')
# ==============================================================================
from dynamic_pipeline import (
    start_dynamic_pipeline,
    stop_dynamic_pipeline,
    check_and_process_new_documents,
    dispatch_files_for_processing,
    extract_and_process_zip,
    get_task_progress,
    get_all_active_tasks,
    load_processed_registry,
)


# ==============================================================================
# FASTAPI LIFESPAN: AUTO-RUN PIPELINE ON STARTUP & BACKGROUND WATCHER
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Check for new documents in 'Data' folder, queue to Redis/Celery, and start watcher
    print("\n🚀 [FastAPI Startup] Initializing Enterprise Celery RAG Pipeline...")
    start_dynamic_pipeline(data_dir="Data")
    yield
    # 2. Shutdown: Stop background file watcher
    print("\n🛑 [FastAPI Shutdown] Stopping Dynamic RAG Pipeline watcher...")
    stop_dynamic_pipeline()


# Initialize FastAPI app with lifespan
app = FastAPI(title="Qdrant RAG Chat API (Enterprise Celery & Redis)", lifespan=lifespan)

# Add CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)

# ==========================================
# FRONTEND UI ENDPOINT
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found!</h1>"

# ==========================================
# DYNAMIC PIPELINE & CELERY PROGRESS ENDPOINTS
# ==========================================
@app.get("/pipeline/status")
async def get_pipeline_status():
    """Returns the current registry of indexed files and chunk counts."""
    registry = load_processed_registry()
    total_chunks = sum(item.get("chunks_count", 0) for item in registry.values())
    return {
        "total_documents": len(registry),
        "total_chunks": total_chunks,
        "indexed_documents": registry
    }

@app.post("/pipeline/sync")
async def sync_pipeline():
    """
    Manually triggers check of Data folder. If 1 or 1,000 new documents exist,
    dispatches them to Redis & Celery and returns the Celery Task ID.
    """
    dispatch_info = check_and_process_new_documents("Data")
    return {
        "status": "triggered",
        "dispatch_info": dispatch_info
    }

@app.get("/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """
    Returns real-time Celery task progress (0% - 100%), per-document percentages, and stage details.
    """
    return get_task_progress(task_id)

@app.get("/tasks/{task_id}/documents")
async def get_task_documents_endpoint(task_id: str):
    """
    Returns breakdown of each document's processing percentage (0% - 100%), stage, and chunks.
    """
    info = get_task_progress(task_id)
    return {
        "task_id": task_id,
        "status": info.get("status"),
        "overall_progress": info.get("progress"),
        "total_documents": info.get("total"),
        "processed_documents": info.get("processed"),
        "current_file": info.get("current_file"),
        "documents": info.get("documents", [])
    }

@app.get("/tasks")
async def get_all_tasks_endpoint():
    """Returns list of all active/recent Celery tasks."""
    return get_all_active_tasks()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Direct document or ZIP archive upload endpoint.
    - If .zip: extracts all documents and dispatches batch task to Redis & Celery.
    - If document (.pdf, .txt, .docx, .xlsx, .csv): saves to Data/ and dispatches Celery task.
    Returns Celery Task ID so client can monitor percentage progress.
    """
    data_dir = "Data"
    os.makedirs(data_dir, exist_ok=True)
    destination = os.path.join(data_dir, file.filename)

    try:
        content = await file.read()
        with open(destination, "wb") as f:
            f.write(content)

        # Handle ZIP archive upload (e.g. bulk 1,000 documents)
        if file.filename.lower().endswith(".zip"):
            result = extract_and_process_zip(destination, extract_to=data_dir)
            try:
                os.remove(destination)  # clean up zip archive after extraction
            except Exception:
                pass
            return {
                "message": f"Successfully extracted ZIP '{file.filename}' and queued for processing.",
                "type": "zip_batch",
                "task_id": result.get("task_id"),
                "total_documents": result.get("total_documents", 0),
                "status": "QUEUED"
            }
        else:
            # Single document upload -> dispatch Celery task
            dispatch_result = dispatch_files_for_processing([destination])
            return {
                "message": f"Successfully uploaded '{file.filename}' and queued for processing.",
                "type": "single_document",
                "task_id": dispatch_result.get("task_id"),
                "total_documents": 1,
                "status": "QUEUED"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")

# ==========================================
# FASTAPI CHAT ENDPOINT
# ==========================================
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # The rag_chain does the heavy lifting:
        # 1. Embeds the 'request.question'
        # 2. Does a semantic search (Cosine similarity) against Qdrant
        # 3. Injects the winning chunks into the prompt
        # 4. Sends it to Azure OpenAI to generate a human-readable answer
        answer = rag_chain.invoke(request.question)
        
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8081, reload=True)

