import os
import sys
import time
import json
import zipfile
import hashlib
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from threading import Thread, Lock
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Core LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Azure OpenAI & Qdrant
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = object
    WATCHDOG_AVAILABLE = False

# Celery & Redis
from celery import Celery
from celery.result import AsyncResult

# Initialize Rich Console & Env
console = Console()
load_dotenv()

# Directories & Registry
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_files.json")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".xlsx", ".xls", ".docx", ".doc", ".csv"}

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Local Task Tracking Cache (keeps progress available even across polling/in-memory)
_task_progress_cache: Dict[str, Dict[str, Any]] = {}
_task_cache_lock = Lock()
_active_task_ids: List[str] = []

# Watcher state
_ingestion_lock = Lock()
_active_observer: Optional[Any] = None
_polling_thread: Optional[Thread] = None
_stop_polling = False


# ==============================================================================
# 1. CELERY APPLICATION CONFIGURATION
# ==============================================================================

celery_app = Celery(
    "rag_pipeline",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,
    broker_connection_retry_on_startup=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)


# ==============================================================================
# 2. REGISTRY & FILE HASH UTILITIES
# ==============================================================================

def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file to detect content changes."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_processed_registry() -> Dict[str, Any]:
    """Load the registry of already ingested documents."""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read registry ({e}), returning empty.[/yellow]")
            return {}
    return {}


def save_processed_registry(registry: Dict[str, Any]) -> None:
    """Save updated registry to disk."""
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4)
    except Exception as e:
        console.print(f"[bold red]✖ Error saving registry: {e}[/bold red]")


def wait_for_file_ready(filepath: str, max_retries: int = 10, delay: float = 0.5) -> bool:
    """Wait until a file has finished writing and is readable."""
    last_size = -1
    for _ in range(max_retries):
        if not os.path.exists(filepath):
            return False
        try:
            current_size = os.path.getsize(filepath)
            if current_size > 0 and current_size == last_size:
                with open(filepath, "rb"):
                    return True
            last_size = current_size
        except (PermissionError, OSError):
            pass
        time.sleep(delay)
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


# ==============================================================================
# 3. DOCUMENT LOADER & SPLITTER
# ==============================================================================

def load_single_document(filepath: str) -> List[Document]:
    """
    Load a single document using the appropriate LangChain loader.
    Supports PDF, TXT, Excel (.xlsx, .xls), Word (.docx, .doc), and CSV.
    """
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        return loader.load()

    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        try:
            loader = TextLoader(filepath, encoding="utf-8")
            return loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(filepath, encoding="latin-1")
            return loader.load()

    elif ext in [".xlsx", ".xls"]:
        from langchain_community.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(filepath, mode="elements")
        return loader.load()

    elif ext in [".docx", ".doc"]:
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(filepath)
            return loader.load()
        except ImportError:
            raise ImportError("Docx2txtLoader is not available in langchain_community.")

    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(filepath)
        return loader.load()

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def split_documents(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[Document]:
    """Splits documents into overlapping chunks with metadata."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    return splitter.split_documents(documents)


# ==============================================================================
# 4. EMBEDDINGS & VECTOR STORE (QDRANT)
# ==============================================================================

def get_embeddings_model() -> AzureOpenAIEmbeddings:
    """Initialize Azure OpenAI Embeddings."""
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not all([api_key, endpoint, api_version, deployment]):
        raise ValueError("Missing Azure OpenAI credentials in environment (.env).")

    return AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment
    )


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant Cloud or cluster."""
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not all([qdrant_url, qdrant_api_key]):
        raise ValueError("Missing Qdrant credentials (QDRANT_URL, QDRANT_API_KEY).")

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def get_or_create_vector_store() -> QdrantVectorStore:
    """Ensures Qdrant collection exists with cosine similarity and returns VectorStore."""
    client = get_qdrant_client()
    embeddings = get_embeddings_model()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "Rag_project")

    if not client.collection_exists(collection_name=collection_name):
        console.print(f"[yellow]Collection '{collection_name}' not found. Creating it...[/yellow]")
        test_vec = embeddings.embed_query("test vector dimensions")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(test_vec), distance=Distance.COSINE),
        )
        console.print(f"[bold green]✔ Collection '{collection_name}' created successfully![/bold green]")

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )


def store_chunks_in_qdrant(chunks: List[Document]) -> int:
    """Stores chunks into Qdrant vector database."""
    if not chunks:
        return 0
    vector_store = get_or_create_vector_store()
    vector_store.add_documents(chunks)
    return len(chunks)


# ==============================================================================
# 5. CORE INGESTION WORKFLOW
# ==============================================================================

def execute_document_ingestion(filepath: str, force: bool = False, progress_callback=None) -> Dict[str, Any]:
    """
    Executes the ingestion pipeline on a single file:
    Load -> Split -> Embed & Store in Qdrant -> Update Registry
    """
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    if not os.path.isfile(filepath) or filename.startswith(".") or filename == "desktop.ini":
        return {"status": "skipped", "file": filename, "reason": "Ignored system/hidden file"}

    if ext not in SUPPORTED_EXTENSIONS:
        return {"status": "skipped", "file": filename, "reason": f"Unsupported format '{ext}'"}

    if not wait_for_file_ready(filepath):
        return {"status": "error", "file": filename, "reason": "File was locked or incomplete"}

    current_hash = calculate_file_hash(filepath)
    registry = load_processed_registry()

    if not force and filename in registry:
        prev = registry[filename]
        if prev.get("hash") == current_hash and prev.get("status") == "completed":
            if progress_callback:
                progress_callback(100, f"Already processed: {filename}")
            return {
                "status": "skipped",
                "file": filename,
                "reason": "File unchanged, already indexed",
                "chunks": prev.get("chunks_count", 0)
            }

    start_time = time.time()
    try:
        # Stage 1: Load (25%)
        if progress_callback:
            progress_callback(25, f"Loading: {filename}")
        raw_docs = load_single_document(filepath)
        if not raw_docs:
            return {"status": "warning", "file": filename, "reason": "Empty document"}

        # Stage 2: Split (50%)
        if progress_callback:
            progress_callback(50, f"Splitting chunks: {filename}")
        chunks = split_documents(raw_docs)

        # Stage 3: Embed & Store in Qdrant (75%)
        if progress_callback:
            progress_callback(75, f"Embedding & indexing in Qdrant: {filename}")
        stored_count = store_chunks_in_qdrant(chunks)

        duration = round(time.time() - start_time, 2)

        # Stage 4: Update Registry (100%)
        registry[filename] = {
            "filename": filename,
            "filepath": filepath,
            "hash": current_hash,
            "chunks_count": stored_count,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "status": "completed",
            "duration_sec": duration
        }
        save_processed_registry(registry)

        if progress_callback:
            progress_callback(100, f"Completed: {filename}")

        return {
            "status": "success",
            "file": filename,
            "chunks": stored_count,
            "duration_sec": duration
        }
    except Exception as e:
        console.print(f"[bold red]✖ Error processing {filename}:[/bold red] {e}")
        return {"status": "error", "file": filename, "error": str(e)}


# ==============================================================================
# 6. CELERY TASKS WITH REAL-TIME PERCENTAGE TRACKING
# ==============================================================================

@celery_app.task(bind=True, name="tasks.process_single_document")
def process_document_task(self, filepath: str, force: bool = False) -> Dict[str, Any]:
    """Celery task for processing a single document with granular stage progress."""
    filename = os.path.basename(filepath)

    def update_progress(percent: int, stage_desc: str):
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": percent,
                "stage": stage_desc,
                "filename": filename,
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }
        )

    self.update_state(state="PROGRESS", meta={"progress": 5, "stage": f"Starting {filename}", "filename": filename})
    result = execute_document_ingestion(filepath, force=force, progress_callback=update_progress)
    return {
        "progress": 100,
        "status": "completed",
        "result": result
    }


@celery_app.task(bind=True, name="tasks.process_batch_documents")
def process_batch_documents_task(self, file_paths: List[str], force: bool = False) -> Dict[str, Any]:
    """
    Celery task that processes a batch (e.g., 100 or 1,000 documents)
    and reports exact overall percentage progress and per-document percentage to Celery & Redis.
    """
    total = len(file_paths)
    if total == 0:
        return {"progress": 100, "status": "completed", "total": 0, "processed": 0, "results": [], "documents": []}

    results = []
    total_chunks = 0
    docs_state = [
        {
            "filename": os.path.basename(fp),
            "status": "pending",
            "progress": 0,
            "stage": "Waiting in queue",
            "chunks": 0
        }
        for fp in file_paths
    ]

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 0,
            "processed": 0,
            "total": total,
            "stage": f"Starting batch of {total} documents",
            "current_file": os.path.basename(file_paths[0]),
            "current_file_progress": 0,
            "current_file_stage": "Queued",
            "documents": docs_state,
            "results": []
        }
    )

    for idx, filepath in enumerate(file_paths):
        filename = os.path.basename(filepath)
        current_num = idx + 1
        docs_state[idx]["status"] = "processing"
        docs_state[idx]["stage"] = "Starting..."

        def doc_progress_cb(doc_pct: int, stage_desc: str):
            docs_state[idx]["progress"] = doc_pct
            docs_state[idx]["stage"] = stage_desc
            blended_pct = min(99, int(((idx + (doc_pct / 100.0)) / total) * 100))
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": blended_pct,
                    "processed": idx,
                    "total": total,
                    "current_file": filename,
                    "current_file_progress": doc_pct,
                    "current_file_stage": stage_desc,
                    "stage": f"[{current_num}/{total}] {stage_desc}",
                    "documents": list(docs_state),
                    "results": results
                }
            )

        res = execute_document_ingestion(filepath, force=force, progress_callback=doc_progress_cb)
        results.append(res)
        chunks = res.get("chunks", 0)
        docs_state[idx]["status"] = res.get("status", "completed")
        docs_state[idx]["progress"] = 100
        docs_state[idx]["stage"] = "Completed" if res.get("status") == "success" else res.get("reason", "Finished")
        docs_state[idx]["chunks"] = chunks
        if res.get("status") == "success":
            total_chunks += chunks

        progress_pct = int(((current_num) / total) * 100)
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": progress_pct,
                "processed": current_num,
                "total": total,
                "current_file": filename,
                "current_file_progress": 100,
                "current_file_stage": "Completed",
                "stage": f"Finished {current_num}/{total}: {filename}",
                "documents": list(docs_state),
                "results": results
            }
        )

    return {
        "progress": 100,
        "status": "completed",
        "total": total,
        "processed": total,
        "total_chunks": total_chunks,
        "documents": docs_state,
        "results": results
    }


# ==============================================================================
# 7. ASYNCHRONOUS TASK DISPATCHER & PROGRESS CHECKER
# ==============================================================================

def update_local_task_cache(task_id: str, state: str, progress: int, stage: str, meta: Dict[str, Any] = None):
    """Keep track of task progress in memory so API can instantly query it."""
    with _task_cache_lock:
        if task_id not in _active_task_ids:
            _active_task_ids.append(task_id)
        _task_progress_cache[task_id] = {
            "task_id": task_id,
            "status": state,
            "progress": progress,
            "stage": stage,
            "meta": meta or {},
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }


def _run_batch_in_background_thread(task_id: str, file_paths: List[str], force: bool = False):
    """
    Fallback execution in a background thread if Celery worker is not currently running.
    Provides identical progress tracking (0% -> 100%) and updates the task cache.
    """
    total = len(file_paths)
    results = []
    total_chunks = 0
    docs_state = [
        {
            "filename": os.path.basename(fp),
            "status": "pending",
            "progress": 0,
            "stage": "Waiting in queue",
            "chunks": 0
        }
        for fp in file_paths
    ]

    update_local_task_cache(task_id, "PROGRESS", 0, f"Initializing batch of {total} documents", {
        "total": total,
        "processed": 0,
        "current_file": os.path.basename(file_paths[0]),
        "current_file_progress": 0,
        "current_file_stage": "Queued",
        "documents": docs_state
    })

    for idx, fp in enumerate(file_paths):
        filename = os.path.basename(fp)
        current_num = idx + 1
        docs_state[idx]["status"] = "processing"
        docs_state[idx]["stage"] = "Starting..."

        def make_doc_cb(item_idx, item_name):
            def cb(doc_pct: int, stage_desc: str):
                docs_state[item_idx]["progress"] = doc_pct
                docs_state[item_idx]["stage"] = stage_desc
                blended_pct = min(99, int(((item_idx + (doc_pct / 100.0)) / total) * 100))
                update_local_task_cache(task_id, "PROGRESS", blended_pct, f"[{item_idx + 1}/{total}] {stage_desc}", {
                    "total": total,
                    "processed": item_idx,
                    "current_file": item_name,
                    "current_file_progress": doc_pct,
                    "current_file_stage": stage_desc,
                    "documents": list(docs_state),
                    "results": results
                })
            return cb

        res = execute_document_ingestion(fp, force=force, progress_callback=make_doc_cb(idx, filename))
        results.append(res)
        chunks = res.get("chunks", 0)
        docs_state[idx]["status"] = res.get("status", "completed")
        docs_state[idx]["progress"] = 100
        docs_state[idx]["stage"] = "Completed" if res.get("status") == "success" else res.get("reason", "Finished")
        docs_state[idx]["chunks"] = chunks
        if res.get("status") == "success":
            total_chunks += chunks

        pct = int(((current_num) / total) * 100)
        update_local_task_cache(task_id, "PROGRESS", pct, f"Finished {current_num}/{total}: {filename}", {
            "total": total,
            "processed": current_num,
            "current_file": filename,
            "current_file_progress": 100,
            "current_file_stage": "Completed",
            "documents": list(docs_state),
            "results": results
        })

    update_local_task_cache(task_id, "SUCCESS", 100, f"Batch complete! Indexed {total} documents.", {
        "total": total,
        "processed": total,
        "total_chunks": total_chunks,
        "documents": docs_state,
        "results": results
    })


def dispatch_files_for_processing(file_paths: List[str], force: bool = False) -> Dict[str, Any]:
    """
    Dispatches a list of document filepaths into the Redis / Celery pipeline.
    Returns the generated Celery Task ID and metadata.
    """
    if not file_paths:
        return {"task_id": None, "status": "empty", "message": "No files to process"}

    task_id = str(uuid.uuid4())
    total_docs = len(file_paths)

    initial_docs = [
        {
            "filename": os.path.basename(f),
            "status": "pending",
            "progress": 0,
            "stage": "Queued in Celery/Redis",
            "chunks": 0
        }
        for f in file_paths
    ]

    update_local_task_cache(
        task_id=task_id,
        state="PENDING",
        progress=0,
        stage=f"Queued {total_docs} document(s) for Redis/Celery processing",
        meta={
            "total": total_docs,
            "files": [os.path.basename(f) for f in file_paths],
            "documents": initial_docs
        }
    )

    # Check if a dedicated Celery worker process is actively listening
    worker_available = False
    try:
        insp = celery_app.control.inspect(timeout=0.3)
        pings = insp.ping()
        worker_available = bool(pings)
    except Exception:
        worker_available = False

    if worker_available:
        try:
            process_batch_documents_task.apply_async(args=[file_paths, force], task_id=task_id)
            console.print(Panel(
                f"[bold green]✔ Dispatched {total_docs} document(s) to Active Celery Worker![/bold green]\n"
                f"[bold cyan]Celery Task ID:[/bold cyan] [bold white]{task_id}[/bold white]\n"
                f"[dim]Tracking via:[/dim] GET /tasks/{task_id}",
                title="[bold magenta]⚡ Celery Distributed Ingestion[/bold magenta]",
                border_style="cyan"
            ))
        except Exception as e:
            console.print(f"[yellow]Notice: Celery dispatch exception ({e}), falling back to background worker thread...[/yellow]")
            Thread(target=_run_batch_in_background_thread, args=(task_id, file_paths, force), daemon=True).start()
    else:
        # Also queue to Celery broker in Redis, and execute via background worker thread
        try:
            process_batch_documents_task.apply_async(args=[file_paths, force], task_id=task_id)
        except Exception:
            pass

        console.print(Panel(
            f"[bold green]✔ Queued {total_docs} document(s) with Celery Task ID![/bold green]\n"
            f"[bold cyan]Celery Task ID:[/bold cyan] [bold white]{task_id}[/bold white]\n"
            f"[dim]Tracking via:[/dim] GET /tasks/{task_id}",
            title="[bold magenta]⚡ Celery & Redis Task Initialized[/bold magenta]",
            border_style="cyan"
        ))
        Thread(target=_run_batch_in_background_thread, args=(task_id, file_paths, force), daemon=True).start()

    return {
        "task_id": task_id,
        "status": "QUEUED",
        "total_documents": total_docs,
        "files": [os.path.basename(f) for f in file_paths],
        "documents": initial_docs,
        "progress": 0
    }


def get_task_progress(task_id: str) -> Dict[str, Any]:
    """
    Retrieves real-time processing status and percentage from Celery (Redis)
    or local task cache.
    """
    # 1. Check Celery AsyncResult
    try:
        res = AsyncResult(task_id, app=celery_app)
        state = res.state

        if state == "PROGRESS":
            info = res.info or {}
            return {
                "task_id": task_id,
                "status": "PROGRESS",
                "progress": info.get("progress", 0),
                "stage": info.get("stage", "Processing..."),
                "processed": info.get("processed", 0),
                "total": info.get("total", 0),
                "current_file": info.get("current_file", ""),
                "current_file_progress": info.get("current_file_progress", 0),
                "current_file_stage": info.get("current_file_stage", ""),
                "documents": info.get("documents", []),
                "details": info.get("results", [])
            }
        elif state == "SUCCESS":
            info = res.result or {}
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "progress": 100,
                "stage": "All documents successfully processed!",
                "total": info.get("total", 0),
                "total_chunks": info.get("total_chunks", 0),
                "documents": info.get("documents", []),
                "details": info.get("results", [])
            }
        elif state == "FAILURE":
            return {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": 0,
                "stage": "Task failed",
                "error": str(res.result)
            }
    except Exception:
        pass

    # 2. Check local task cache fallback
    with _task_cache_lock:
        if task_id in _task_progress_cache:
            item = _task_progress_cache[task_id]
            meta = item.get("meta", {})
            return {
                "task_id": task_id,
                "status": item.get("status", "PENDING"),
                "progress": item.get("progress", 0),
                "stage": item.get("stage", "Pending..."),
                "processed": meta.get("processed", 0),
                "total": meta.get("total", 0),
                "current_file": meta.get("current_file", ""),
                "current_file_progress": meta.get("current_file_progress", 0),
                "current_file_stage": meta.get("current_file_stage", ""),
                "documents": meta.get("documents", []),
                "total_chunks": meta.get("total_chunks", 0),
                "details": meta.get("results", [])
            }

    return {
        "task_id": task_id,
        "status": "NOT_FOUND",
        "progress": 0,
        "stage": "Task ID not found"
    }


def get_all_active_tasks() -> List[Dict[str, Any]]:
    """Return all known task progress summaries."""
    tasks = []
    with _task_cache_lock:
        for tid in _active_task_ids[-20:]:  # Last 20 tasks
            tasks.append(get_task_progress(tid))
    return tasks


# ==============================================================================
# 8. ZIP ARCHIVE EXTRACTOR & BATCH UPLOAD
# ==============================================================================

def extract_and_process_zip(zip_filepath: str, extract_to: str = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """
    Extracts a uploaded ZIP file into the Data folder and dispatches all
    contained documents to the Redis Celery processing pipeline.
    """
    if not os.path.exists(zip_filepath):
        raise FileNotFoundError(f"Zip file '{zip_filepath}' not found.")

    os.makedirs(extract_to, exist_ok=True)
    extracted_doc_paths: List[str] = []

    console.print(f"[bold cyan]📦 Extracting ZIP archive:[/bold cyan] {zip_filepath}")

    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Skip folders and hidden files
            filename = os.path.basename(member)
            if not filename or filename.startswith(".") or filename == "desktop.ini":
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                # Prevent path traversal attacks by extracting to flat safe filename
                dest_path = os.path.join(extract_to, filename)
                with open(dest_path, "wb") as outfile:
                    outfile.write(zip_ref.read(member))
                extracted_doc_paths.append(dest_path)

    console.print(f"[bold green]✔ Extracted {len(extracted_doc_paths)} valid document(s) from ZIP archive.[/bold green]")

    # Dispatch extracted documents to Redis / Celery
    dispatch_result = dispatch_files_for_processing(extracted_doc_paths)
    dispatch_result["extracted_count"] = len(extracted_doc_paths)
    return dispatch_result


# ==============================================================================
# 9. DYNAMIC WATCHER & DIRECTORY SCANNER
# ==============================================================================

def check_and_process_new_documents(
    data_dir: str = DEFAULT_DATA_DIR,
    force: bool = False
) -> Dict[str, Any]:
    """
    Scans the data directory, identifies any brand new or modified documents,
    and dispatches them to Redis / Celery with a task ID.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return {"status": "empty", "message": "Created directory", "pending": 0}

    files_in_dir = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
        and not f.startswith(".")
        and f != "desktop.ini"
        and os.path.splitext(f)[1].lower() in (SUPPORTED_EXTENSIONS | {".zip"})
    ]

    registry = load_processed_registry()
    pending_files = []

    for filename in files_in_dir:
        filepath = os.path.join(data_dir, filename)

        # If it's a zip file dropped in Data folder, unpack it!
        if filename.lower().endswith(".zip"):
            try:
                zip_res = extract_and_process_zip(filepath, extract_to=data_dir)
                os.remove(filepath)  # Remove zip after unpacking
                return zip_res
            except Exception as e:
                console.print(f"[red]Error unpacking {filename}: {e}[/red]")
                continue

        if force or filename not in registry:
            pending_files.append(filepath)
        else:
            try:
                curr_hash = calculate_file_hash(filepath)
                if registry[filename].get("hash") != curr_hash:
                    pending_files.append(filepath)
            except Exception:
                pending_files.append(filepath)

    if not pending_files:
        console.print("[bold green]✔ All documents in Data/ are up to date in Qdrant.[/bold green]")
        return {"status": "up_to_date", "pending": 0, "message": "All documents are up-to-date"}

    console.print(f"\n[bold yellow]⚡ Found {len(pending_files)} new/modified document(s) in {data_dir}. Dispatching to Redis & Celery...[/bold yellow]")
    for fp in pending_files:
        console.print(f"  • [yellow]{os.path.basename(fp)}[/yellow]")

    return dispatch_files_for_processing(pending_files, force=force)


class DocumentUploadHandler(FileSystemEventHandler):
    """Watches Data/ folder for file additions/modifications and dispatches to Celery."""
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.last_events: Dict[str, float] = {}

    def _should_process(self, filepath: str) -> bool:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        if filename.startswith(".") or filename == "desktop.ini" or ext not in (SUPPORTED_EXTENSIONS | {".zip"}):
            return False

        now = time.time()
        last_time = self.last_events.get(filepath, 0)
        if (now - last_time) < 2.0:
            return False

        self.last_events[filepath] = now
        return True

    def on_created(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            console.print(f"\n[bold yellow]📥 New document detected in Data folder:[/bold yellow] {os.path.basename(event.src_path)}")
            if event.src_path.lower().endswith(".zip"):
                Thread(target=extract_and_process_zip, args=(event.src_path, self.data_dir), daemon=True).start()
            else:
                dispatch_files_for_processing([event.src_path])

    def on_modified(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            console.print(f"\n[bold yellow]📝 Modified document detected:[/bold yellow] {os.path.basename(event.src_path)}")
            dispatch_files_for_processing([event.src_path])


def _polling_loop(data_dir: str, interval: float = 4.0):
    """Periodic directory poller fallback."""
    global _stop_polling
    while not _stop_polling:
        try:
            check_and_process_new_documents(data_dir)
        except Exception as e:
            pass
        time.sleep(interval)


def start_dynamic_pipeline(data_dir: str = DEFAULT_DATA_DIR):
    """
    Starts the dynamic pipeline on app startup:
    1. Checks for new documents and dispatches them to Celery/Redis.
    2. Starts background watcher to detect new files/zips placed into Data folder.
    """
    global _active_observer, _polling_thread, _stop_polling

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Immediate scan & enqueue
    check_and_process_new_documents(data_dir)

    # Real-time watcher
    if WATCHDOG_AVAILABLE and Observer is not None:
        if _active_observer is not None and _active_observer.is_alive():
            return _active_observer

        console.print(f"[bold cyan]👀 Starting real-time file watcher (Watchdog) on directory:[/bold cyan] {data_dir}")
        event_handler = DocumentUploadHandler(data_dir)
        observer = Observer()
        observer.schedule(event_handler, path=data_dir, recursive=False)
        observer.daemon = True
        observer.start()

        _active_observer = observer
        console.print("[bold green]✔ Dynamic watcher is running with Redis Celery dispatch![/bold green]\n")
        return observer
    else:
        if _polling_thread is not None and _polling_thread.is_alive():
            return _polling_thread

        _stop_polling = False
        console.print(f"[bold cyan]👀 Starting background poller on:[/bold cyan] {data_dir}")
        _polling_thread = Thread(target=_polling_loop, args=(data_dir,), daemon=True)
        _polling_thread.start()
        return _polling_thread


def stop_dynamic_pipeline() -> None:
    """Stop directory watcher gracefully."""
    global _active_observer, _polling_thread, _stop_polling
    _stop_polling = True

    if _active_observer is not None and _active_observer.is_alive():
        _active_observer.stop()
        _active_observer.join(timeout=2.0)
        _active_observer = None

    if _polling_thread is not None and _polling_thread.is_alive():
        _polling_thread.join(timeout=2.0)
        _polling_thread = None
        console.print("[yellow]Dynamic directory watcher stopped.[/yellow]")


# ==============================================================================
# 10. CLI ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    console.print(Panel(
        "[bold cyan]ENTERPRISE REDIS & CELERY RAG PIPELINE[/bold cyan]\n"
        "• Scalable architecture designed for 1K - 1M documents\n"
        "• Redis queue & Celery distributed tasks with live percentage progress\n"
        "• Supports PDF, TXT, Excel, Word, CSV, and ZIP archives\n"
        "• Azure OpenAI embeddings & Qdrant vector storage",
        title="[bold magenta]🚀 Celery Dynamic Pipeline Ready[/bold magenta]",
        border_style="magenta"
    ))

    start_dynamic_pipeline(DEFAULT_DATA_DIR)
    console.print("[bold cyan]Press Ctrl+C to stop...[/bold cyan]")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_dynamic_pipeline()
        console.print("[green]Stopped.[/green]")
