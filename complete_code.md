# Complete Project Codebase

This document contains the complete source code for the Qdrant RAG Chat API project. It is compiled here for easy reference and review.

---

## 1. `app.py`
The main FastAPI web server and application entry point.

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from routes import auth_router

# Import the rag_chain logic from search.py
from search import rag_chain

# Initialize FastAPI app
app = FastAPI(title="Qdrant RAG Chat API")

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
    # Start the server on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 2. `database.py`
Configures the SQLite database and SQLAlchemy Object-Relational Mapper (ORM) models.

```python
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# check_same_thread=False is needed only for SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token_hash = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime)
    revoked = Column(Boolean, default=False)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## 3. `security.py`
Handles password hashing and JSON Web Token (JWT) generation.

```python
from datetime import datetime, timedelta
import bcrypt
import jwt

SECRET_KEY = "your-secret-key-for-development-only"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def hash_password(password: str) -> str:
    # bcrypt requires bytes, so we encode the password
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_byte_enc = plain_password.encode('utf-8')
    hashed_password_byte_enc = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte_enc, hashed_password_byte_enc)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

import hashlib
import secrets

def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()

def generate_refresh_token() -> str:
    return secrets.token_urlsafe(32)
```

---

## 4. `document_loader.py`
Responsible for reading raw files (PDF, TXT, Excel) from the file system.

```python
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader
)
from rich.console import Console
from rich.panel import Panel

# Initialize the Rich console for colorful prints
console = Console()

def load_documents(data_dir: str = "Data"): 
    documents = []
    
    if not os.path.exists(data_dir): 
        console.print(f"[bold red]✖ Directory '{data_dir}' not found.[/bold red]")
        return documents

    console.print(f"\n[bold cyan]📂 Scanning directory:[/bold cyan] {data_dir}\n")

    for filename in os.listdir(data_dir): 
        filepath = os.path.join(data_dir, filename)
        
        # Skip if it's a directory or a hidden system file (like desktop.ini)
        if not os.path.isfile(filepath) or filename.startswith('.') or filename == "desktop.ini":
            continue
            
        console.print(f"[yellow]Processing:[/yellow] {filename} ...")
        
        try:
            if filename.lower().endswith(".pdf"):  
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                console.print(f"  [green]✔ Success! Loaded {len(docs)} pages.[/green]")
                
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                console.print(f"  [green]✔ Success! Loaded {len(docs)} document chunks.[/green]")
                
            elif filename.lower().endswith((".xlsx", ".xls")):
                loader = UnstructuredExcelLoader(filepath, mode="elements")
                docs = loader.load()
                documents.extend(docs)
                console.print(f"  [green]✔ Success! Loaded {len(docs)} excel elements.[/green]")
                
            else:
                console.print(f"  [dim]⚠ Skipped unsupported file type.[/dim]")
                
        except Exception as e:
            console.print(f"  [bold red]✖ Error loading {filename}:[/bold red] {str(e)}")
            
    return documents

if __name__ == "__main__":
    # Load all documents
    docs = load_documents()
    
    # Print a beautiful summary
    console.print("\n[bold magenta]📊 --- SUMMARY ---[/bold magenta]")
    console.print(f"Total documents (pages/elements) loaded: [bold white]{len(docs)}[/bold white]")
    
    # If we successfully loaded documents, show a pretty preview of the first one
    if docs:
        for idx, doc in enumerate(docs):
            # We don't truncate anymore, we print the FULL text
            full_text = doc.page_content.strip()
            
            console.print(f"\n[bold cyan]📄 Document #{idx + 1} (Full Text):[/bold cyan]")
            
            # Create a formatted panel box using Rich for EVERY page/document
            panel = Panel(
                full_text,
                title=f"[bold green]Source: {doc.metadata.get('source', 'Unknown')}[/bold green]",
                subtitle=f"[dim]Page/Element: {doc.metadata.get('page', 'N/A')}[/dim]",
                expand=False,
                border_style="cyan"
            )
            console.print(panel)
        console.print("\n[bold magenta]✔ Finished printing all data![/bold magenta]\n")
```

---

## 5. `splitter.py`
Breaks large documents into smaller overlapping text chunks for the embedding model.

```python
from document_loader import load_documents
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
# SemanticChunker requires an embedding model to evaluate the meaning of sentences
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

from rich.console import Console
from rich.panel import Panel

console = Console()

def run_splitters():
    # 1. Load the documents using our existing loader script
    console.print("\n[bold cyan]Step 1: Loading documents...[/bold cyan]")
    docs = load_documents("Data")
    # docs=documents found in the data_loader--8 docs
    
    if not docs:
        console.print("[bold red]No documents found to split![/bold red]")
        return
        
    console.print(f"[bold green]✔ Successfully loaded {len(docs)} documents/pages.[/bold green]\n")

    # ==========================================
    # DEFINING ALL AVAILABLE SPLITTERS
    # ==========================================
    
    # A. Fixed Character Text Splitter (splits strictly by a specific character, usually newline)
    fixed_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # B. Semantic Chunker (splits based on the semantic meaning/context using an AI embedding model)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        semantic_splitter = SemanticChunker(embeddings)
    except Exception as e:
        console.print(f"[yellow]Semantic splitter warning: {e}[/yellow]")
    
    # C. Recursive Character Text Splitter (The industry standard default)
    # It tries to split on paragraphs \n\n, then sentences \n, then spaces " "
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # Max size of each chunk
        chunk_overlap=100,    # Overlap between chunks so we don't cut contexts/sentences in half
        length_function=len,
        add_start_index=True, # Keeps track of where in the original doc this chunk came from
    )
    
    # ==========================================
    # PASS DOCUMENTS TO RECURSIVE SPLITTER
    # ==========================================
    console.print("[bold magenta]Step 2: Splitting documents using RecursiveCharacterTextSplitter...[/bold magenta]")
    
    # The split_documents method takes the list of LangChain Document objects and chunks them
    chunked_docs = recursive_splitter.split_documents(docs)
    print(chunked_docs)
    
    console.print(f"Original pages/elements count: [bold]{len(docs)}[/bold]")
    console.print(f"Total chunks created: [bold]{len(chunked_docs)}[/bold]\n")
    
    # Let's print a preview of up to 35 chunks to see how beautifully it split the text
    console.print("[bold cyan]Preview of up to 35 chunks:[/bold cyan]")
    for i in range(min(35, len(chunked_docs))):
        chunk = chunked_docs[i]
        
        # Format the metadata to be clearly visible
        meta_str = ", ".join([f"{k}: {v}" for k, v in chunk.metadata.items()])
        
        panel = Panel(
            chunk.page_content.strip(),
            title=f"[bold green]Chunk #{i + 1}[/bold green]",
            subtitle=f"[dim]Metadata: {meta_str}[/dim]",
            expand=False,
            border_style="cyan"
        )
        console.print(panel)
        
    return chunked_docs

if __name__ == "__main__":
    run_splitters()
```

---

## 6. `embeddings.py`
Generates vectors from the text chunks using Azure OpenAI and stores them in Qdrant.

```python
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rich.console import Console

# Import the chunked documents from our splitter script
from splitter import run_splitters

console = Console()

def generate_embeddings():
    # 1. Load environment variables from .env file
    load_dotenv()
    
    # Retrieve Azure OpenAI credentials
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Check for AZURE_OPENAI_EMBEDDING_DEPLOYMENT first, then AZURE_OPENAI_DEPLOYMENT
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    # Retrieve Qdrant credentials
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "my_documents")
    
    if not all([api_key, endpoint, api_version, deployment]):
        console.print("[bold red]✖ Missing Azure OpenAI credentials (or deployment name) in .env file![/bold red]")
        return
        
    if not all([qdrant_url, qdrant_api_key]):
        console.print("[bold red]✖ Missing Qdrant credentials (QDRANT_URL, QDRANT_API_KEY) in .env file![/bold red]")
        console.print("[yellow]Please update your .env file with QDRANT_URL and QDRANT_API_KEY.[/yellow]")
        return
        
    console.print("\n[bold cyan]Step 3: Initializing Azure OpenAI Embeddings...[/bold cyan]")
    
    # 2. Get the chunked documents from splitter.py
    chunks = run_splitters()  # 35 chunks
    
    if not chunks:
        console.print("[bold red]✖ No chunks received from splitter![/bold red]")
        return
        
    console.print(f"\n[bold green]✔ Received {len(chunks)} chunks for embedding.[/bold green]")
    
    # 3. Initialize Azure OpenAI Embeddings
    # NOTE: The deployment MUST be an embeddings model (like text-embedding-ada-002).
    # If AZURE_OPENAI_DEPLOYMENT points to an LLM like gpt-4o, this will fail.
    try:
        embeddings_model = AzureOpenAIEmbeddings(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_deployment=deployment 
        )
        
        # 4. Embed the first chunk as a test
        first_chunk_text = chunks[0].page_content
        console.print(f"[yellow]Generating embedding vector for Chunk #1 (length: {len(first_chunk_text)} chars)...[/yellow]")
        
        # embed_query converts the text into a mathematical vector
        vector = embeddings_model.embed_query(first_chunk_text)
        
        console.print(f"[bold green]✔ Successfully generated embedding![/bold green]")
        console.print(f"Vector Dimensions: [bold white]{len(vector)}[/bold white]")
        console.print(f"Preview of first 5 numbers: {vector[:5]}")
        
        console.print("\n[bold cyan]Step 5: Connecting to Qdrant Database...[/bold cyan]")
        
        # 5. Connect to Qdrant Cloud
        qdrant_client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key
        )
        
        # Check if collection exists, if not create it
        if not qdrant_client.collection_exists(collection_name=qdrant_collection_name):
            console.print(f"[yellow]Collection '{qdrant_collection_name}' not found. Creating it...[/yellow]")
            # We use len(vector) to automatically set the vector size based on the test embedding above
            qdrant_client.create_collection(
                collection_name=qdrant_collection_name,
                vectors_config=VectorParams(size=len(vector), distance=Distance.COSINE),
            )
            
        console.print(f"\n[bold cyan]Step 6: Storing {len(chunks)} chunks in Qdrant...[/bold cyan]")
        
        # 6. Store embeddings in Qdrant
        qdrant = QdrantVectorStore(
            client=qdrant_client,
            collection_name=qdrant_collection_name,
            embedding=embeddings_model,
        )
        
        # Add documents to Qdrant
        qdrant.add_documents(chunks)
        
        console.print(f"[bold green]✔ Successfully stored embeddings in Qdrant![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✖ Error generating embeddings:[/bold red] {e}")
        console.print("[dim yellow]Note: Ensure your AZURE_OPENAI_DEPLOYMENT in .env points to an *Embedding* model (e.g. text-embedding-ada-002), NOT an LLM (like gpt-4o-mini). If it's an LLM, OpenAI will throw a model not found / unsupported error.[/dim yellow]")

if __name__ == "__main__":
    generate_embeddings()
```

---

## 7. `search.py`
The RAG pipeline. Searches the vector database and passes context to the LLM to generate an answer.

```python
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# ==========================================
# 1. SETUP EMBEDDINGS & QDRANT CONNECTION
# ==========================================
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# The exact same embedding model used to store the vectors
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

embeddings_model = AzureOpenAIEmbeddings(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version,
    azure_deployment=embedding_deployment
)

# Connect to the exact same Qdrant Cloud cluster
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "Rag_project")

# Wrap the connection in the LangChain VectorStore
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=qdrant_collection_name,
    embedding=embeddings_model,
)

# Create a Retriever: This is the tool that embeds the question and runs the Cosine search.
# We tell it to return the top 3 closest chunks (k=3).
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ==========================================
# 2. SETUP THE CHAT AI (LLM)
# ==========================================
# This reads MODEL_NAME from your .env which you set to "gpt-5.4-mini-1" (or similar)
chat_deployment = os.getenv("MODEL_NAME")

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version,
    azure_deployment=chat_deployment,
    temperature=0.2 # Keep it low so the AI is factual and doesn't hallucinate
)

# Give the AI strict instructions to only use our database chunks to answer
prompt_template = """You are a helpful assistant. Answer the user's question based ONLY on the following context retrieved from the database.
If the context doesn't contain the answer, just say "I don't know based on the provided documents."

Context (Database Chunks):
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Helper function to join the 3 chunks together into one big string for the LLM to read
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the complete pipeline
# We export this so app.py can import and use it!
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 8. `check_qdrant.py`
A small utility script to debug and verify that the data (and vectors) actually exist in the Qdrant cloud database.

```python
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load the environment variables
load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "Rag_project")

print("Connecting to Qdrant...")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

print(f"Fetching 1 document from the '{qdrant_collection_name}' collection...")
# We use scroll to just grab the very first point in the database
results, _ = qdrant_client.scroll(
    collection_name=qdrant_collection_name,
    limit=1,
    with_vectors=True, # THIS TELLS QDRANT TO SEND THE VECTOR BACK
    with_payload=True
)

if results:
    point = results[0]
    print("\n--- WE FOUND THE POINT! ---")
    print(f"ID: {point.id}")
    
    print("\n--- HERE IS THE PAYLOAD (What you saw in the screenshot) ---")
    print(point.payload)
    
    print("\n--- HERE IS THE HIDDEN VECTOR! ---")
    print(f"Total numbers in the vector: {len(point.vector)}")
    print(f"First 10 numbers: {point.vector[:10]}")
    print("...")
else:
    print("No points found in the database.")
```
