from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from routes.auth import router as auth_router

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
