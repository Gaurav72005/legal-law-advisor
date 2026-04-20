import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrievalchain import (
    load_vector_store,
    build_llm,
    init_db,
    rag_query
)

# -----------------------------------------------------------------
# INITIALIZE APP
# -----------------------------------------------------------------
app = FastAPI(
    title="Motor & Cyber Law Advisor API",
    description="Backend API for Legal RAG application",
    version="1.0.0"
)

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# GLOBALS FOR RESOURCES
# -----------------------------------------------------------------
DB = None
LLM = None
CONN = None
PROVIDER = None
RESOURCES_LOADED = False

@app.on_event("startup")
def startup_event():
    global DB, LLM, CONN, PROVIDER, RESOURCES_LOADED
    try:
        print("Loading legal database resources...")
        DB = load_vector_store()
        LLM, PROVIDER = build_llm()
        CONN = init_db()
        RESOURCES_LOADED = True
        print(f"Resources loaded successfully. Provider: {PROVIDER}")
    except Exception as e:
        print(f"Failed to load resources: {e}")
        RESOURCES_LOADED = False

@app.on_event("shutdown")
def shutdown_event():
    global CONN
    if CONN:
        CONN.close()
        print("Database connection closed.")

# -----------------------------------------------------------------
# DATA MODELS
# -----------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    chunks: list
    latency_ms: int
    answer_found: int
    provider: str

# -----------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not RESOURCES_LOADED:
        raise HTTPException(status_code=503, detail="Service Unavailable: Resources not loaded.")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        result = rag_query(DB, LLM, CONN, request.query)
        
        # Serialize chunks (Langchain Document to list of dicts)
        serialized_chunks = []
        for doc in result["chunks"]:
            serialized_chunks.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return {
            "question": result["question"],
            "answer": result["answer"],
            "chunks": serialized_chunks,
            "latency_ms": result["latency_ms"],
            "answer_found": result["answer_found"],
            "provider": PROVIDER or "unknown"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def status_endpoint():
    return {
        "status": "online" if RESOURCES_LOADED else "offline",
        "provider": PROVIDER
    }

# -----------------------------------------------------------------
# MOUNT FRONTEND
# -----------------------------------------------------------------
from fastapi.staticfiles import StaticFiles
frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"

if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    def index():
        return {"message": "Frontend build not found. Please build the React app into frontend/dist."}
