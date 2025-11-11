"""
agent_fastapi.py
-------------------
FastAPI + LangGraph Content Creation Agent with Streaming Support

Features:
- REST API endpoints
- Real-time streaming responses
- Markdown formatted output
- CORS enabled for frontend integration
- Async processing

Run: uvicorn agent_fastapi:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
import re
import asyncio
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM

# -------------------------------
# FastAPI App Setup
# -------------------------------
app = FastAPI(
    title="AI Content Creation Agent",
    description="LangGraph-powered content generation with Ollama",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# LLM Setup
# -------------------------------
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.1,
    num_predict=512,
    num_ctx=2048,
    top_p=0.9,
    top_k=40
)

llm_writer = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.2,
    num_predict=1024,
    num_ctx=2048,
    top_p=0.9,
    top_k=40
)

# -------------------------------
# Pydantic Models (Request/Response)
# -------------------------------
class ContentRequest(BaseModel):
    project_brief: str = Field(..., description="Brief description of the content project")
    audience: str = Field(default="general", description="Target audience")
    tone: str = Field(default="informative", description="Content tone")
    keywords: List[str] = Field(default=[], description="Keywords to include")

class ContentResponse(BaseModel):
    status: str
    strategy: Optional[str] = None
    draft_content: Optional[str] = None
    quality_score: Optional[float] = None
    reviewer_comments: Optional[str] = None
    needs_escalation: bool = False
    final_content: Optional[str] = None
    timestamp: str

class StreamUpdate(BaseModel):
    stage: str
    content: str
    progress: int

# -------------------------------
# State Schema
# -------------------------------
class ContentState(TypedDict, total=False):
    project_brief: str
    audience: str
    tone: str
    keywords: List[str]
    strategy: str
    draft_content: str
    quality_score: float
    needs_escalation: bool
    reviewer_comments: str
    final_content: str

# -------------------------------
# Agent Nodes (Async versions)
# -------------------------------
async def content_strategist_async(state: ContentState) -> ContentState:
    """Generate content strategy"""
    brief = state.get("project_brief", "")
    audience = state.get("audience", "general")
    tone = state.get("tone", "informative")
    keywords = ", ".join(state.get("keywords", []))

    prompt = f"""Create a brief content strategy in markdown format:

**Brief:** {brief}
**Audience:** {audience}
**Tone:** {tone}
**Keywords:** {keywords}

Output format:
## Content Strategy
### Main Sections (3-5)
- Section 1
- Section 2

### Key Angles (3)
1. Angle 1
2. Angle 2

### Style Notes
- Note 1
- Note 2

Be concise."""

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(executor, llm.invoke, prompt)
    
    state["strategy"] = response.strip()
    return state


async def content_writer_async(state: ContentState) -> ContentState:
    """Write the actual content"""
    strategy = state.get("strategy", "")
    tone = state.get("tone", "informative")
    keywords = ", ".join(state.get("keywords", []))

    prompt = f"""Write a complete article in markdown format following this strategy:

{strategy}

**Tone:** {tone}
**Keywords:** {keywords}

Write the complete article with:
- Clear headings (##, ###)
- Proper paragraphs
- Bullet points where appropriate
- Bold/italic for emphasis

Start writing now:"""

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(executor, llm_writer.invoke, prompt)
    
    state["draft_content"] = response.strip()
    return state


async def content_reviewer_async(state: ContentState) -> ContentState:
    """Review content quality"""
    draft = state.get("draft_content", "")
    if not draft:
        state["quality_score"] = 0.0
        state["reviewer_comments"] = "No draft to review."
        state["needs_escalation"] = True
        return state

    prompt = f"""Review this draft. Output only JSON:

{draft[:1000]}...

{{"score": <0-10>, "comments": "<brief feedback>"}}"""

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        review_text = await loop.run_in_executor(executor, llm.invoke, prompt)

    try:
        match = re.search(r"\{[^{}]*\}", review_text)
        if match:
            data = json.loads(match.group(0))
        else:
            data = {"score": 7.0, "comments": "Review completed"}
    except Exception:
        data = {"score": 7.0, "comments": review_text[:200]}

    state["quality_score"] = float(data.get("score", 7.0))
    state["reviewer_comments"] = data.get("comments", "No comments")
    state["needs_escalation"] = state["quality_score"] < 6.0
    return state


async def finalizer_async(state: ContentState) -> ContentState:
    """Finalize content or escalate"""
    if state.get("needs_escalation"):
        state["final_content"] = ""
        return state

    state["final_content"] = state.get("draft_content", "")
    return state


# -------------------------------
# Pipeline Runner
# -------------------------------
async def run_content_pipeline_async(input_data: dict) -> dict:
    """Run the full content creation pipeline"""
    state = ContentState(**input_data)
    
    state = await content_strategist_async(state)
    state = await content_writer_async(state)
    state = await content_reviewer_async(state)
    state = await finalizer_async(state)
    
    return state


# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Content Creation Agent API",
        "status": "running",
        "model": LLM_MODEL,
        "endpoints": {
            "generate": "/api/generate",
            "stream": "/api/stream",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Check if Ollama is running"""
    try:
        test_response = llm.invoke("Hi")
        return {"status": "healthy", "model": LLM_MODEL, "ollama": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not available: {str(e)}")


@app.post("/api/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    """
    Generate content (non-streaming)
    Returns complete result when finished
    """
    try:
        input_data = {
            "project_brief": request.project_brief,
            "audience": request.audience,
            "tone": request.tone,
            "keywords": request.keywords
        }
        
        result = await run_content_pipeline_async(input_data)
        
        return ContentResponse(
            status="completed" if not result.get("needs_escalation") else "escalated",
            strategy=result.get("strategy"),
            draft_content=result.get("draft_content"),
            quality_score=result.get("quality_score"),
            reviewer_comments=result.get("reviewer_comments"),
            needs_escalation=result.get("needs_escalation", False),
            final_content=result.get("final_content"),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream")
async def stream_content(request: ContentRequest):
    """
    Stream content generation in real-time
    Returns Server-Sent Events (SSE)
    """
    async def generate_stream():
        try:
            input_data = {
                "project_brief": request.project_brief,
                "audience": request.audience,
                "tone": request.tone,
                "keywords": request.keywords
            }
            
            state = ContentState(**input_data)
            
            # Stage 1: Strategy
            yield f"data: {json.dumps({'stage': 'strategy', 'content': 'Generating strategy...', 'progress': 25})}\n\n"
            state = await content_strategist_async(state)
            yield f"data: {json.dumps({'stage': 'strategy', 'content': state.get('strategy'), 'progress': 25})}\n\n"
            
            # Stage 2: Writing
            yield f"data: {json.dumps({'stage': 'writing', 'content': 'Writing content...', 'progress': 50})}\n\n"
            state = await content_writer_async(state)
            yield f"data: {json.dumps({'stage': 'writing', 'content': state.get('draft_content'), 'progress': 50})}\n\n"
            
            # Stage 3: Review
            yield f"data: {json.dumps({'stage': 'review', 'content': 'Reviewing content...', 'progress': 75})}\n\n"
            state = await content_reviewer_async(state)
            review_data = {
                'score': state.get('quality_score'),
                'comments': state.get('reviewer_comments')
            }
            yield f"data: {json.dumps({'stage': 'review', 'content': json.dumps(review_data), 'progress': 75})}\n\n"
            
            # Stage 4: Finalize
            state = await finalizer_async(state)
            final_data = {
                'status': 'completed' if not state.get('needs_escalation') else 'escalated',
                'final_content': state.get('final_content'),
                'needs_escalation': state.get('needs_escalation')
            }
            yield f"data: {json.dumps({'stage': 'complete', 'content': json.dumps(final_data), 'progress': 100})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'content': str(e), 'progress': 0})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.get("/api/models")
async def list_models():
    """List available Ollama models"""
    return {
        "current_model": LLM_MODEL,
        "recommended_models": [
            "llama3",
            "llama3:8b-instruct-q4_K_M",
            "phi3:mini",
            "gemma:2b",
            "mistral:7b-instruct"
        ]
    }


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Content Creation Agent API...")
    print("📝 Docs available at: http://localhost:8000/docs")
    print("🔄 Stream endpoint: http://localhost:8000/api/stream")
    print("⚡ Generate endpoint: http://localhost:8000/api/generate")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)