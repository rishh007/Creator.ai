"""
agent_fastapi.py
-------------------
FastAPI + LangGraph Content Creation Agent with Streaming Support
"""

import dotenv
dotenv.load_dotenv()  # very important.

import httpx
import base64
import os
import json
import re
import asyncio
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain_tavily import TavilySearch  # 1. IMPORT TAVILY
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request

# -------------------------------
# FastAPI App Setup
# -------------------------------
app = FastAPI(
    title="AI Content Creation Agent",
    description="LangGraph-powered content generation with Ollama and Tavily Search",
    version="1.1.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Environment Variables
# -------------------------------
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
BLOGGER_BLOG_ID = os.getenv("BLOGGER_BLOG_ID")

# -------------------------------
# LLM Setup
# -------------------------------
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
    num_ctx=4096,
    top_p=0.9,
    top_k=40
)

# -------------------------------
# Pydantic Models
# -------------------------------
class ContentRequest(BaseModel):
    project_brief: str = Field(..., description="Brief description of the content project")
    audience: str = Field(default="general", description="Target audience")
    tone: str = Field(default="informative", description="Content tone")
    keywords: List[str] = Field(default=[], description="Keywords to include")
    web_search: int = Field(default=0, description="Number of web search results to include (0 to disable)")

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

class BloggerPublishRequest(BaseModel):
    access_token: str
    title: str
    content: str

# -------------------------------
# State Schema
# -------------------------------
class ContentState(TypedDict, total=False):
    project_brief: str
    audience: str
    tone: str
    keywords: List[str]
    web_search: int
    search_results: Optional[Dict[str, Any]]
    strategy: str
    draft_content: str
    quality_score: float
    needs_escalation: bool
    reviewer_comments: str
    final_content: str

# -------------------------------
# Agent Nodes
# -------------------------------

async def web_search_async(state: ContentState) -> ContentState:
    """Search the web for the project brief if requested."""
    max_results = state.get("web_search", 0)

    if max_results == 0:
        state["search_results"] = None
        return state

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        search_tool = TavilySearch(max_results=max_results, include_images=True)
        try:
            results = await loop.run_in_executor(
                executor,
                search_tool.invoke,
                state.get("project_brief")
            )
            state["search_results"] = results
        except Exception as e:
            # On error, leave search_results as None but continue pipeline
            state["search_results"] = None
            # Optionally log or attach error info in state if desired
            state["search_error"] = str(e)

    return state


async def content_strategist_async(state: ContentState) -> ContentState:
    """Generate content strategy"""
    brief = state.get("project_brief", "")
    audience = state.get("audience", "general")
    tone = state.get("tone", "informative")
    keywords = ", ".join(state.get("keywords", []))

    search_results = state.get("search_results")
    search_context = "No web search results."
    if search_results and isinstance(search_results, dict):
        search_context = "## Web Search Context:\n"
        for i, result in enumerate(search_results.get("results", [])):
            search_context += f"Source {i+1} ({result.get('url')}):\n{result.get('content')}\n\n"

    prompt = f"""You are a content strategist. Use the following web search context to inform your strategy.

{search_context}

---
Create a brief content strategy in markdown format based on the context and the user's request:

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

    search_results = state.get("search_results")
    search_context = "No web search results."
    if search_results and isinstance(search_results, dict):
        search_context = "## Web Search Context (for your reference):\n"

        # Add Text results
        text_results = search_results.get("results", [])
        if text_results:
            for i, result in enumerate(text_results):
                search_context += f"Source {i+1} ({result.get('url')}):\n{result.get('content')}\n\n"

        # Add Image results
        image_results = search_results.get("images", [])
        if image_results:
            search_context += "## Available Images:\n"
            for img_url in image_results:
                search_context += f"- {img_url}\n"

    prompt = f"""You are an expert content writer. Your task is to write a complete article in markdown format.
You MUST follow the strategy and use the provided search context.

{strategy}

---
{search_context}
---

**CRITICAL INSTRUCTIONS:**
1. Write the complete article based on the strategy.
2. You MUST embed relevant images from the 'Available Images' list directly into the article where they make sense.
3. Use this exact Markdown format: ![A descriptive alt text](image_url)
4. Do not just list the image URLs at the end. They must be part of the article's flow.

**Tone:** {tone}
**Keywords:** {keywords}

Start writing the complete article now:"""

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

    # Add search as first step in pipeline
    state = await web_search_async(state)
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
        llm.invoke("Hi")
        return {"status": "healthy", "model": LLM_MODEL, "ollama": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not available: {str(e)}")


@app.post("/api/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    """Generate content (non-streaming)"""
    try:
        input_data = {
            "project_brief": request.project_brief,
            "audience": request.audience,
            "tone": request.tone,
            "keywords": request.keywords,
            "web_search": request.web_search
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
    """Stream content generation in real-time"""
    async def generate_stream():
        try:
            input_data = {
                "project_brief": request.project_brief,
                "audience": request.audience,
                "tone": request.tone,
                "keywords": request.keywords,
                "web_search": request.web_search
            }

            state = ContentState(**input_data)

            # Stage 0: Web search
            yield f"data: {json.dumps({'stage': 'search', 'content': 'Searching the web...', 'progress': 10})}\n\n"
            state = await web_search_async(state)
            yield f"data: {json.dumps({'stage': 'search', 'content': state.get('search_results'), 'progress': 10})}\n\n"

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
            "llama3.2",
            "llama3:8b-instruct-q4_K_M",
            "phi3:mini",
            "gemma:2b",
            "mistral:7b-instruct"
        ]
    }


@app.get("/api/download-image")
async def download_image(url: str, request: Request):
    """
    Proxies an image download to bypass CORS.
    """
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.stream("GET", url, headers={"User-Agent": request.headers.get("user-agent", "FastAPI-Proxy")})

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch image")

            content_type = response.headers.get("content-type", "application/octet-stream")
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="URL does not point to a valid image")

            filename = url.split('/')[-1].split('?')[0] or "image.jpg"

            return StreamingResponse(
                response.aiter_bytes(),
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename=\"{filename}\""}
            )

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image: {e}")


@app.get("/auth/google/login")
async def google_login():
    """Initiate Google OAuth flow"""
    scope = "https://www.googleapis.com/auth/blogger"

    url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={urllib.parse.quote(GOOGLE_REDIRECT_URI)}&"
        "response_type=code&"
        f"scope={urllib.parse.quote(scope)}&"
        "access_type=offline&"
        "prompt=consent"
    )

    return {"login_url": url}


@app.get("/auth/google/callback")
async def google_callback(request: Request):
    """Handle Google OAuth callback"""
    code = request.query_params.get("code")
    token_url = "https://oauth2.googleapis.com/token"

    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": GOOGLE_REDIRECT_URI,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(token_url, data=data)
        token_info = r.json()

    frontend_redirect = f"http://localhost:5500/index.html#token={token_info.get('access_token')}"
    return RedirectResponse(frontend_redirect)


@app.post("/api/publish/blogger")
async def publish_blogger_post(request: BloggerPublishRequest):
    """Publish a post to Blogger"""
    blog_id = BLOGGER_BLOG_ID
    
    if not blog_id:
        return {"status": "error", "message": "Blog ID not configured"}
    
    post_data = {
        "kind": "blogger#post",
        "blog": {"id": blog_id},
        "title": request.title,
        "content": request.content
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'https://www.googleapis.com/blogger/v3/blogs/{blog_id}/posts',
                headers={
                    'Authorization': f'Bearer {request.access_token}',
                    'Content-Type': 'application/json'
                },
                json=post_data
            )
        
        if response.status_code == 200:
            return {"status": "published", "result": response.json()}
        else:
            return {
                "status": "error", 
                "message": response.text, 
                "status_code": response.status_code
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
