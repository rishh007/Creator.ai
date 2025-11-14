# Creator.ai
Automated content creation pipeline with Strategist, Writer and Reviewer 


# AI Content Creation Agent - Setup Guide

## 🚀 Quick Start

### Prerequisites
1. **Python 3.8+** installed
2. **Ollama** installed and running ([ollama.ai](https://ollama.ai))
3. **Llama3 model** pulled in Ollama

### Step 1: Install Ollama & Model
```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull the model
ollama pull llama3

# Or use a faster model:
ollama pull phi3:mini
ollama pull gemma:2b
```

### Step 2: Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Start the Backend API
```bash
# Start the FastAPI server
python agent_fastapi.py

# Or using uvicorn directly:

```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Interactive Swagger UI)
- **ReDoc**: http://localhost:8000/redoc

### Step 4: Open the Frontend
```bash
# Option 1: Open the HTML file directly in browser
open frontend.html  # On Mac
start frontend.html  # On Windows
xdg-open frontend.html  # On Linux

# Option 2: Use a simple HTTP server
python -m http.server 3000
# Then visit: http://localhost:3000/frontend.html
```

---

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health
```
Check if Ollama is connected and running.

**Response:**
```json
{
  "status": "healthy",
  "model": "llama3",
  "ollama": "connected"
}
```

### 2. Generate Content (Non-Streaming)
```bash
POST /api/generate
```

**Request Body:**
```json
{
  "project_brief": "Write an article about AI agents",
  "audience": "developers",
  "tone": "technical",
  "keywords": ["AI", "automation", "LLM"]
}
```

**Response:**
```json
{
  "status": "completed",
  "strategy": "## Content Strategy...",
  "draft_content": "# AI Agents...",
  "quality_score": 8.5,
  "reviewer_comments": "Well structured",
  "needs_escalation": false,
  "final_content": "# AI Agents...",
  "timestamp": "2024-11-11T10:30:00"
}
```

### 3. Stream Content Generation
```bash
POST /api/stream
```

Returns Server-Sent Events (SSE) with real-time updates:

```javascript
const eventSource = new EventSource('/api/stream');
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.stage, data.content, data.progress);
};
```

---

## 🔧 Configuration

### Environment Variables
```bash
# Set custom model
export OLLAMA_MODEL=phi3:mini

# Or in Python
os.environ["OLLAMA_MODEL"] = "phi3:mini"
```

### Optimize for Speed
```python
# In agent_fastapi.py, modify LLM settings:
llm = OllamaLLM(
    model="gemma:2b",      # Fastest model
    temperature=0.1,       # Lower = faster
    num_predict=256,       # Fewer tokens
    num_ctx=1024,         # Smaller context
    num_gpu=1,            # Use GPU
)
```

---

## 🧪 Testing the API

### Using curl
```bash
# Test health endpoint
curl http://localhost:8000/health

# Generate content
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "project_brief": "Write about machine learning",
    "audience": "beginners",
    "tone": "casual",
    "keywords": ["ML", "AI", "learning"]
  }'
```

### Using Python requests
```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "project_brief": "Explain quantum computing",
        "audience": "general",
        "tone": "informative",
        "keywords": ["quantum", "computing", "qubits"]
    }
)

result = response.json()
print(result['final_content'])
```

---

## 🎨 Frontend Features

The included frontend (`frontend.html`) provides:

✅ **Real-time streaming** - See content generation live  
✅ **Markdown rendering** - Beautiful formatted output  
✅ **Progress tracking** - Visual pipeline stages  
✅ **Tabbed interface** - Strategy, Content, Review  
✅ **Keyword management** - Easy tag-based input  
✅ **Responsive design** - Works on all devices  

---

## 🚀 Performance Optimization Tips

### 1. Use Faster Models
```bash
# Fastest to slowest:
ollama pull gemma:2b          # ~5-10 tokens/sec
ollama pull phi3:mini         # ~8-12 tokens/sec
ollama pull llama3:8b-q4_K_M  # ~10-15 tokens/sec (quantized)
ollama pull llama3            # ~5-8 tokens/sec
```

### 2. GPU Acceleration
```bash
# Ensure Ollama is using GPU
OLLAMA_NUM_GPU=1 ollama serve
```

### 3. Adjust Token Limits
Lower `num_predict` for faster responses:
```python
llm = OllamaLLM(
    model="llama3",
    num_predict=256,  # Shorter responses
)
```

### 4. Enable Caching
Add simple caching for repeated queries:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate(brief: str):
    return llm.invoke(brief)
```

---

## 📦 Project Structure

```
.
├── agent_fastapi.py      # FastAPI backend with streaming
├── index.html         # Web UI for content generation
├── style.css         # css file 
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## 🐛 Troubleshooting

### Issue: "Ollama not available"
**Solution:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Verify model is downloaded
ollama pull llama3
```

### Issue: "CORS error in browser"
**Solution:** The API has CORS enabled. Ensure you're accessing the frontend via HTTP (not file://)
```bash
python -m http.server 3000
```

### Issue: "Slow generation"
**Solutions:**
1. Use a smaller/faster model (gemma:2b or phi3:mini)
2. Reduce `num_predict` parameter
3. Enable GPU acceleration
4. Lower `temperature` to 0.1

### Issue: "Connection refused"
**Solution:**
```bash
# Check if port 8000 is available
lsof -i :8000  # On Mac/Linux
netstat -ano | findstr :8000  # On Windows

# Use different port if needed
uvicorn agent_fastapi:app --port 8080
```

---

## 🔗 API Integration Examples

### React Integration
```jsx
const generateContent = async (brief) => {
  const response = await fetch('http://localhost:8000/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      project_brief: brief,
      audience: 'developers',
      tone: 'technical',
      keywords: ['AI', 'automation']
    })
  });
  
  const data = await response.json();
  return data.final_content;
};
```

### Node.js Integration
```javascript
const axios = require('axios');

async function generateContent() {
  const response = await axios.post('http://localhost:8000/api/generate', {
    project_brief: 'Write about AI',
    audience: 'general',
    tone: 'casual',
    keywords: ['AI', 'machine learning']
  });
  
  console.log(response.data.final_content);
}
```

---

## 📊 Expected Performance

| Model | Speed | Quality | Recommended For |
|-------|-------|---------|----------------|
| gemma:2b | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Quick drafts |
| phi3:mini | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | **Best balance** |
| llama3:8b-q4 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Quality content |
| llama3:8b | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best quality |

---

## 📝 License

This project is for educational purposes. Feel free to modify and use as needed.

---

## 🤝 Contributing

Improvements welcome! Areas to enhance:
- Add more agent nodes (SEO optimizer, fact checker)
- Implement vector database for research
- Add user authentication
- Save/load previous generations
- Export to various formats (PDF, DOCX)

---

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Ollama documentation: https://ollama.ai/docs
3. Check FastAPI docs: https://fastapi.tiangolo.com
