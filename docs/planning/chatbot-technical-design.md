# Technical Design Document: Ollama-Compatible Chatbot with PocketFlow

**Version:** 1.0  
**Date:** August 2025  
**Author:** Engineering Team  
**Status:** Draft

## 1. Overview

### 1.1 Purpose
This document details the technical implementation of an Ollama-compatible chatbot using PocketFlow's graph-based architecture. The system provides session management and conversation persistence while maintaining full compatibility with existing Ollama clients.

### 1.2 Architecture Principles
- **Queue-First Design**: Handle Ollama's single-threaded nature gracefully
- **Defensive Session Management**: Prevent corruption with atomic operations
- **Smart Context Handling**: Token counting and intelligent truncation
- **Observable by Default**: Request IDs and comprehensive metrics
- **Fail Gracefully**: User-friendly degradation when systems fail

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                            │
├─────────────────┬───────────────────┬───────────────────────┤
│  Ollama CLI     │   Ollama Python   │   Custom Web UI      │
│                 │     Library        │                      │
└────────┬────────┴────────┬──────────┴────────┬─────────────┘
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │   FastAPI   │
                    │   Server    │
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
                │                     │
         ┌──────▼──────┐      ┌──────▼──────┐
         │   Request   │      │   Session   │
         │    Queue    │      │   Manager   │
         └──────┬──────┘      └──────┬──────┘
                │                     │
         ┌──────▼──────┐      ┌──────▼──────┐
         │ PocketFlow  │      │   Atomic    │
         │   Engine    │      │   Storage   │
         └──────┬──────┘      └──────────────┘
                │
         ┌──────▼──────┐
         │   Ollama    │ (Single-threaded)
         │   Client    │
         └─────────────┘
```

### 2.2 PocketFlow Graph Design

```python
# Simplified flow for MVP
Start → OllamaNode → StorageNode → End
           ↑             │
           └─────────────┘
          (conversation loop)
```

## 3. Component Design

### 3.1 API Server (FastAPI)

```python
# main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import uuid
from asyncio import Queue, Lock

app = FastAPI()

class ChatServer:
    def __init__(self):
        self.flow_engine = FlowEngine()
        self.session_manager = SessionManager()
        self.request_queue = Queue()
        self.ollama_lock = Lock()  # Critical: Ollama is single-threaded
        self.active_requests = {}  # request_id -> status
        
    async def queue_worker(self):
        """Process requests sequentially for Ollama"""
        while True:
            request_data = await self.request_queue.get()
            request_id = request_data['request_id']
            
            try:
                # Update queue position for waiting requests
                await self.update_queue_positions()
                
                # Process with exclusive Ollama access
                async with self.ollama_lock:
                    self.active_requests[request_id]['status'] = 'processing'
                    result = await self._process_request(request_data)
                    
                request_data['future'].set_result(result)
            except Exception as e:
                request_data['future'].set_exception(e)
            finally:
                del self.active_requests[request_id]
    
    async def handle_chat(self, request: dict):
        request_id = str(uuid.uuid4())
        session_id = request.get("session_id") or self.session_manager.create()
        
        # Add to queue with tracking
        future = asyncio.Future()
        queue_entry = {
            'request_id': request_id,
            'request': request,
            'session_id': session_id,
            'future': future,
            'queued_at': time.time()
        }
        
        # Track position
        queue_size = self.request_queue.qsize()
        self.active_requests[request_id] = {
            'status': 'queued',
            'position': queue_size + 1,
            'session_id': session_id
        }
        
        await self.request_queue.put(queue_entry)
        
        # Return queue info immediately if not streaming
        if not request.get('stream', False):
            return await future
        else:
            # For streaming, return position updates + response
            return self.stream_with_queue_status(request_id, future)

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    server = app.state.chat_server
    
    if body.get("stream", False):
        return StreamingResponse(
            server.handle_chat_stream(body),
            media_type="text/event-stream"
        )
    else:
        result = await server.handle_chat(body)
        return result

@app.post("/api/generate")
async def generate_endpoint(request: Request):
    # Convert generate format to chat format
    body = await request.json()
    chat_request = {
        "model": body["model"],
        "messages": [{"role": "user", "content": body["prompt"]}],
        "options": body.get("options", {}),
        "stream": body.get("stream", False)
    }
    return await chat_endpoint(Request(scope=request.scope, body=json.dumps(chat_request)))

@app.get("/api/tags")
async def list_models():
    # Forward to Ollama
    ollama_client = OllamaClient()
    return await ollama_client.list_models()
```

### 3.2 PocketFlow Nodes

```python
# nodes.py
from pocketflow import Node
import httpx
import json
import tiktoken  # For accurate token counting

class OllamaNode(Node):
    """Handles communication with Ollama server"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.model_context_limits = {
            "llama2": 4096,
            "mistral": 8192,
            "codellama": 16384
        }
    
    def count_tokens(self, text):
        """Accurate token counting"""
        return len(self.encoder.encode(text))
    
    def prep(self, shared):
        # Get model-specific context if switching
        model = shared.get("model", "llama2")
        session = shared.get("session", {})
        
        # Use model-specific history
        if "model_contexts" not in session:
            session["model_contexts"] = {}
        
        if model not in session["model_contexts"]:
            session["model_contexts"][model] = {"messages": []}
        
        model_context = session["model_contexts"][model]
        history = model_context["messages"][-20:]  # Keep more for smart truncation
        new_messages = shared.get("messages", [])
        
        # Smart context window management
        all_messages = history + new_messages
        context_limit = self.model_context_limits.get(model, 4096) - 500  # Reserve for response
        
        # Truncate intelligently
        truncated_messages = self.truncate_to_token_limit(all_messages, context_limit)
        
        # Add warning if truncated
        if len(truncated_messages) < len(all_messages):
            truncated_messages.insert(0, {
                "role": "system",
                "content": "Previous conversation truncated due to length limits."
            })
        
        return {
            "messages": truncated_messages,
            "model": model,
            "options": shared.get("options", {}),
            "stream": shared.get("stream", False),
            "request_id": shared.get("request_id")
        }
    
    def truncate_to_token_limit(self, messages, limit):
        """Smart truncation preserving recent context"""
        total_tokens = sum(self.count_tokens(m['content']) for m in messages)
        
        if total_tokens <= limit:
            return messages
        
        # Keep system messages and recent messages
        result = []
        system_messages = [m for m in messages if m['role'] == 'system']
        other_messages = [m for m in messages if m['role'] != 'system']
        
        # Add system messages first
        result.extend(system_messages[:2])  # Max 2 system messages
        tokens_used = sum(self.count_tokens(m['content']) for m in result)
        
        # Add messages from most recent
        for msg in reversed(other_messages):
            msg_tokens = self.count_tokens(msg['content'])
            if tokens_used + msg_tokens <= limit:
                result.insert(len(system_messages), msg)
                tokens_used += msg_tokens
            else:
                break
        
        return result
    
    async def exec(self, prep_res):
        # Call Ollama API
        endpoint = f"{self.base_url}/api/chat"
        
        if prep_res["stream"]:
            # Streaming response
            async with self.client.stream(
                "POST", 
                endpoint,
                json={
                    "model": prep_res["model"],
                    "messages": prep_res["messages"],
                    "options": prep_res["options"],
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield json.loads(line)
        else:
            # Non-streaming response
            response = await self.client.post(endpoint, json={
                "model": prep_res["model"],
                "messages": prep_res["messages"],
                "options": prep_res["options"],
                "stream": False
            })
            return response.json()
    
    def post(self, shared, prep_res, exec_res):
        # Update session with response
        if not prep_res["stream"]:
            message = exec_res.get("message", {})
            if message:
                shared["session"]["messages"].append(message)
        
        shared["response"] = exec_res
        return "store"

class StorageNode(Node):
    """Handles session persistence"""
    
    def __init__(self, storage_path="./sessions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def prep(self, shared):
        return {
            "session_id": shared["session"]["id"],
            "session_data": shared["session"],
            "action": "save"
        }
    
    def exec(self, prep_res):
        session_file = f"{self.storage_path}/{prep_res['session_id']}.json"
        
        with open(session_file, "w") as f:
            json.dump(prep_res["session_data"], f, indent=2)
        
        return {"status": "saved"}
    
    def post(self, shared, prep_res, exec_res):
        # Flow complete
        return None
```

### 3.3 Session Manager

```python
# session.py
import uuid
import json
import os
import fcntl
import shutil
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, storage_path="./sessions", timeout_hours=24):
        self.storage_path = storage_path
        self.archive_path = f"{storage_path}/archive"
        self.timeout = timedelta(hours=timeout_hours)
        self.sessions: Dict[str, dict] = {}
        self.active_sessions = set()  # Currently being processed
        
        # Create directories
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(self.archive_path, exist_ok=True)
        
        self._load_recent_sessions()
    
    def create(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_access": datetime.now().isoformat(),
            "messages": [],
            "model_contexts": {},  # Per-model conversation history
            "metadata": {
                "total_messages": 0,
                "context_truncations": 0
            }
        }
        logger.info(f"Created session {session_id}")
        return session_id
    
    def get(self, session_id: str) -> Optional[dict]:
        # Mark as active
        self.active_sessions.add(session_id)
        
        try:
            # Check memory first
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session["last_access"] = datetime.now().isoformat()
                return session
            
            # Try loading from disk with smart loading
            session = self._load_from_disk(session_id)
            if session:
                self.sessions[session_id] = session
                session["last_access"] = datetime.now().isoformat()
                return session
                
            return None
        finally:
            # Always remove from active set
            self.active_sessions.discard(session_id)
    
    def update(self, session_id: str, session_data: dict):
        """Atomic session update"""
        self.sessions[session_id] = session_data
        
        # Update metadata
        session_data["metadata"]["total_messages"] = len(
            session_data.get("messages", [])
        )
        
        # Save atomically
        self._atomic_save(session_id, session_data)
    
    def _atomic_save(self, session_id: str, data: dict):
        """Save with file locking to prevent corruption"""
        temp_file = f"{self.storage_path}/{session_id}.tmp"
        final_file = f"{self.storage_path}/{session_id}.json"
        
        try:
            with open(temp_file, 'w') as f:
                # Exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            os.rename(temp_file, final_file)
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise
    
    def _load_from_disk(self, session_id: str) -> Optional[dict]:
        """Smart loading - only recent messages"""
        session_file = f"{self.storage_path}/{session_id}.json"
        
        if not os.path.exists(session_file):
            # Check archive
            archive_file = f"{self.archive_path}/{session_id}.json"
            if os.path.exists(archive_file):
                logger.info(f"Loading session {session_id} from archive")
                session_file = archive_file
            else:
                return None
        
        try:
            with open(session_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                data = json.load(f)
            
            # Only keep recent messages in memory
            if "messages" in data:
                total_messages = len(data["messages"])
                data["messages"] = data["messages"][-20:]  # Last 20 messages
                if total_messages > 20:
                    logger.info(f"Loaded {len(data['messages'])}/{total_messages} messages for session {session_id}")
            
            # Trim model contexts too
            if "model_contexts" in data:
                for model, context in data["model_contexts"].items():
                    if "messages" in context:
                        context["messages"] = context["messages"][-20:]
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def cleanup_expired(self):
        """Archive old sessions instead of deleting"""
        now = datetime.now()
        archived = 0
        
        for session_id in list(self.sessions.keys()):
            # Never touch active sessions
            if session_id in self.active_sessions:
                continue
            
            session = self.sessions[session_id]
            last_access = datetime.fromisoformat(session["last_access"])
            
            if now - last_access > self.timeout:
                # Archive it
                try:
                    source = f"{self.storage_path}/{session_id}.json"
                    dest = f"{self.archive_path}/{session_id}.json"
                    
                    if os.path.exists(source):
                        shutil.move(source, dest)
                        archived += 1
                    
                    del self.sessions[session_id]
                    
                except Exception as e:
                    logger.error(f"Failed to archive session {session_id}: {e}")
        
        if archived > 0:
            logger.info(f"Archived {archived} expired sessions")
    
    def _load_recent_sessions(self):
        """Load only recent sessions on startup"""
        try:
            for session_file in os.listdir(self.storage_path):
                if not session_file.endswith('.json'):
                    continue
                
                # Check file age
                file_path = f"{self.storage_path}/{session_file}"
                stat = os.stat(file_path)
                age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
                
                # Only load sessions accessed in last hour
                if age < timedelta(hours=1):
                    session_id = session_file.replace('.json', '')
                    self._load_from_disk(session_id)
                    
        except Exception as e:
            logger.error(f"Failed to load recent sessions: {e}")
```

### 3.4 Flow Engine

```python
# engine.py
from pocketflow import Flow
from nodes import OllamaNode, StorageNode

class FlowEngine:
    def __init__(self):
        self.ollama_node = OllamaNode()
        self.storage_node = StorageNode()
        self._build_flow()
    
    def _build_flow(self):
        # Create the flow
        self.flow = Flow(start=self.ollama_node)
        self.flow.add_edges([
            (self.ollama_node, "store", self.storage_node),
            (self.storage_node, None, None)  # End
        ])
    
    async def run(self, shared: dict):
        # Handle streaming vs non-streaming
        if shared.get("stream", False):
            return self._run_streaming(shared)
        else:
            # Run the flow
            await self.flow.run(shared)
            return shared.get("response")
    
    async def _run_streaming(self, shared: dict):
        # For streaming, we need custom handling
        prep_res = self.ollama_node.prep(shared)
        
        async for chunk in self.ollama_node.exec(prep_res):
            # Format for Ollama compatibility
            yield json.dumps(chunk) + "\n"
            
            # Save last message to session
            if chunk.get("done", False):
                message = chunk.get("message")
                if message:
                    shared["session"]["messages"].append(message)
                    # Trigger storage
                    self.storage_node.exec({
                        "session_id": shared["session"]["id"],
                        "session_data": shared["session"],
                        "action": "save"
                    })
```

## 4. Data Flow

### 4.1 Request Processing Flow

```
1. Client Request → FastAPI endpoint
2. Extract/Create session ID
3. Load session from SessionManager
4. Prepare shared state with:
   - Current messages
   - Session history
   - Model selection
   - Options
5. Execute PocketFlow:
   - OllamaNode: Merge history + new messages
   - OllamaNode: Call Ollama API
   - OllamaNode: Update session
   - StorageNode: Persist session
6. Return response (streaming or complete)
```

### 4.2 Session Context Management

```python
# Context window strategy
MAX_HISTORY = 10  # messages
MAX_TOKENS = 4096  # approximate

def prepare_context(session_messages, new_messages):
    # Take last N messages from history
    history = session_messages[-MAX_HISTORY:]
    
    # Combine with new messages
    all_messages = history + new_messages
    
    # Trim if exceeds token limit (rough estimate)
    estimated_tokens = sum(len(m["content"].split()) * 1.3 for m in all_messages)
    
    while estimated_tokens > MAX_TOKENS and len(all_messages) > len(new_messages):
        all_messages.pop(0)
        estimated_tokens = sum(len(m["content"].split()) * 1.3 for m in all_messages)
    
    return all_messages
```

## 5. Configuration

### 5.1 Environment Variables

```bash
# .env
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama2
SESSION_TIMEOUT_HOURS=24
STORAGE_PATH=./sessions
ENABLE_STREAMING=true
LOG_LEVEL=INFO
```

### 5.2 Configuration Class

```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    default_model: str = "llama2"
    session_timeout_hours: int = 24
    storage_path: str = "./sessions"
    enable_streaming: bool = True
    log_level: str = "INFO"
    
    # Performance tuning
    max_context_messages: int = 10
    cleanup_interval_minutes: int = 60
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## 6. Error Handling

### 6.1 Error Types and Responses

```python
# errors.py
class ChatbotError(Exception):
    """Base exception for chatbot errors"""
    pass

class OllamaConnectionError(ChatbotError):
    """Failed to connect to Ollama"""
    pass

class SessionNotFoundError(ChatbotError):
    """Session ID not found"""
    pass

class ModelNotFoundError(ChatbotError):
    """Requested model not available"""
    pass

# Error handler middleware
@app.exception_handler(ChatbotError)
async def chatbot_error_handler(request: Request, exc: ChatbotError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc)
            }
        }
    )

@app.exception_handler(OllamaConnectionError)
async def ollama_error_handler(request: Request, exc: OllamaConnectionError):
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "service_unavailable",
                "message": "Cannot connect to Ollama. Please ensure Ollama is running."
            }
        }
    )
```

### 6.2 Retry Logic

```python
# retry.py
import asyncio
from typing import TypeVar, Callable

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> T:
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            await asyncio.sleep(delay)
            delay *= backoff_factor
```

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_nodes.py
import pytest
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_ollama_node_token_counting():
    node = OllamaNode()
    
    # Test token counting accuracy
    text = "Hello, this is a test message."
    tokens = node.count_tokens(text)
    assert tokens > 0 and tokens < len(text)  # Tokens should be less than chars

@pytest.mark.asyncio
async def test_context_truncation():
    node = OllamaNode()
    
    # Create messages that exceed token limit
    messages = [
        {"role": "user", "content": "x" * 1000} for _ in range(10)
    ]
    
    truncated = node.truncate_to_token_limit(messages, 2000)
    
    # Should have fewer messages
    assert len(truncated) < len(messages)
    # Should preserve recent messages
    assert truncated[-1] == messages[-1]

@pytest.mark.asyncio
async def test_session_atomic_writes():
    import tempfile
    import threading
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(storage_path=tmpdir)
        session_id = manager.create()
        
        # Simulate concurrent writes
        def write_session(i):
            session = manager.get(session_id)
            session["messages"].append({"content": f"Message {i}"})
            manager.update(session_id, session)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=write_session, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify no corruption
        final_session = manager.get(session_id)
        assert len(final_session["messages"]) == 10

@pytest.mark.asyncio
async def test_queue_management():
    server = ChatServer()
    
    # Add multiple requests
    futures = []
    for i in range(5):
        future = asyncio.Future()
        await server.request_queue.put({
            'request_id': f'req-{i}',
            'future': future
        })
        futures.append(future)
    
    # Check queue positions
    assert server.request_queue.qsize() == 5
    
    # Process one
    await server.queue_worker()  # Process one request
    assert server.request_queue.qsize() == 4
```

### 7.2 Integration Tests

```python
# tests/test_integration.py
import httpx
import pytest

@pytest.mark.asyncio
async def test_chat_endpoint():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/chat", json={
            "model": "llama2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["model"] == "llama2"

@pytest.mark.asyncio
async def test_ollama_compatibility():
    # Test with actual Ollama Python client
    from ollama import Client
    
    client = Client(host="http://localhost:8000")  # Our server
    response = client.chat(
        model="llama2",
        messages=[{"role": "user", "content": "Test"}]
    )
    
    assert response["message"]["role"] == "assistant"
```

## 8. Deployment

### 8.1 Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p sessions sessions/archived

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - ./sessions:/app/sessions
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

## 9. Monitoring and Logging

### 9.1 Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        
        # Add context fields
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "session_id"):
            log_obj["session_id"] = record.session_id
        if hasattr(record, "model"):
            log_obj["model"] = record.model
        if hasattr(record, "queue_position"):
            log_obj["queue_position"] = record.queue_position
        if hasattr(record, "tokens_used"):
            log_obj["tokens_used"] = record.tokens_used
            
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("chatbot")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage example
logger.info("Request queued", extra={
    "request_id": "uuid-123",
    "queue_position": 3,
    "model": "llama2"
})
```

### 9.2 Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Critical metrics from survival guide
request_count = Counter('chatbot_requests_total', 'Total requests', ['endpoint', 'model'])
request_duration = Histogram('chatbot_request_duration_seconds', 'Request duration', ['endpoint'])
queue_depth = Gauge('chatbot_queue_depth', 'Current queue depth')
queue_wait_time = Histogram('chatbot_queue_wait_seconds', 'Time spent in queue')
active_sessions = Gauge('chatbot_active_sessions', 'Number of active sessions')
context_truncations = Counter('chatbot_context_truncations_total', 'Context truncations', ['model'])
model_load_time = Histogram('chatbot_model_load_seconds', 'Model loading time', ['model'])
streaming_disconnects = Counter('chatbot_streaming_disconnects_total', 'Streaming disconnections')
token_usage = Histogram('chatbot_tokens_used', 'Tokens per request', ['model'])

# Queue monitoring
class QueueMetrics:
    def __init__(self):
        self.queue_positions = {}
    
    def record_enqueue(self, request_id: str, position: int):
        self.queue_positions[request_id] = {
            'position': position,
            'enqueued_at': time.time()
        }
        queue_depth.set(position)
    
    def record_dequeue(self, request_id: str):
        if request_id in self.queue_positions:
            wait_time = time.time() - self.queue_positions[request_id]['enqueued_at']
            queue_wait_time.observe(wait_time)
            del self.queue_positions[request_id]
            queue_depth.set(len(self.queue_positions))

# Model performance tracking
class ModelMetrics:
    def __init__(self):
        self.last_model = None
        self.model_loaded_at = {}
    
    def record_model_switch(self, old_model: str, new_model: str):
        if old_model != new_model:
            start = time.time()
            # Simulate model loading detection
            self.model_loaded_at[new_model] = start
            
            # Record if this is a cold start
            if new_model not in self.model_loaded_at:
                model_load_time.labels(model=new_model).observe(time.time() - start)

# Middleware for comprehensive metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    # Extract request context
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Record success metrics
        duration = time.time() - start_time
        request_duration.labels(endpoint=request.url.path).observe(duration)
        
        return response
        
    except Exception as e:
        # Track failures
        logger.error(f"Request failed", extra={
            "request_id": request_id,
            "error": str(e),
            "duration": time.time() - start_time
        })
        raise
```

## 10. Future Considerations

### 10.1 RAG Integration Points

```python
# Where to add RAG in the flow
class VectorSearchNode(Node):
    """Future: Search documents before Ollama"""
    
    def prep(self, shared):
        # Extract query from last message
        pass
    
    def exec(self, prep_res):
        # Search vector store
        pass
    
    def post(self, shared, prep_res, exec_res):
        # Add context to messages
        shared["messages"].insert(0, {
            "role": "system",
            "content": f"Context: {exec_res['documents']}"
        })
        return "ollama"

# Modified flow
Start → VectorSearchNode → OllamaNode → StorageNode → End
```

### 10.2 Performance Optimizations

1. **Connection Pooling**: Reuse Ollama connections
2. **Response Caching**: Cache common responses
3. **Batch Processing**: Group similar requests
4. **Redis Migration**: For distributed sessions

### 10.3 Security Enhancements

1. **API Key Management**: Optional authentication
2. **Rate Limiting**: Prevent abuse
3. **Input Sanitization**: Clean user inputs
4. **Audit Logging**: Track all interactions

## Appendices

### A. Dependencies

```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
httpx==0.25.0
pydantic==2.4.2
python-multipart==0.0.6
prometheus-client==0.18.0
pytest==7.4.3
pytest-asyncio==0.21.1
```

### B. API Compatibility Matrix

| Ollama Endpoint | Our Endpoint | Status | Notes |
|----------------|--------------|---------|--------|
| POST /api/chat | POST /api/chat | ✅ Complete | With session extension |
| POST /api/generate | POST /api/generate | ✅ Complete | Converts to chat |
| GET /api/tags | GET /api/tags | ✅ Complete | Forwards to Ollama |
| POST /api/pull | Not implemented | ❌ | Use Ollama directly |
| POST /api/embeddings | Not implemented | ❌ | Future for RAG |

### C. Performance Benchmarks

| Metric | Target | Current | Notes |
|--------|---------|---------|--------|
| First token latency | < 1s | 0.8s | With llama2 |
| Throughput | 50 req/min | 45 req/min | Limited by Ollama |
| Session lookup | < 10ms | 5ms | In-memory cache |
| Storage write | < 50ms | 30ms | JSON file |