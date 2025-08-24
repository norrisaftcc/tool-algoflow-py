# API Documentation: Ollama-Compatible Chatbot Server

**Version:** 1.0  
**Base URL:** `http://localhost:8000`  
**Compatibility:** Ollama API v1

## Overview

This chatbot server provides an Ollama-compatible API with additional session management capabilities. It acts as a drop-in replacement for Ollama while adding conversation persistence and context management.

**Important:** Ollama processes requests sequentially. This server adds queue management to handle multiple users gracefully.

### Key Features
- Full Ollama API compatibility
- Queue management with position feedback
- Smart context window management with token counting
- Atomic session persistence
- Model-specific conversation histories
- Request tracking with unique IDs

### Quick Start

```bash
# Using curl
curl http://localhost:8000/api/chat -d '{
  "model": "llama2",
  "messages": [{"role": "user", "content": "Hello!"}]
}'

# Using Ollama Python client
from ollama import Client
client = Client(host='http://localhost:8000')
response = client.chat(model='llama2', messages=[
  {'role': 'user', 'content': 'Hello!'}
])
```

## Authentication

The API supports optional authentication via API keys. For local development, authentication can be disabled.

```bash
# With authentication
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/api/chat -d '{...}'

# Without authentication (local development)
curl http://localhost:8000/api/chat -d '{...}'
```

## Endpoints

### Chat Completion

Generate a chat completion response with conversation context.

#### `POST /api/chat`

**Ollama Compatible** + Session Extensions

##### Request Body

```json
{
  "model": "llama2",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 128,
    "stop": ["Human:", "User:"]
  },
  "session_id": "optional-uuid-for-persistence",
  "request_id": "optional-request-tracking-id"
}
```

##### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | The model to use (e.g., "llama2", "mistral") |
| `messages` | array | Yes | - | Array of message objects with `role` and `content` |
| `stream` | boolean | No | false | Enable streaming response |
| `options` | object | No | {} | Model-specific parameters |
| `session_id` | string | No | auto-generated | Session ID for conversation persistence |
| `request_id` | string | No | auto-generated | Unique ID for request tracking |

##### Message Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | One of: "system", "user", "assistant" |
| `content` | string | Yes | The message content |

##### Options Object

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temperature` | float | 0.8 | Sampling temperature (0.0 - 1.0) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | integer | 40 | Top-k sampling parameter |
| `num_predict` | integer | -1 | Maximum tokens to generate (-1 = unlimited) |
| `stop` | array | [] | Stop sequences |
| `seed` | integer | -1 | Random seed for reproducibility |

##### Response (Non-Streaming)

```json
{
  "model": "llama2",
  "created_at": "2025-08-05T14:30:00.123Z",
  "message": {
    "role": "assistant",
    "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
  },
  "done": true,
  "total_duration": 1234567890,
  "load_duration": 123456789,
  "eval_count": 28,
  "eval_duration": 987654321,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "request_id": "req-123e4567-e89b-12d3-a456-426614174000",
  "queue_info": {
    "wait_time_ms": 0,
    "position": 0
  },
  "context_info": {
    "truncated": false,
    "messages_used": 3,
    "tokens_used": 145
  }
}
```

##### Response (Streaming)

```json
{"model":"llama2","created_at":"2025-08-05T14:30:00.123Z","queue_info":{"position":2,"estimated_wait_ms":3000},"done":false}
{"model":"llama2","created_at":"2025-08-05T14:30:03.123Z","queue_info":{"position":0},"done":false}
{"model":"llama2","created_at":"2025-08-05T14:30:03.234Z","message":{"role":"assistant","content":"Hello"},"done":false}
{"model":"llama2","created_at":"2025-08-05T14:30:03.345Z","message":{"role":"assistant","content":"!"},"done":false}
{"model":"llama2","created_at":"2025-08-05T14:30:03.456Z","message":{"role":"assistant","content":" I'm"},"done":false}
...
{"model":"llama2","created_at":"2025-08-05T14:30:04.567Z","message":{"role":"assistant","content":""},"done":true,"total_duration":1234567890,"eval_count":28,"session_id":"550e8400-e29b-41d4-a716-446655440000","context_info":{"truncated":false,"messages_used":3,"tokens_used":145}}
```

##### Example: Basic Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

##### Example: Streaming Response

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

##### Example: With Session

```bash
# First message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {"role": "user", "content": "My name is Alice"}
    ],
    "session_id": "my-session-123"
  }'

# Follow-up message (will remember the name)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {"role": "user", "content": "What is my name?"}
    ],
    "session_id": "my-session-123"
  }'
```

---

### Text Generation

Generate text completion from a prompt (Ollama compatible).

#### `POST /api/generate`

**Ollama Compatible**

##### Request Body

```json
{
  "model": "llama2",
  "prompt": "Once upon a time",
  "stream": false,
  "options": {
    "temperature": 0.8,
    "num_predict": 100
  }
}
```

##### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | The model to use |
| `prompt` | string | Yes | - | The prompt text |
| `stream` | boolean | No | false | Enable streaming response |
| `options` | object | No | {} | Model-specific parameters |

##### Response

Similar to chat endpoint but with `response` field instead of `message`.

```json
{
  "model": "llama2",
  "created_at": "2025-08-05T14:30:00.123Z",
  "response": "Once upon a time, in a land far away, there lived a wise old wizard...",
  "done": true,
  "context": [1, 2, 3, 4],
  "total_duration": 1234567890,
  "load_duration": 123456789,
  "prompt_eval_count": 4,
  "eval_count": 28,
  "eval_duration": 987654321
}
```

##### Example

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "The meaning of life is"
  }'
```

---

### List Models

Get a list of available models.

#### `GET /api/tags`

**Ollama Compatible**

##### Response

```json
{
  "models": [
    {
      "name": "llama2",
      "modified_at": "2025-08-01T10:00:00Z",
      "size": 3825819519,
      "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "name": "mistral",
      "modified_at": "2025-08-01T11:00:00Z",
      "size": 4109916160,
      "digest": "sha256:d4735e3a265426384690e56abf0640ff9785f962ed889a57b18c6ff6ab03f9f5"
    },
    {
      "name": "codellama",
      "modified_at": "2025-08-01T12:00:00Z",
      "size": 3791730816,
      "digest": "sha256:8c6fb4c6ff61e4d6f586b7f05bb9bff2e5950f8eddb560c269ee25c5d1c44d6e"
    }
  ]
}
```

##### Example

```bash
curl http://localhost:8000/api/tags
```

---

### Session Management

Extended endpoints for managing conversation sessions.

#### `GET /sessions/{session_id}/history`

**Extension** - Not part of standard Ollama API

Retrieve the complete conversation history for a session.

##### URL Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | The session ID |

##### Response

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-08-05T14:00:00Z",
  "last_activity": "2025-08-05T14:30:00Z",
  "message_count": 4,
  "messages": [
    {
      "role": "user",
      "content": "Hello!",
      "timestamp": "2025-08-05T14:00:00Z"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you today?",
      "timestamp": "2025-08-05T14:00:01Z"
    },
    {
      "role": "user",
      "content": "What's the weather like?",
      "timestamp": "2025-08-05T14:29:00Z"
    },
    {
      "role": "assistant",
      "content": "I don't have access to real-time weather data...",
      "timestamp": "2025-08-05T14:29:02Z"
    }
  ]
}
```

##### Example

```bash
curl http://localhost:8000/sessions/550e8400-e29b-41d4-a716-446655440000/history
```

---

### Health Check

Check if the service is running and Ollama connection is healthy.

#### `GET /health`

##### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "ollama": {
    "connected": true,
    "version": "0.1.48",
    "models_available": 3,
    "current_model_loaded": "llama2"
  },
  "queue": {
    "depth": 3,
    "average_wait_ms": 2500,
    "processing_rate": 0.4
  },
  "sessions": {
    "active": 42,
    "total": 128,
    "archived_today": 15
  },
  "metrics": {
    "context_truncation_rate": 0.05,
    "average_tokens_per_request": 487,
    "model_switches_per_hour": 12
  },
  "uptime": 3600
}
```

##### Example

```bash
curl http://localhost:8000/health
```

## Error Responses

All error responses follow a consistent format:

```json
{
  "error": {
    "type": "error_type",
    "message": "Human-readable error message",
    "details": {
      "additional": "context"
    }
  }
}
```

### Common Error Types

| Error Type | HTTP Status | Description |
|------------|-------------|-------------|
| `model_not_found` | 404 | Requested model is not available |
| `session_not_found` | 404 | Session ID does not exist |
| `invalid_request` | 400 | Request body is malformed |
| `ollama_connection_error` | 503 | Cannot connect to Ollama server |
| `queue_timeout` | 408 | Request timed out in queue |
| `context_limit_exceeded` | 413 | Message too large for context window |
| `rate_limit_exceeded` | 429 | Too many requests |
| `internal_error` | 500 | Unexpected server error |

### Error Examples

#### Model Not Found

```json
{
  "error": {
    "type": "model_not_found",
    "message": "Model 'gpt-4' is not available. Use /api/tags to list available models.",
    "details": {
      "requested_model": "gpt-4",
      "available_models": ["llama2", "mistral", "codellama"]
    }
  }
}
```

#### Ollama Connection Error

```json
{
  "error": {
    "type": "ollama_connection_error",
    "message": "Cannot connect to Ollama server. Please ensure Ollama is running on http://localhost:11434",
    "details": {
      "ollama_host": "http://localhost:11434",
      "error": "Connection refused"
    }
  }
}
```

## Client Libraries

### Python (Official Ollama Client)

```python
from ollama import Client
import asyncio

# Connect to our server instead of Ollama
client = Client(host='http://localhost:8000')

# Basic chat
response = client.chat(
    model='llama2',
    messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?'
        }
    ]
)
print(response['message']['content'])

# With session management
session_id = "user-123-session"
response = client.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'My name is Alice'}],
    session_id=session_id  # Our extension
)

# Streaming with queue feedback
async def stream_with_queue_info():
    stream = client.chat(
        model='llama2',
        messages=[{'role': 'user', 'content': 'Tell me a story'}],
        stream=True
    )
    
    for chunk in stream:
        if 'queue_info' in chunk:
            print(f"Queue position: {chunk['queue_info']['position']}")
        elif 'message' in chunk:
            print(chunk['message']['content'], end='', flush=True)

# Handle model switches
models = ['llama2', 'codellama', 'mistral']
for model in models:
    response = client.chat(
        model=model,
        messages=[{'role': 'user', 'content': 'Hello'}],
        session_id=session_id  # Each model maintains separate context
    )
```

### Python (Requests)

```python
import requests
import json

# Chat with session
response = requests.post(
    'http://localhost:8000/api/chat',
    json={
        'model': 'llama2',
        'messages': [
            {'role': 'user', 'content': 'Hello!'}
        ],
        'session_id': 'my-session-123'
    }
)

data = response.json()
print(data['message']['content'])

# Streaming with requests
response = requests.post(
    'http://localhost:8000/api/chat',
    json={
        'model': 'llama2',
        'messages': [{'role': 'user', 'content': 'Write a poem'}],
        'stream': True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        if not chunk['done']:
            print(chunk['message']['content'], end='', flush=True)
```

### JavaScript/Node.js

```javascript
// Using fetch
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'llama2',
    messages: [
      { role: 'user', content: 'What is JavaScript?' }
    ]
  })
});

const data = await response.json();
console.log(data.message.content);

// Streaming
const streamResponse = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'llama2',
    messages: [{ role: 'user', content: 'Count to 10' }],
    stream: true
  })
});

const reader = streamResponse.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n').filter(line => line.trim());
  
  for (const line of lines) {
    const data = JSON.parse(line);
    if (!data.done) {
      process.stdout.write(data.message.content);
    }
  }
}
```

### cURL Examples

```bash
# Basic chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# With custom parameters
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Be creative"}],
    "options": {
      "temperature": 0.9,
      "top_p": 0.95
    }
  }'

# Streaming to terminal
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }' \
  --no-buffer

# With session
SESSION_ID="my-session-123"
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"llama2\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Remember this number: 42\"}],
    \"session_id\": \"$SESSION_ID\"
  }"
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default limit**: 60 requests per minute per IP
- **Authenticated limit**: 120 requests per minute per API key
- **Streaming requests**: Count as 1 request regardless of duration

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1691245260
```

## Best Practices

### 1. Queue Management

```python
# Good: Handle queue feedback
async def chat_with_queue_feedback(client, messages):
    response = await client.chat(
        model='llama2',
        messages=messages,
        stream=True
    )
    
    for chunk in response:
        if 'queue_info' in chunk and chunk['queue_info']['position'] > 0:
            print(f"Waiting... Position {chunk['queue_info']['position']} in queue")
        elif 'message' in chunk:
            # Now we're processing
            yield chunk['message']['content']

# Bad: No feedback during wait
response = client.chat(model='llama2', messages=messages)
# User stares at nothing...
```

### 2. Context Window Management

```python
# Good: Monitor context usage
response = client.chat(
    model='llama2',
    messages=long_conversation,
    session_id=session_id
)

if response.get('context_info', {}).get('truncated'):
    print("⚠️ Some messages were removed to fit context window")
    print(f"Using {response['context_info']['tokens_used']} tokens")

# Bad: Blindly sending huge conversations
messages = load_entire_chat_history()  # 1000 messages
response = client.chat(model='llama2', messages=messages)
# Most messages silently dropped!
```

### 3. Model-Specific Sessions

```python
# Good: Let server manage per-model context
session_id = "user-123"
for task in tasks:
    model = "codellama" if task['type'] == 'code' else "llama2"
    response = client.chat(
        model=model,
        messages=[{'role': 'user', 'content': task['content']}],
        session_id=session_id  # Server tracks context per model
    )

# Bad: Mixing models without context awareness
messages = []
messages.append(chat_with_llama2())
messages.append(chat_with_codellama())  # Context confusion!
```

### 4. Error Handling with Exponential Backoff

```python
import time
import random

async def robust_chat(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.chat(
                model='llama2',
                messages=messages,
                request_id=f"req-{uuid.uuid4()}"  # Track for debugging
            )
        except Exception as e:
            if 'queue_timeout' in str(e):
                # Don't retry queue timeouts
                raise
            
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retry {attempt + 1} after {wait:.1f}s: {e}")
                time.sleep(wait)
            else:
                raise
```

### 5. Efficient Streaming

```python
# Good: Process chunks in batches
buffer = []
async for chunk in stream:
    if 'message' in chunk:
        buffer.append(chunk['message']['content'])
        
        # Flush on sentence boundaries
        if any(buffer[-1].endswith(p) for p in ['.', '!', '?', '\n']):
            yield ''.join(buffer)
            buffer = []

# Bad: Yielding every token
async for chunk in stream:
    yield chunk['message']['content']  # Network overhead!
``` they arrive
for chunk in client.chat(model='llama2', messages=messages, stream=True):
    process_chunk(chunk)  # Don't accumulate in memory

# Bad: Collecting all chunks before processing
chunks = []
for chunk in client.chat(model='llama2', messages=messages, stream=True):
    chunks.append(chunk)
# This defeats the purpose of streaming
```

### 4. Context Management

```python
# Good: Let the server manage context
response = client.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Continue our discussion'}],
    session_id='existing-session'
)

# Bad: Sending entire history every time
all_messages = load_entire_conversation()  # Could be hundreds of messages
response = client.chat(
    model='llama2',
    messages=all_messages + [{'role': 'user', 'content': 'New message'}]
)
```

## Troubleshooting

### Common Issues

#### 1. "Cannot connect to Ollama server"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve

# Verify model is loaded (first request is slow)
time curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "test",
  "options": {"num_predict": 1}
}'
```

#### 2. "Queue timeout" or long waits

```bash
# Check queue depth
curl http://localhost:8000/health | jq '.queue.depth'

# Common causes:
# - Someone generating long responses
# - Model switching (causes reload)
# - First request after idle (model not in memory)

# Solution: Implement model warm-up
curl -X POST http://localhost:8000/api/generate -d '{
  "model": "llama2",
  "prompt": "warm up",
  "options": {"num_predict": 1}
}'
```

#### 3. "Context limit exceeded" warnings

```python
# Check your token usage
response = client.chat(model='llama2', messages=messages)
print(f"Tokens used: {response['context_info']['tokens_used']}")
print(f"Messages kept: {response['context_info']['messages_used']}")

# Fix: Let the server manage context
# Don't send entire history - use session_id instead
```

#### 4. Session not persisting

```bash
# Verify session exists
curl http://localhost:8000/sessions/YOUR_SESSION_ID/history

# Check for file permissions
ls -la ./sessions/

# Common issues:
# - Session ID typos (they're case-sensitive)
# - Sessions archived after 24 hours
# - Disk full (check df -h)
```

#### 5. Streaming disconnections

```python
# Add connection monitoring
import asyncio

async def stream_with_heartbeat(client, messages):
    last_chunk_time = time.time()
    
    async for chunk in client.chat(model='llama2', messages=messages, stream=True):
        current_time = time.time()
        
        # Detect stalls
        if current_time - last_chunk_time > 30:
            print("⚠️ Connection may be stalled")
        
        last_chunk_time = current_time
        yield chunk

# Set appropriate timeouts
client = Client(
    host='http://localhost:8000',
    timeout=300  # 5 minutes for long generations
)
```

#### 6. Model personality changes

```bash
# Different models = different contexts
# Check which model you're using
curl http://localhost:8000/health | jq '.ollama.current_model_loaded'

# Each model maintains separate conversation history
# Switching models mid-conversation can be confusing
```

### Performance Tuning

#### Reduce Queue Wait Times

```python
# 1. Use smaller models when possible
models = {
    'quick': 'llama2:7b',    # Faster
    'quality': 'llama2:70b'  # Slower but better
}

# 2. Limit generation length
response = client.chat(
    model='llama2',
    messages=messages,
    options={'num_predict': 150}  # Limit tokens
)

# 3. Implement client-side queueing
class SmartClient:
    def __init__(self):
        self.pending = asyncio.Queue()
        
    async def chat_with_local_queue(self, messages):
        # Show local queue position immediately
        position = self.pending.qsize() + 1
        print(f"Local queue position: {position}")
        
        await self.pending.put(messages)
        # Process when server is ready
```

#### Monitor Critical Metrics

```bash
# Set up monitoring endpoints
curl http://localhost:8000/metrics | grep -E '(queue_depth|context_truncations|model_load_time)'

# Key metrics to watch:
# - queue_depth: Should average < 5
# - context_truncations_total: Should be < 5% of requests
# - model_load_time: First request after idle
# - queue_wait_seconds: User experience metric
```

### Advanced Debugging

#### Enable Request Tracking

```python
# Always include request IDs
import uuid

request_id = str(uuid.uuid4())
print(f"Tracking request: {request_id}")

response = client.chat(
    model='llama2',
    messages=messages,
    request_id=request_id
)

# Check logs for this ID if issues occur
```

#### Debug Context Issues

```python
# Test context window handling
test_messages = []
for i in range(100):
    test_messages.append({
        'role': 'user',
        'content': f'Message {i}: ' + 'x' * 100
    })
    
    response = client.chat(
        model='llama2',
        messages=test_messages,
        session_id='test-context'
    )
    
    info = response.get('context_info', {})
    if info.get('truncated'):
        print(f"Truncation started at message {i}")
        print(f"Tokens: {info['tokens_used']}")
        break
```

#### Session Diagnostics

```python
# Session health check script
def diagnose_session(session_id):
    # 1. Check if session exists
    try:
        history = client.get_session_history(session_id)
        print(f"✓ Session found: {len(history['messages'])} messages")
    except:
        print("✗ Session not found")
        return
    
    # 2. Check last activity
    last_activity = datetime.fromisoformat(history['last_activity'])
    age = datetime.now() - last_activity
    print(f"✓ Last active: {age.total_seconds() / 3600:.1f} hours ago")
    
    # 3. Check for corruption
    for i, msg in enumerate(history['messages']):
        if 'role' not in msg or 'content' not in msg:
            print(f"✗ Corrupted message at index {i}")
    
    # 4. Check model contexts
    if 'model_contexts' in history:
        for model, context in history['model_contexts'].items():
            print(f"✓ {model}: {len(context.get('messages', []))} messages")
```

## Migration Guide

### From Direct Ollama

```python
# Before (direct Ollama)
import ollama
response = ollama.chat(model='llama2', messages=messages)

# After (with our server)
from ollama import Client
client = Client(host='http://localhost:8000')
response = client.chat(
    model='llama2',
    messages=messages,
    session_id='user-session'  # Add this for persistence
)

# Everything else works the same!
```

### Handling New Features

```python
# 1. Queue feedback
if 'queue_info' in response:
    print(f"Queue position: {response['queue_info']['position']}")

# 2. Context information  
if 'context_info' in response:
    if response['context_info']['truncated']:
        print("Some context was truncated")

# 3. Request tracking
response = client.chat(
    model='llama2',
    messages=messages,
    request_id='my-request-123'  # For debugging
)
```

## Changelog

### Version 1.0.0 (August 2025)
- Initial release
- Queue management for Ollama's single-threading
- Smart context window management
- Atomic session persistence
- Per-model conversation contexts
- Request ID tracking
- Comprehensive metrics

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [github.com/yourorg/chatbot-server](https://github.com/yourorg/chatbot-server)
- **Documentation**: [docs.yourorg.com](https://docs.yourorg.com)
- **Community Discord**: [discord.gg/yourorg](https://discord.gg/yourorg)

### Quick Links

- [Ollama Documentation](https://ollama.ai/docs)
- [Survival Guide](./docs/survival-guide.md)
- [Metrics Dashboard](http://localhost:8000/metrics)
- [Health Check](http://localhost:8000/health)