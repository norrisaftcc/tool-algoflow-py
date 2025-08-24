# The Ollama Chatbot Survival Guide: A Time Traveler's Manual

*Found in the desk drawer of Jamie Chen, Senior Engineer*  
*Date on sticky note: "October 2025 - Read this FIRST!!"*

---

Hey there, past me (or new team member)! 

If you're reading this, you're about to embark on building our Ollama-powered chatbot with PocketFlow. I'm writing this from 2 months in the future, covered in pizza grease and caffeine stains, to save you from the face-palm moments we discovered the hard way.

Consider this your cheat sheet from future-you. You're welcome.

## 🚨 The "Oh No" Moments That Will Save Your Sanity

### 1. Ollama Doesn't Queue (The Single-Threaded Surprise)

**What we thought:** Ollama handles multiple requests like any normal API.

**Reality:** Ollama processes ONE request at a time. Period. If someone's generating War and Peace, everyone else waits.

```python
# What we tried first (DON'T DO THIS):
async def handle_concurrent_requests():
    # This will cause random timeouts and angry users
    tasks = [ollama_request(msg) for msg in messages]
    return await asyncio.gather(*tasks)  # 💥 BOOM

# What actually works:
from asyncio import Queue, Lock

ollama_lock = Lock()
request_queue = Queue()

async def ollama_worker():
    while True:
        request = await request_queue.get()
        async with ollama_lock:  # One at a time, folks
            result = await process_with_ollama(request)
        request['future'].set_result(result)
```

**Survival tip:** Show users their queue position. They're much happier knowing they're #3 in line vs. staring at a spinner.

### 2. The Streaming Memory Leak That Ate Production

**The trap:** Storing streaming chunks in memory "temporarily."

```python
# The road to OOM hell:
class BadStreamHandler:
    def __init__(self):
        self.chunks = []  # Seems innocent...
    
    async def handle_stream(self, stream):
        async for chunk in stream:
            self.chunks.append(chunk)  # Memory goes brrrrr
            yield chunk

# What you actually need:
class GoodStreamHandler:
    async def handle_stream(self, stream, session_id):
        buffer = []
        async for chunk in stream:
            buffer.append(chunk['message']['content'])
            
            # Save complete sentences, not every chunk
            if chunk['message']['content'].endswith(('.', '!', '?')):
                await save_to_session(session_id, ''.join(buffer))
                buffer = []
            
            yield chunk
```

**Why this matters:** Users love to ask "Write me a 10,000 word essay." Your server shouldn't die when they do.

### 3. Session Storage: The JSON Time Bomb

**What we did:** "JSON files are simple! What could go wrong?"

**What went wrong at 3 AM:** 
- Concurrent writes = corrupted sessions
- No cleanup = 50GB of chat history
- Loading full history = 5-second response times

```python
# The naive approach that seemed fine in dev:
def save_session(session_id, data):
    with open(f"{session_id}.json", "w") as f:
        json.dump(data, f)  # Race condition party!

# The battle-tested version:
import fcntl
from datetime import datetime, timedelta

class SessionStorage:
    def save_session(self, session_id, data):
        temp_file = f"{session_id}.tmp"
        
        # Atomic writes or bust
        with open(temp_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        
        os.rename(temp_file, f"{session_id}.json")
    
    def load_session_smart(self, session_id):
        # Don't load 10MB of history into memory
        with open(f"{session_id}.json", 'r') as f:
            data = json.load(f)
            # Only keep recent messages in memory
            data['messages'] = data['messages'][-20:]
            return data
```

### 4. The Context Window Cliff

**The surprise:** Ollama doesn't tell you when you exceed the context window. It just... forgets stuff.

```python
# What happens without protection:
# User: "My name is Alice"
# Bot: "Nice to meet you, Alice!"
# ... 50 messages later ...
# User: "What's my name?"
# Bot: "I don't have that information" 🤦

# The context guardian:
def prepare_context(messages, model="llama2"):
    # Rough token counting (1.3x word count)
    total_tokens = sum(len(m['content'].split()) * 1.3 for m in messages)
    
    # Llama2 = 4096 tokens, keep some headroom
    MAX_CONTEXT = 3500
    
    while total_tokens > MAX_CONTEXT and len(messages) > 2:
        # Remove old messages but keep system prompt
        if messages[1]['role'] != 'system':
            removed = messages.pop(1)
            total_tokens -= len(removed['content'].split()) * 1.3
    
    # Add context warning if we trimmed
    if total_tokens > MAX_CONTEXT:
        messages.insert(1, {
            'role': 'system',
            'content': 'Previous conversation truncated due to length.'
        })
    
    return messages
```

### 5. Model Switching Mid-Conversation (The Personality Disorder)

**What users do:** Start with llama2, switch to codellama, wonder why the bot forgot everything.

**The fix nobody tells you about:**

```python
# Store model-specific contexts
session_data = {
    "contexts": {
        "llama2": {"messages": [...], "personality": "helpful"},
        "codellama": {"messages": [...], "personality": "technical"},
        "mistral": {"messages": [...], "personality": "creative"}
    },
    "current_model": "llama2"
}

# Add model transition context
def switch_model(session, old_model, new_model):
    if old_model != new_model:
        transition_msg = {
            'role': 'system',
            'content': f'Switching from {old_model} to {new_model}. Previous context may be limited.'
        }
        session['contexts'][new_model]['messages'].append(transition_msg)
```

## 🎮 The Cheat Codes

### Auto-Retry with Exponential Backoff (Because Ollama Hiccups)

```python
async def ollama_call_with_retry(prompt, max_retries=3):
    delays = [1, 2, 4]  # seconds
    last_error = None
    
    for i in range(max_retries):
        try:
            return await ollama.generate(prompt)
        except httpx.ConnectError as e:
            last_error = e
            if i < max_retries - 1:
                await asyncio.sleep(delays[i])
                # Ollama might be reloading the model
                continue
    
    # Give user something useful
    return {
        "response": "I'm having connection issues. Please try again in a moment.",
        "error": str(last_error)
    }
```

### The "Model Not Loaded" Dance

```python
# Ollama unloads models after inactivity. First request = slow.
async def warm_up_model(model_name):
    """Call this on startup or after idle periods"""
    try:
        await ollama.generate(
            model=model_name,
            prompt="Hello",
            options={"num_predict": 1}  # Minimal generation
        )
    except:
        pass  # Model will load on next real request

# Pro tip: Schedule this every 5 minutes for popular models
```

### Session Cleanup That Actually Works

```python
# What we learned: Never delete active sessions, even if "expired"
class SmartSessionCleanup:
    def __init__(self):
        self.active_sessions = set()  # Currently processing
    
    async def cleanup(self):
        for session_file in glob.glob("sessions/*.json"):
            session_id = os.path.basename(session_file).replace('.json', '')
            
            # Skip active sessions
            if session_id in self.active_sessions:
                continue
            
            # Check age
            stat = os.stat(session_file)
            age = time.time() - stat.st_mtime
            
            if age > 86400:  # 24 hours
                # Archive, don't delete (users WILL ask for old convos)
                archive_path = f"sessions/archive/{session_id}.json"
                shutil.move(session_file, archive_path)
```

## 🔥 Performance Hacks We Wish We Knew

### 1. The Streaming Sweet Spot

```python
# Don't stream single tokens (network overhead kills you)
buffer = []
buffer_size = 0

async for chunk in ollama_stream:
    buffer.append(chunk)
    buffer_size += len(chunk.get('response', ''))
    
    # Send chunks of ~50 chars or at natural breaks
    if buffer_size > 50 or chunk.get('done', False):
        combined = ''.join(b['response'] for b in buffer)
        yield combined
        buffer = []
        buffer_size = 0
```

### 2. The "Typing Indicator" Trick

Users hate waiting without feedback. Add this before calling Ollama:

```python
async def chat_with_typing(websocket, prompt):
    # Send typing indicator immediately
    await websocket.send_json({"type": "typing", "status": "thinking"})
    
    # Now they know something's happening
    response = await ollama_generate(prompt)
    
    await websocket.send_json({"type": "message", "content": response})
```

### 3. Preload Common Prompts

```python
# Cache responses for FAQ-style questions
COMMON_PROMPTS = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I do for you?",
    "help": "I'm here to help! You can ask me questions...",
}

def maybe_use_cache(prompt):
    normalized = prompt.lower().strip()
    if normalized in COMMON_PROMPTS:
        return COMMON_PROMPTS[normalized]
    return None  # Hit Ollama
```

## 🚑 Emergency Procedures

### When Ollama Dies Mid-Conversation

```python
# The "Ollama CPR" procedure
async def health_check_with_recovery():
    try:
        await ollama.list_models()
        return True
    except:
        # Try to restart Ollama
        os.system("pkill ollama")
        await asyncio.sleep(2)
        os.system("ollama serve &")
        await asyncio.sleep(5)
        
        # Reload the last model used
        if last_used_model:
            await warm_up_model(last_used_model)
        
        return False
```

### The "Disk Full" Disaster

We learned this at 2 AM on a Saturday:

```python
# Add to your startup checks
def check_disk_space():
    stat = os.statvfs('/')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    
    if free_gb < 5:  # Less than 5GB
        # Start aggressive cleanup
        archive_old_sessions(days=1)  # Usually 7
        clear_temp_files()
        
        if free_gb < 1:
            # EMERGENCY MODE
            return "READ_ONLY"
    
    return "OK"
```

## 🎯 The "If I Could Start Over" List

1. **Use Redis from day 1.** JSON files are not your friend at scale.
2. **Build queue position indicators.** Users need to know why they're waiting.
3. **Log everything.** You'll need it when debugging "it worked yesterday."
4. **Add request IDs.** Tracing issues without them is hell.
5. **Implement graceful degradation.** When Ollama dies, at least show cached responses.
6. **Monitor model loading times.** Some models take 30+ seconds to load.
7. **Add a /metrics endpoint.** Prometheus + Grafana will save your life.
8. **Rate limit by session, not IP.** Corporate users share IPs.
9. **Version your sessions.** Migration scripts are easier than fixing corrupted data.
10. **Document model quirks.** Llama2 and Mistral format system prompts differently.

## 📊 Metrics That Actually Matter

Forget request count. Track these:

- **Queue depth**: How many requests are waiting?
- **Model load time**: First request after idle
- **Context truncation rate**: How often do you hit limits?
- **Session resurrection rate**: How often users continue old sessions?
- **Streaming disconnect rate**: Are users rage-quitting mid-response?

## 🎪 The Ollama Circus Acts

### Different Models, Different Personalities

```python
MODEL_QUIRKS = {
    "llama2": {
        "system_prompt": "You are a helpful assistant",  # Vanilla works
        "temperature": 0.7,  # Sweet spot
        "gotcha": "Tends to be verbose, add 'Be concise' to prompts"
    },
    "mistral": {
        "system_prompt": "[INST] You are a helpful assistant [/INST]",  # Needs tags
        "temperature": 0.8,
        "gotcha": "Sometimes ignores system prompts, repeat important instructions"
    },
    "codellama": {
        "system_prompt": "You are a coding assistant",
        "temperature": 0.3,  # Keep it focused
        "gotcha": "Will explain code even when not asked. Add 'no explanation needed' when appropriate"
    }
}
```

## 🏁 Final Wisdom from Future Me

1. **Users will break everything.** They'll paste entire books, use emoji as prompts, and switch languages mid-sentence.

2. **Ollama updates change behavior.** What works today might not tomorrow. Pin your versions.

3. **Sessions are sacred.** Users remember conversations from weeks ago. Never lose them.

4. **Streaming is an art.** Too granular = slow. Too chunky = appears frozen.

5. **Context is everything.** The difference between a smart bot and a goldfish is context management.

6. **Monitor, monitor, monitor.** You can't fix what you can't see.

7. **Users appreciate honesty.** "I'm a bit slow right now" beats mysterious timeouts.

Remember: We're not building a chatbot. We're building a conversation platform that happens to use Ollama. Design accordingly.

Good luck, past me. You're gonna need it! 🚀

P.S. - The coffee machine breaks next Thursday. Bring your own.

---

*Signed,*  
*Future Jamie (who has seen some stuff)*

*P.P.S - When you implement RAG in December, DON'T index the entire Wikipedia. Trust me on this one.*