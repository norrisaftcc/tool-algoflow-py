# Product Requirements Document: AI Chatbot MVP

**Version:** 1.1  
**Date:** August 2025  
**Status:** Draft  
**Product Owner:** [To be assigned]  
**Technical Lead:** [To be assigned]

## Executive Summary

This PRD defines the requirements for building a Minimum Viable Product (MVP) chatbot using the PocketFlow framework with Ollama as the LLM provider. The MVP implements Pattern 1 (Basic Chatbot with Memory) while preparing for Pattern 2 (RAG) in the next phase. 

**Critical Constraint:** Ollama processes requests sequentially (single-threaded), requiring queue management and clear user feedback about wait times.

## Table of Contents

1. [Product Overview](#product-overview)
2. [Goals and Objectives](#goals-and-objectives)
3. [User Stories](#user-stories)
4. [Functional Requirements](#functional-requirements)
5. [Non-Functional Requirements](#non-functional-requirements)
6. [Technical Architecture](#technical-architecture)
7. [API Specification](#api-specification)
8. [Data Models](#data-models)
9. [Future Design: RAG Integration](#future-design-rag-integration)
10. [Success Metrics](#success-metrics)
11. [Timeline and Milestones](#timeline-and-milestones)
12. [Risks and Mitigations](#risks-and-mitigations)

## Product Overview

### Problem Statement
Organizations need a flexible, scalable conversational AI system that can maintain context across interactions while being extensible for future knowledge-base integration. Current solutions are either too complex for initial deployment or lack the architectural flexibility for growth.

### Solution
A graph-based chatbot built on PocketFlow that provides:
- Queue management for Ollama's single-threaded processing
- Atomic session persistence with race condition prevention
- Smart context window management with token counting
- Request tracking with unique IDs for debugging
- Clear user feedback about processing status

### Target Users
- **Primary:** Development teams needing to embed conversational AI
- **Secondary:** Customer service departments requiring automated first-line support
- **Future:** Knowledge workers needing document-aware assistants

## Goals and Objectives

### MVP Goals (Phase 1)
1. **Functional Conversations:** Support multi-turn dialogues with context retention
2. **Session Persistence:** Maintain conversation history across user sessions
3. **API-First Design:** Provide simple integration points for existing systems
4. **Production Ready:** Handle concurrent users with acceptable performance

### Future Goals (Phase 2)
1. **Knowledge Integration:** Add RAG capabilities for document-aware responses
2. **Multi-Agent Support:** Enable specialized agent coordination
3. **Advanced Patterns:** Implement remaining patterns from the architecture

### Non-Goals for MVP
- Multi-language support (English only)
- Voice/audio interfaces
- Complex authentication (API key optional for local deployment)
- Analytics dashboard
- RAG implementation (design only)

## User Stories

### Core User Stories (MVP)

**US-1: Basic Conversation**
> As a user, I want to have a natural conversation with the chatbot so that I can get help with my questions.

**Acceptance Criteria:**
- User can send messages and receive contextually appropriate responses
- Responses arrive within 3 seconds
- Conversation feels natural and coherent

**US-2: Context Retention**
> As a user, I want the chatbot to remember our conversation context so that I don't have to repeat information.

**Acceptance Criteria:**
- Chatbot references previous messages appropriately
- Context window maintains last 10 messages minimum
- User can refer to "it", "that", etc. and bot understands

**US-3: Session Continuity**
> As a returning user, I want to continue previous conversations so that I can pick up where I left off.

**Acceptance Criteria:**
- Sessions persist for at least 24 hours
- User can retrieve conversation history
- Clear indication when starting new vs continuing conversation

**US-4: API Integration**
> As a developer, I want to integrate the chatbot into my application via API so that I can provide conversational features to my users.

**Acceptance Criteria:**
- RESTful API with clear documentation
- Support for session management
- Error responses follow standard patterns
- Rate limiting prevents abuse

**US-5: Graceful Degradation**
> As a user, I want clear feedback when the system can't help me so that I know what to do next.

**Acceptance Criteria:**
- Bot acknowledges when it doesn't understand
- Provides helpful error messages
- Suggests alternatives or human support when needed

### Future User Stories (RAG Design)

**US-6: Document-Aware Responses**
> As a user, I want the chatbot to answer questions based on uploaded documents so that I get accurate, specific information.

**US-7: Source Attribution**
> As a user, I want to know where the chatbot's information comes from so that I can verify important details.

## Functional Requirements

### MVP Requirements

#### FR-1: Conversation Management
- **FR-1.1:** Accept text input with token counting for context limits
- **FR-1.2:** Generate responses using selected Ollama model
- **FR-1.3:** Support model switching with context preservation per model
- **FR-1.4:** Maintain conversation context (last 10 exchanges or 3500 tokens)
- **FR-1.5:** Support standard Ollama parameters (temperature, top_p, etc.)
- **FR-1.6:** Provide queue position feedback during processing

#### FR-2: Session Management
- **FR-2.1:** Generate unique session IDs (UUID v4) with request tracking
- **FR-2.2:** Store conversation history with atomic writes
- **FR-2.3:** Support session retrieval with smart loading (recent messages only)
- **FR-2.4:** Implement session archival after 24 hours (not deletion)
- **FR-2.5:** Prevent concurrent session corruption with file locking

#### FR-3: Storage and Persistence
- **FR-3.1:** Save conversations to persistent storage (file system for MVP)
- **FR-3.2:** Support JSON serialization of conversation data
- **FR-3.3:** Implement automatic cleanup of expired sessions
- **FR-3.4:** Provide conversation export functionality

#### FR-4: API Endpoints
- **FR-4.1:** POST `/api/chat` - Ollama-compatible chat endpoint
- **FR-4.2:** POST `/api/generate` - Ollama-compatible completion endpoint  
- **FR-4.3:** GET `/api/tags` - List available models (Ollama format)
- **FR-4.4:** GET `/sessions/{id}/history` - Retrieve conversation history
- **FR-4.5:** GET `/health` - System health check

#### FR-6: Queue Management
- **FR-6.1:** Implement request queue for sequential Ollama processing
- **FR-6.2:** Display queue position to waiting users
- **FR-6.3:** Support queue depth monitoring
- **FR-6.4:** Implement fair queuing (FIFO with timeout)
- **FR-6.5:** Provide estimated wait time based on model

### Future Requirements (RAG Design Only)

#### FR-6: Document Management
- **FR-6.1:** Accept document uploads (PDF, TXT, DOCX)
- **FR-6.2:** Process documents into searchable chunks
- **FR-6.3:** Store document embeddings in vector database

#### FR-7: Retrieval-Augmented Generation
- **FR-7.1:** Search relevant documents based on user query
- **FR-7.2:** Include document context in LLM prompts
- **FR-7.3:** Provide source citations in responses

## Non-Functional Requirements

### Performance
- **NFR-1:** Queue position updates < 100ms
- **NFR-2:** Support 50 concurrent sessions (queued appropriately)
- **NFR-3:** Memory usage < 200MB + 10MB per active session
- **NFR-4:** Token counting accuracy within 10% of actual

### Reliability  
- **NFR-5:** Automatic reconnection to Ollama with exponential backoff
- **NFR-6:** No session data loss on concurrent access
- **NFR-7:** Graceful handling of context window limits
- **NFR-8:** Model warm-up on startup to prevent cold starts

### Monitoring
- **NFR-9:** Request ID tracking for debugging
- **NFR-10:** Queue depth and wait time metrics
- **NFR-11:** Context truncation rate tracking
- **NFR-12:** Model loading time measurements

## Technical Architecture

### System Components

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Ollama Client  │────▶│  PocketFlow API  │────▶│   Ollama Server │
│  (Existing)     │     │  (Compatibility) │     │   (localhost)   │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                        ┌────────▼─────────┐
                        │                  │
                        │  Session Store   │
                        │   (JSON/Redis)   │
                        │                  │
                        └──────────────────┘
```

### PocketFlow Implementation

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│             │     │             │     │              │
│ OllamaNode  │────▶│  ChatNode   │────▶│ StorageNode  │
│             │     │             │     │              │
└──────┬──────┘     └──────┬──────┘     └──────────────┘
       │                   │
       └───────────────────┘
        (Loop for continuation)
```

### Technology Stack
- **Framework:** PocketFlow (Python)
- **Web Server:** FastAPI (async support for streaming)
- **LLM Provider:** Ollama (local)
- **Storage:** JSON files (MVP), Redis-ready
- **Models:** Any Ollama-compatible model

## API Specification

### Ollama Compatibility
The API mimics Ollama's interface to ensure existing clients work without modification.

### Endpoints

#### POST /api/chat
Ollama-compatible chat endpoint with session management.

**Request:**
```json
{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9
  },
  "session_id": "optional-uuid"  // Extension for persistence
}
```

**Response (non-streaming):**
```json
{
  "model": "llama2",
  "created_at": "2025-08-05T10:30:00Z",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "done": true,
  "session_id": "uuid-string"  // Extension
}
```

#### POST /api/generate
Ollama-compatible completion endpoint.

**Request:**
```json
{
  "model": "llama2",
  "prompt": "Once upon a time",
  "stream": true,
  "options": {
    "temperature": 0.8
  }
}
```

**Response (streaming):**
```json
{"model":"llama2","created_at":"2025-08-05T10:30:00Z","response":"there","done":false}
{"model":"llama2","created_at":"2025-08-05T10:30:00Z","response":" was","done":false}
{"model":"llama2","created_at":"2025-08-05T10:30:00Z","response":"","done":true}
```

#### GET /api/tags
List available models (Ollama format).

**Response:**
```json
{
  "models": [
    {
      "name": "llama2",
      "modified_at": "2025-08-01T10:00:00Z",
      "size": 3825819519
    },
    {
      "name": "mistral",
      "modified_at": "2025-08-01T10:00:00Z", 
      "size": 4109916160
    }
  ]
}
```

## Data Models

### Session Model
```python
{
  "session_id": "uuid",
  "created_at": "datetime",
  "last_activity": "datetime",
  "api_key_hash": "string",
  "messages": [Message],
  "metadata": {
    "user_agent": "string",
    "ip_address": "string"
  }
}
```

### Message Model
```python
{
  "id": "uuid",
  "role": "user|assistant|system",
  "content": "string",
  "timestamp": "datetime",
  "metadata": {
    "tokens_used": "integer",
    "processing_time": "float"
  }
}
```

### Configuration Model
```python
{
  "ollama_host": "http://localhost:11434",
  "default_model": "llama2",
  "context_window": 10,
  "session_timeout": 86400,  # 24 hours
  "storage_backend": "json",  # or "redis"
  "enable_streaming": true
}
```

### Ollama Integration Model
```python
{
  "model": "string",  # Model name from Ollama
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": -1,
    "stop": ["Human:", "User:"]
  }
}
```

## Future Design: RAG Integration

### Architecture Extension
The MVP architecture is designed to accommodate RAG with minimal changes:

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│             │     │             │     │              │
│ VectorStore │────▶│  ChatNode   │────▶│ ResponseNode │
│    Node     │     │   (w/RAG)   │     │              │
└─────────────┘     └─────────────┘     └──────────────┘
       ▲
       │
┌──────┴──────┐
│             │
│  Documents  │
│             │
└─────────────┘
```

### Integration Points
1. **Document Upload API:** New endpoints for document management
2. **Vector Database:** Pinecone or Weaviate for embeddings
3. **Retrieval Node:** New PocketFlow node for similarity search
4. **Enhanced Chat Node:** Modified to include retrieved context

### Migration Path
1. Add document upload endpoints
2. Implement vector storage node
3. Modify chat flow to check for relevant documents
4. Enhance prompts with retrieved context
5. Add source citation in responses

## Success Metrics

### MVP Metrics
- **Queue Performance:** 95% of users see queue position within 1 second
- **Context Preservation:** <5% of conversations hit context truncation
- **Session Reliability:** Zero corrupted sessions in production
- **User Experience:** <10% abandonment rate while queued
- **Model Performance:** Average queue depth <5 requests

### Future Metrics (Post-RAG)
- **Accuracy:** 90%+ relevant document retrieval
- **Coverage:** 80%+ questions answered from documents
- **Citations:** 95%+ responses include sources

## Timeline and Milestones

### Phase 1: MVP Development (4 weeks)

**Week 1: Core Development**
- PocketFlow setup with Ollama integration
- Basic chat node implementation
- Ollama API compatibility layer

**Week 2: Session Management**
- Session persistence
- Context management
- Model switching support

**Week 3: API Completion**
- Full Ollama endpoint compatibility
- Streaming response support
- Error handling

**Week 4: Testing and Deployment**
- Compatibility testing with existing client
- Documentation
- Local deployment guide

### Phase 2: RAG Design (1 week)
- Technical design for document integration
- Vector store selection
- Proof of concept

## Risks and Mitigations

### Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Ollama single-threading bottleneck | High | Certain | Queue management, clear user feedback |
| Context window exceeded silently | High | High | Token counting, user notifications |
| Session file corruption | High | Medium | Atomic writes, file locking |
| Memory leaks from streaming | Medium | Medium | Buffer management, chunking |
| Model cold starts | Medium | High | Warm-up on startup, keep-alive |

### Business Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Client compatibility | High | Low | Extensive testing |
| Performance expectations | Medium | Medium | Clear model benchmarks |
| Local resource limits | Medium | High | Document requirements |

## Appendices

### A. Glossary
- **LLM:** Large Language Model
- **RAG:** Retrieval-Augmented Generation
- **MVP:** Minimum Viable Product
- **UUID:** Universally Unique Identifier

### B. References
- PocketFlow Documentation
- Ollama API Documentation
- FastAPI Streaming Guide

### C. Example Code
```python
# Ollama-compatible request handling
async def chat(request: OllamaChatRequest):
    # Use existing session or create new
    session = get_or_create_session(request.session_id)
    
    # Forward to Ollama with session context
    messages = session.get_context() + request.messages
    
    # Stream response while saving to session
    async for chunk in ollama.chat(
        model=request.model,
        messages=messages,
        stream=request.stream
    ):
        yield chunk
        session.add_response(chunk)
```

### D. Change Log
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Jan 2025 | Initial draft | - |
| 1.1 | Aug 2025 | Ollama integration, simplified scope | - |

---

**Approval Signatures**

Product Owner: _________________________ Date: _________

Technical Lead: ________________________ Date: _________

Engineering Manager: ___________________ Date: _________