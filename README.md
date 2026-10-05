# 🤖 Jarvis: 24/7 Autonomous Multimodal AI Assistant on WhatsApp

Jarvis is a production-grade, 24/7 personal AI agent deployed on WhatsApp via the **Meta WhatsApp Cloud API**. Engineered to run continuously on an **AWS EC2 `t3.small`** instance within strict zero-cost / credit-budget boundaries, Jarvis combines multimodal processing (Voice, Vision, Documents), an autonomous **ReAct reasoning loop**, an isolated **Model Context Protocol (MCP)** server, and a **Qdrant Embedded Vector RAG** memory store.

---

## Capabilities

-  **Voice Processing**: High-fidelity speech transcription via Groq **Whisper-large-v3** with automatic translation from Malayalam, Hindi, and regional languages to English.
- **Computer Vision**: Dual-engine image analysis (Google Gemini 3.6 Flash + AWS Bedrock Claude 3.5 Sonnet / Nova Lite) for instant OCR, document transcription, and scene understanding.
-  **Document Intelligence**: In-memory streaming text extraction and question answering for digital PDFs using `pypdf`.
- **OpenStreetMap MCP Server**: Standalone Model Context Protocol (MCP) server running via `stdio` JSON-RPC 2.0. Provides keyless, zero-cost radial bounding-box POI searches (`viewbox`) and OSRM driving distance/route pathfinding.
- **Qdrant Embedded Vector RAG**: Persistent semantic memory running directly in Python via compiled Rust bindings. Uses Google Gemini **768-dimensional embeddings** and **HNSW Cosine Similarity** with a tiny **~35 MB RAM footprint**.
- **Autonomous Reminders**: Independent 15-second background polling daemon managing scheduled reminders, recurring alerts, and proactive WhatsApp notification dispatches.
- **Self-Healing ReAct Loop**: Autonomous reasoning loop with automatic error interception (`tool_use_failed` regex recovery), tool name aliasing, rate-limit backoff, and strict persona guardrails against prompt injection.

---

##  System Architecture

```
                    ┌──────────────────────────────┐
                    │      WhatsApp User App       │
                    └──────────────┬───────────────┘
                                   │ HTTPS Webhook
                                   ▼
                    ┌──────────────────────────────┐
                    │      Nginx Reverse Proxy     │
                    │   (Let's Encrypt SSL/TLS)    │
                    └──────────────┬───────────────┘
                                   │ Proxy Pass (127.0.0.1:8000)
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │             FastAPI Backend (Uvicorn 24/7)            │
        │                                                       │
        │  ┌─────────────────┐           ┌───────────────────┐  │
        │  │ Webhook Handler │           │ Reminder Worker   │  │
        │  └────────┬────────┘           │ (Polls every 15s) │  │
        │           │                    └─────────┬─────────┘  │
        │           ▼                              │            │
        │  ┌─────────────────────────────────────┐ │            │
        │  │      Hermes ReAct Agent Core        │ │            │
        │  │  (Reason -> Act -> Observe -> Done) │ │            │
        │  └──────┬──────────────────────┬───────┘ │            │
        └─────────┼──────────────────────┼─────────┼────────────┘
                  │                      │         │
       ┌──────────┴──────────┐           │         │
       │ External AI APIs    │           │         │
       │ • Groq (LLM & STT)  │           │         │
       │ • Gemini (Vision)   │           │         │
       │ • Bedrock (Vision)  │           │         │
       └─────────────────────┘           │         │
                                         ▼         ▼
      ┌────────────────────────────────────────────────────────┐
      │               Local Storage & Vector Store             │
      │  • SQLite: sessions.db, memories.db, reminders.db      │
      │  • Qdrant Embedded: data/qdrant_db (HNSW 768-dim)      │
      │  • Model Context Protocol (MCP): OpenStreetMap Server  │
      └────────────────────────────────────────────────────────┘
```

---

##  Repository Structure

```
├── agent/
│   ├── __init__.py
│   └── core.py                 # Hermes ReAct Agent loop, ToolRegistry & auto-recovery
├── app/
│   ├── document.py             # In-memory PDF text extraction
│   ├── llm.py                  # System prompt generator, WhatsApp formatters & configs
│   ├── main.py                 # FastAPI webhook router & background dispatchers
│   ├── session.py              # SQLite chat history manager & sanitizer
│   ├── transcriber.py          # Groq Whisper-large-v3 voice transcriber
│   ├── vision.py               # Gemini & Bedrock multimodal vision analyzer
│   └── whatsapp_client.py      # Meta Graph API async client
├── data/                       # (Git-ignored) Local SQLite DBs and Qdrant storage
│   ├── memories.db
│   ├── qdrant_db/
│   ├── reminders.db
│   └── sessions.db
├── mcp/
│   ├── client.py               # Universal Anthropic MCP JSON-RPC 2.0 stdio client
│   ├── servers/
│   │   └── osm_server.py       # Standalone OpenStreetMap MCP server
│   └── tools/
│       ├── browse.py           # Clean webpage scraping tool
│       ├── memory.py           # Hybrid Qdrant Vector + SQLite memory tool
│       ├── reminder.py         # Reminder scheduling & polling tool
│       └── search.py           # DuckDuckGo live web search tool
├── scripts/
│   └── test_bedrock.py         # Standalone AWS Bedrock latency benchmarking
├── .env.example                # Configuration template
├── .gitignore                  # Comprehensive secrets & database exclusion rules
├── requirements.txt            # Pinned production Python dependencies
└── README.md
```

---

## Setup & Deployment Guide

### 1. Prerequisites
- **Linux Server**: AWS EC2 `t3.small` (Ubuntu 24.04 recommended) with a 2 GB swapfile.
- **Python**: Python 3.12+ with `virtualenv`.
- **Meta WhatsApp Business Account**: Valid Phone Number ID and System User Access Token.
- **API Keys**: Groq Cloud API key and Google Gemini API key.

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/jarvis-whatsapp-agent.git
cd jarvis-whatsapp-agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy the sample environment file and configure your credentials:

```bash
cp .env.example .env
nano .env
```

Ensure the following variables are populated:
- `WEBHOOK_VERIFY_TOKEN`: Your secret string for Meta webhook verification.
- `WHATSAPP_TOKEN`: Permanent Meta WhatsApp Cloud API access token.
- `WHATSAPP_PHONE_NUMBER_ID`: Your Meta WhatsApp Business Phone Number ID.
- `GROQ_API_KEY`: API key from [Groq Console](https://console.groq.com/).
- `GEMINI_API_KEY`: API key from [Google AI Studio](https://aistudio.google.com/).
- `ALLOWED_PHONE_NUMBERS`: Your personal phone number with country code (e.g. `91987...0`) for access security.

### 4. Running the Service

For production, run Jarvis as a `systemd` service:

```ini
# /etc/systemd/system/whatsapp-agent.service
[Unit]
Description=WhatsApp AI Agent FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/whatsapp-agent
ExecStart=/home/ubuntu/whatsapp-agent/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable whatsapp-agent.service
sudo systemctl start whatsapp-agent.service
```

##  Security & Privacy

- **Guardrails**: Persona protection blocks prompt injections, prevents disclosing backend schemas, and enforces strict system prompt isolation.

---

## 📄 License
This project is licensed under the [Apache 2.0 License](LICENSE).

