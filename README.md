# 🤖 Stamper

> *Your personal Jarvis. One assistant, infinite agents.*

Stamper is a personalized, multi-agent AI architecture built in Python. It brings together a growing suite of intelligent agents — each specialized for a different task — all under one roof. Think of it as your all-in-one AI companion that works the way you do.

---

## ✨ What Can Stamper Do?

| Agent / Module | Description |
|---|---|
| 🔬 **Research Agent** | Deep-dives into topics and surfaces structured, useful information |
| 🌐 **Web Search** | Searches the web in real time to fetch current information |
| 🖥️ **Coding Agent** | Writes, debugs, and explains code across languages |
| 🧠 **Memory** | Remembers context across your conversations |
| 🔊 **Voice** | Interact with Stamper using your voice |
| 📋 **Scraper** | Extracts data from websites |
| 📲 **WhatsApp Reminders** | Sends you reminders directly on WhatsApp |
| ✅ **Checklist (n8n)** | Manages tasks and checklists via n8n integration |
| *...and more coming* | Stamper is built to grow — new agents are always in the pipeline |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/stamper.git
cd stamper

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Stamper
python main.py
```

---

## 🗂️ Project Structure

```
stamper/
├── main.py                  # Entry point — starts the Stamper system
├── Memory.py                # Handles memory and context persistence
├── Voice.py                 # Voice input/output interface
├── scraper.py               # Web scraping module
├── Stamper.ipynb            # Interactive notebook for exploration
├── integrations/            # External integrations (n8n, WhatsApp, etc.)
├── stamper_memory_db/       # Local memory database
├── requirements.txt         # Python dependencies
└── Untitled Diagram.drawio.png  # Architecture diagram
```

---

## 🧩 Architecture

Stamper is built on a **multi-agent architecture** — a central orchestrator that routes tasks to the right specialized agent depending on what you ask. Agents can work independently or collaborate on complex tasks.

![Architecture Diagram](Untitled%20Diagram.drawio.png)

---

## ⚙️ Configuration

Some modules require additional setup:

- **WhatsApp Reminders** — Requires WhatsApp API configuration (see `integrations/`).
- **n8n Checklist** — Requires a running n8n instance and webhook setup.
- **Memory Persistence** — Uses a local DB in `stamper_memory_db/`. No extra setup needed.

> Detailed configuration instructions are available in the `integrations/` folder.

---

## 🧠 Truly Personalized — Like a Friend Who Actually Remembers

Most AI tools forget you the moment the conversation ends. Stamper doesn't.

Thanks to its built-in memory, Stamper remembers things about you — your preferences, your ongoing projects, your past conversations. Over time, it gets to know you. You can talk to Stamper the way you'd talk to a close friend: casually, in context, without having to re-explain yourself every single time. It's not just an assistant — it's *your* assistant.

---

## 📱 Available on Mobile

Stamper is coming to your pocket. A mobile app is in the works so you'll be able to access all your agents, reminders, and conversations on the go — right from your phone.

---

## 🌍 Open Source & Agent Marketplace

Stamper is fully open source. The community is at the heart of it.

If you run into a problem and think *"there should be an agent for this"* — build one! Anyone can develop a new agent and submit it to the **Stamper Agent Marketplace**. Once reviewed and approved, your agent becomes available to all Stamper users worldwide.

This means the more people use Stamper, the smarter and more capable it gets — driven entirely by real problems that real people face.

**Want to contribute an agent?**
1. Build your agent following the contribution guidelines
2. Submit a pull request with your agent
3. Once approved, it gets listed on the marketplace for everyone to use

> Contributions of all kinds are welcome — agents, bug fixes, integrations, and ideas.

---

## 🛣️ Roadmap

- [x] Research Agent
- [x] Web Search Module
- [x] Coding Agent
- [x] Voice Interface
- [x] Memory (session-based)
- [x] WhatsApp Reminders
- [x] Scraper
- [x] n8n Checklist Integration
- [ ] Persistent memory across sessions
- [ ] Improved scraper for broader queries
- [ ] Mobile app (iOS & Android)
- [ ] Agent Marketplace
- [ ] More agents — the list keeps growing!

---

## 📄 License

This project is licensed under the terms of the MIT file.

---

<p align="center">Built with curiosity. Powered by agents. 🚀</p>
