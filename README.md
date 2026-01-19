# 🛡️ VeriSmart - Real-Time Credibility Engine

<div align="center">

![VeriSmart](https://img.shields.io/badge/VeriSmart-Real--Time%20Credibility%20Engine-6366f1?style=for-the-badge&logo=shield&logoColor=white)

[![Powered by Pathway](https://img.shields.io/badge/Powered%20by-Pathway-orange?style=flat-square)](https://pathway.com)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-yellow?style=flat-square)](https://huggingface.co/spaces)
[![Render](https://img.shields.io/badge/Render-Frontend-46E3B7?style=flat-square)](https://render.com)

**A real-time misinformation detection and correction system powered by Pathway for live data streaming and RAG capabilities.**

[🚀 Live Demo](#-live-demo) • [✨ Features](#-features) • [🏗️ Architecture](#️-architecture) • [📦 Deployment](#-deployment)

</div>

---

## 🎯 Problem Statement

Misinformation spreads faster than corrections, reaching millions before being debunked. Users lack:
- ⏱️ Real-time credibility signals for viral claims
- 📚 Context-aware evidence that explains why claims are true/false
- 📬 Timely corrections delivered when most effective
- 🔍 Transparent, auditable scoring they can trust

**VeriSmart** solves this by providing a **Live AI** system that perceives, learns, and reasons in real-time.

---

## 🚀 Live Demo

| Component | URL |
|-----------|-----|
| **Frontend Dashboard** | [https://verismart-1tcm.onrender.com](https://verismart-1tcm.onrender.com) |
| **Backend API** | [https://aryan12345ark-verismart.hf.space](https://huggingface.co/spaces/aryan12345ark/VeriSmart)) |

---

## ✨ Features

### Core Capabilities
- 📊 **Real-time Claim Monitoring** - Ingests news from NewsAPI & GNews continuously
- 🔬 **Evidence-based Credibility Scoring** - 0-100% scores with full transparency
- 📚 **Multi-source Evidence Collection** - PubMed, WHO, CDC, fact-checkers
- 🤖 **LLM-powered Claim Extraction** - OpenAI/Gemini for intelligent parsing
- 📬 **Non-backfire Correction Delivery** - Research-backed framing
- 🎚️ **Customizable Evidence Profiles** - Strict Science, Institutional, Broad
- 🔒 **Privacy-preserving Design** - Anonymized, opt-in only

### Technical Highlights
- ⚡ **Pathway Streaming Engine** - Real-time data processing
- 🔄 **Incremental Updates** - No batch reprocessing needed
- 📈 **Live Dashboard** - Auto-updating statistics and charts
- 🛡️ **Transparent Scoring** - Full breakdown of how scores are calculated

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VeriSmart Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  News APIs   │    │   Pathway    │    │   Evidence   │    │ Credibility│ │
│  │  - NewsAPI   │───▶│  Streaming   │───▶│  Collector   │───▶│   Scorer   │ │
│  │  - GNews     │    │   Engine     │    │  - PubMed    │    │            │ │
│  │  - RSS       │    │              │    │  - WHO/CDC   │    │            │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         │                   ▼                   │                   │        │
│         │          ┌──────────────┐             │                   │        │
│         │          │     LLM      │             │                   │        │
│         └─────────▶│   Claim      │◀────────────┘                   │        │
│                    │  Extraction  │                                 │        │
│                    │  - OpenAI    │                                 │        │
│                    │  - Gemini    │                                 │        │
│                    └──────────────┘                                 │        │
│                                                                     │        │
│  ┌──────────────────────────────────────────────────────────────────┘        │
│  │                                                                           │
│  ▼                                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   REST API   │───▶│   Frontend   │───▶│    Users     │                   │
│  │   (Flask)    │    │  Dashboard   │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Deployment

### Backend: HuggingFace Spaces

1. **Create a new Space** at [huggingface.co/spaces](https://huggingface.co/spaces)
   - Select **Gradio** as SDK
   - Choose **Free CPU** tier

2. **Upload files:**
   ```
   app.py                    ← backend/app.py
   requirements.txt          ← backend/requirements-hf.txt (renamed)
   ```

3. **Add Secrets** in Space Settings → Repository secrets:
   | Secret Name | Value |
   |-------------|-------|
   | `NEWS_API_KEY` | Your NewsAPI.org key |
   | `GNEWS_API_KEY` | Your GNews.io key |
   | `GEMINI_API_KEY` | Your Google Gemini key (FREE) |

4. **Deploy** - Space auto-builds and runs!

### Frontend: Render

1. **Create a new Static Site** at [render.com](https://render.com)
   - Connect your GitHub repository
   - Build Command: `echo "No build needed"`
   - Publish Directory: `./`

2. **Update Frontend API URL** in `index.html`:
   ```javascript
   const API_BASE = 'https://YOUR_USERNAME-verismart.hf.space';
   ```

3. **Deploy** - Render auto-deploys on push!

### Keep-Alive Pinger: GitHub Actions

1. **Add Repository Secrets** in GitHub → Settings → Secrets:
   | Secret Name | Value |
   |-------------|-------|
   | `HF_SPACES_URL` | `https://YOUR_USERNAME-verismart.hf.space` |
   | `RENDER_FRONTEND_URL` | `https://verismart.onrender.com` |

2. **Enable Actions** - The workflow runs every 8 minutes automatically

---

## 🔑 API Keys Required

### Required (Get at least one from each)

#### News APIs
| API | Free Tier | Get Key |
|-----|-----------|---------|
| **NewsAPI.org** | 100 requests/day | [newsapi.org/register](https://newsapi.org/register) |
| **GNews.io** | 100 requests/day | [gnews.io](https://gnews.io/) |

#### LLM APIs
| API | Free Tier | Get Key |
|-----|-----------|---------|
| **Google Gemini** ⭐ | FREE generous tier | [ai.google.dev](https://ai.google.dev/) |
| **OpenAI** | Pay-as-you-go | [platform.openai.com](https://platform.openai.com/api-keys) |

---

## 📡 API Reference

### Base URL
```
https://YOUR_USERNAME-verismart.hf.space
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check & status |
| `/api/claims` | GET | List all claims with scores |
| `/api/claims/<id>` | GET | Get detailed claim info |
| `/api/analyze` | POST | Analyze a new claim |
| `/api/stats` | GET | Dashboard statistics |
| `/api/trending` | GET | Trending claims |
| `/api/activity` | GET | Recent activity feed |
| `/api/evidence-sources` | GET | List evidence sources |

### Example: Analyze a Claim
```bash
curl -X POST https://YOUR_USERNAME-verismart.hf.space/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "COVID vaccines cause autism"}'
```

---

## 🎥 Video Demo Script (3 min)

### 0:00-0:20 - Introduction
"Hi, we're Aryan and Khushboo. We built VeriSmart, a real-time credibility engine that detects and corrects misinformation as it spreads."

### 0:20-1:00 - Live Dashboard
"Here's our live dashboard showing claims being tracked in real-time. Watch how the stats update automatically as new news comes in."

### 1:00-1:40 - Evidence Analysis
"Let me analyze a claim live. I'll enter 'Vitamin D prevents COVID' and you can see the evidence being collected from PubMed and scored instantly."

### 1:40-2:20 - Proving Real-Time
"To prove this is truly real-time, watch what happens when I trigger a news refresh... New claims appear immediately in our feed."

### 2:20-2:50 - Correction System
"Here's how corrections are delivered - non-backfire framing that respects users while providing accurate information."

### 2:50-3:00 - Conclusion
"VeriSmart: Real-time credibility at scale. Thank you!"

---

## 📁 Project Structure

```
verismart/
├── index.html                    # Frontend dashboard
├── render.yaml                   # Render deployment config
├── README.md                     # This file
├── backend/
│   ├── app.py                    # HF Spaces main app
│   ├── requirements-hf.txt       # HF dependencies
│   └── README.md                 # Backend docs
└── .github/
    └── workflows/
        └── keep-alive.yml        # Pinger workflow
```

---

## 🏆 Hackathon Criteria Alignment

| Criterion | Weight | Our Implementation |
|-----------|--------|-------------------|
| **Real-Time Capability** | 35% | ✅ Pathway streaming, live news ingestion, instant updates |
| **Technical Implementation** | 30% | ✅ Clean modular code, Pathway patterns, REST API |
| **Innovation & UX** | 20% | ✅ Beautiful dashboard, evidence profiles, transparent scoring |
| **Impact & Feasibility** | 15% | ✅ Real-world problem, scalable architecture, production-ready |

---

## 👥 Team

| Name | Role |
|------|------|
| **Aryan** | Backend & Pathway Integration |
| **Khushboo** | Frontend & UX Design |

---

## 📄 License

© 2026 Aryan & Khushboo. All rights reserved.

---

<div align="center">

**Built with ❤️ for the Pathway Hackathon**

Powered by [Pathway](https://pathway.com) • Hosted on [HuggingFace](https://huggingface.co) & [Render](https://render.com)

</div>
