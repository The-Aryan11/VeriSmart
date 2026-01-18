# VeriSmart Backend - HuggingFace Spaces Deployment

## Setup on HuggingFace Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Select **Gradio** as the SDK
3. Upload these files:
   - `app.py` (main file)
   - `requirements-hf.txt` → rename to `requirements.txt`

4. Add Secrets in Space Settings:
   - `NEWS_API_KEY` - Your NewsAPI.org key
   - `GNEWS_API_KEY` - Your GNews.io key  
   - `OPENAI_API_KEY` - Your OpenAI key (optional)
   - `GEMINI_API_KEY` - Your Google Gemini key (optional)

5. The Space will auto-deploy and provide:
   - Gradio UI at: `https://YOUR_USERNAME-verismart.hf.space`
   - REST API at: `https://YOUR_USERNAME-verismart.hf.space/api/`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/claims` | GET | List all claims |
| `/api/claims/<id>` | GET | Get claim details |
| `/api/analyze` | POST | Analyze new claim |
| `/api/stats` | GET | Dashboard stats |
| `/api/trending` | GET | Trending claims |
| `/api/activity` | GET | Activity feed |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEWS_API_KEY` | Yes* | NewsAPI.org API key |
| `GNEWS_API_KEY` | Yes* | GNews.io API key |
| `OPENAI_API_KEY` | No | OpenAI API key for claim extraction |
| `GEMINI_API_KEY` | No | Google Gemini API key (free alternative) |

*At least one news API key is required for real-time data.

© 2026 Aryan & Khushboo. All rights reserved.
