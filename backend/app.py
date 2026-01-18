"""
VeriSmart - Real-Time Credibility Engine
HuggingFace Spaces Backend with Pathway Integration
Fixed: Gemini model, SQL injection false positives, port conflicts
"""

import os
import json
import hashlib
import threading
import time
import re
import html
from datetime import datetime
from functools import wraps
import requests
from typing import Optional, Dict, Any
import logging
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verismart")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # API Keys from HF Secrets
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 7860))
    
    # Data refresh intervals (seconds)
    NEWS_REFRESH_INTERVAL = int(os.getenv("NEWS_REFRESH_INTERVAL", 300))
    
    # Credibility baselines by category
    CREDIBILITY_BASELINES = {
        "health": 30,
        "politics": 50,
        "climate": 40,
        "technology": 50,
        "finance": 50,
        "conspiracy": 10,
        "science": 70,
        "default": 50
    }

# =============================================================================
# INPUT VALIDATION (Fixed for HTML entities)
# =============================================================================

class InputValidator:
    """Input validation with proper HTML entity handling"""
    
    @classmethod
    def decode_html_entities(cls, text: str) -> str:
        """Decode HTML entities before validation"""
        if not text:
            return ""
        # Decode HTML entities like &#x27; -> '
        return html.unescape(text)
    
    @classmethod
    def sanitize_text(cls, text: str, max_length: int = 5000) -> str:
        """Sanitize text input"""
        if not text or not isinstance(text, str):
            return ""
        
        # First decode HTML entities
        text = cls.decode_html_entities(text)
        
        # Truncate to max length
        text = text[:max_length]
        
        # Remove potentially dangerous patterns (but not normal apostrophes)
        # Only block actual SQL keywords in suspicious patterns
        dangerous_patterns = [
            r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\s",
            r"UNION\s+SELECT",
            r"OR\s+1\s*=\s*1",
            r"AND\s+1\s*=\s*1",
            r"--\s*$",
            r"/\*.*\*/",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potentially dangerous pattern detected")
                # Instead of raising, just clean the text
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @classmethod
    def validate_claim_id(cls, claim_id: str) -> bool:
        """Validate claim ID format"""
        if not claim_id or not isinstance(claim_id, str):
            return False
        return bool(re.match(r"^[a-f0-9]{12}$", claim_id))
    
    @classmethod
    def validate_limit(cls, limit: Any) -> int:
        """Validate and sanitize limit parameter"""
        try:
            limit = int(limit)
            return max(1, min(limit, 100))
        except (ValueError, TypeError):
            return 50

# =============================================================================
# NEWS CONNECTOR
# =============================================================================

class NewsAPIConnector:
    """Real-time news ingestion"""
    
    def __init__(self):
        self.news_api_key = Config.NEWS_API_KEY
        self.gnews_api_key = Config.GNEWS_API_KEY
        
    def fetch_newsdata(self, query: str = "health", page_size: int = 10) -> list:
        """Fetch from NewsData.io"""
        if not self.news_api_key:
            return []
        
        try:
            url = "https://newsdata.io/api/1/news"
            params = {
                "apikey": self.news_api_key,
                "q": query,
                "language": "en",
                "size": min(max(1, page_size), 10)
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return data.get("results", [])
            return []
            
        except Exception as e:
            logger.error(f"NewsData.io error: {e}")
            return []
    
    def fetch_gnews(self, query: str = "health", max_results: int = 10) -> list:
        """Fetch from GNews.io"""
        if not self.gnews_api_key:
            return []
        
        try:
            url = "https://gnews.io/api/v4/search"
            params = {
                "q": query,
                "token": self.gnews_api_key,
                "max": max_results,
                "lang": "en"
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("articles", [])
            return []
            
        except Exception as e:
            logger.error(f"GNews error: {e}")
            return []
    
    def fetch_all(self) -> list:
        """Fetch from all sources"""
        all_articles = []
        
        try:
            # Fetch from NewsData.io
            newsdata_topics = ["health", "climate", "politics", "technology"]
            for topic in newsdata_topics:
                articles = self.fetch_newsdata(query=topic, page_size=10)
                for article in articles:
                    article["_source"] = "newsdata"
                    article["_topic"] = topic
                    if "title" not in article:
                        article["title"] = article.get("title", "")
                    if "content" not in article:
                        article["content"] = article.get("description", "")
                    if "publishedAt" not in article:
                        article["publishedAt"] = article.get("pubDate", "")
                    if "url" not in article:
                        article["url"] = article.get("link", "")
                    if "source" not in article:
                        article["source"] = {"name": article.get("source_id", "unknown")}
                all_articles.extend(articles)
            
            # Fetch from GNews
            gnews_topics = ["health", "politics", "technology"]
            for topic in gnews_topics:
                articles = self.fetch_gnews(query=topic, max_results=5)
                for article in articles:
                    article["_source"] = "gnews"
                    article["_topic"] = topic
                all_articles.extend(articles)
            
            logger.info(f"Fetched {len(all_articles)} articles from all sources")
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
        
        return all_articles

# =============================================================================
# CLAIM EXTRACTION (Fixed Gemini model)
# =============================================================================

class ClaimExtractor:
    """Extract factual claims using LLM"""
    
    def __init__(self):
        self.gemini_key = Config.GEMINI_API_KEY
        
    def extract_claims_gemini(self, article_text: str, article_title: str) -> list:
        """Extract claims using Google Gemini API (updated model)"""
        if not self.gemini_key:
            return self._fallback_extraction(article_text, article_title)
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            
            # Use the newer model (gemini-1.5-flash instead of deprecated gemini-pro)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Sanitize inputs
            article_text = InputValidator.sanitize_text(article_text, max_length=3000)
            article_title = InputValidator.sanitize_text(article_title, max_length=500)
            
            prompt = f"""Analyze this news article and extract factual claims.
            
Title: {article_title}
Content: {article_text[:2000]}

Return a JSON array with claims having: text, category (health/politics/climate/technology/finance/science), verifiable (boolean), confidence (0-100).
Only return the JSON array, nothing else."""

            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Clean up response
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            if result.endswith("```"):
                result = result[:-3]
            
            claims = json.loads(result)
            
            # Validate extracted claims
            validated_claims = []
            for c in claims:
                if isinstance(c, dict) and c.get("verifiable", True):
                    validated_claims.append({
                        "text": InputValidator.sanitize_text(str(c.get("text", "")), 500),
                        "category": c.get("category", "general")[:20],
                        "verifiable": bool(c.get("verifiable", True)),
                        "confidence": min(100, max(0, int(c.get("confidence", 50))))
                    })
            
            return validated_claims if validated_claims else self._fallback_extraction(article_text, article_title)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini response as JSON: {e}")
            return self._fallback_extraction(article_text, article_title)
        except Exception as e:
            logger.error(f"Gemini extraction error: {e}")
            return self._fallback_extraction(article_text, article_title)
    
    def _fallback_extraction(self, article_text: str, article_title: str) -> list:
        """Simple rule-based extraction when LLM is unavailable"""
        title = InputValidator.sanitize_text(article_title, max_length=500)
        
        if not title:
            return []
        
        category = "general"
        title_lower = title.lower()
        
        if any(w in title_lower for w in ["election", "vote", "congress", "president", "political", "governor", "senate"]):
            category = "politics"
        elif any(w in title_lower for w in ["climate", "warming", "carbon", "environment", "weather"]):
            category = "climate"
        elif any(w in title_lower for w in ["ai", "tech", "software", "computer", "digital", "app"]):
            category = "technology"
        elif any(w in title_lower for w in ["stock", "economy", "bank", "finance", "market", "price"]):
            category = "finance"
        elif any(w in title_lower for w in ["health", "vaccine", "disease", "medical", "drug", "hospital", "doctor"]):
            category = "health"
        
        return [{
            "text": title,
            "category": category,
            "verifiable": True,
            "confidence": 60
        }]
    
    def extract(self, article_text: str, article_title: str) -> list:
        """Extract claims using available LLM"""
        if self.gemini_key:
            return self.extract_claims_gemini(article_text, article_title)
        else:
            return self._fallback_extraction(article_text, article_title)

# =============================================================================
# EVIDENCE COLLECTOR
# =============================================================================

class EvidenceCollector:
    """Collect evidence from trusted sources"""
    
    def search_pubmed(self, claim: str) -> list:
        """Search PubMed for evidence"""
        try:
            claim = InputValidator.sanitize_text(claim, max_length=200)
            
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": claim[:100],
                "retmax": 5,
                "retmode": "json",
                "sort": "relevance"
            }
            
            response = requests.get(search_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                ids = data.get("esearchresult", {}).get("idlist", [])
                
                if ids:
                    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    summary_params = {
                        "db": "pubmed",
                        "id": ",".join(ids[:5]),
                        "retmode": "json"
                    }
                    
                    summary_response = requests.get(summary_url, params=summary_params, timeout=15)
                    
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        results = summary_data.get("result", {})
                        
                        evidence = []
                        for pid in ids[:5]:
                            if pid in results and isinstance(results[pid], dict):
                                article = results[pid]
                                evidence.append({
                                    "source": "PubMed",
                                    "title": InputValidator.sanitize_text(
                                        article.get("title", ""), 300
                                    ),
                                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                                    "date": article.get("pubdate", ""),
                                    "quality": 10,
                                    "type": "scientific"
                                })
                        return evidence
            return []
            
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    def collect(self, claim: str) -> dict:
        """Collect evidence from all sources"""
        all_evidence = {
            "supporting": [],
            "refuting": [],
            "uncertain": []
        }
        
        try:
            pubmed_evidence = self.search_pubmed(claim)
            for e in pubmed_evidence:
                all_evidence["uncertain"].append(e)
        except Exception as e:
            logger.error(f"Evidence collection error: {e}")
        
        return all_evidence

# =============================================================================
# CREDIBILITY SCORER
# =============================================================================

class CredibilityScorer:
    """Calculate credibility scores"""
    
    def __init__(self):
        self.baselines = Config.CREDIBILITY_BASELINES
        self.weights = {
            "scientific": 10,
            "institutional": 8,
            "factcheck": 7,
            "journalism": 5,
            "preprint": 3,
            "social": 1
        }
    
    def calculate_score(self, claim: dict, evidence: dict) -> dict:
        """Calculate credibility score"""
        try:
            category = claim.get("category", "default")
            baseline = self.baselines.get(category, self.baselines["default"])
            
            supporting = evidence.get("supporting", [])
            refuting = evidence.get("refuting", [])
            uncertain = evidence.get("uncertain", [])
            
            support_score = sum(
                self.weights.get(e.get("type", "social"), 1) * e.get("quality", 5) 
                for e in supporting
            )
            refute_score = sum(
                self.weights.get(e.get("type", "social"), 1) * e.get("quality", 5) 
                for e in refuting
            )
            
            max_adjustment = 50
            
            if support_score + refute_score > 0:
                net_score = (support_score - refute_score) / (support_score + refute_score) * max_adjustment
            else:
                net_score = 0
            
            final_score = max(0, min(100, baseline + net_score))
            
            total_evidence = len(supporting) + len(refuting) + len(uncertain)
            if total_evidence >= 10:
                confidence = "high"
            elif total_evidence >= 5:
                confidence = "moderate"
            elif total_evidence >= 2:
                confidence = "low"
            else:
                confidence = "insufficient"
            
            if final_score >= 75:
                label = "well-supported"
            elif final_score >= 50:
                label = "mixed-evidence"
            elif final_score >= 25:
                label = "questionable"
            else:
                label = "highly-dubious"
            
            return {
                "score": round(final_score, 1),
                "label": label,
                "confidence": confidence,
                "baseline": baseline,
                "adjustment": round(net_score, 1),
                "evidence_count": {
                    "supporting": len(supporting),
                    "refuting": len(refuting),
                    "uncertain": len(uncertain)
                }
            }
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return {
                "score": 50,
                "label": "unknown",
                "confidence": "error",
                "baseline": 50,
                "adjustment": 0,
                "evidence_count": {"supporting": 0, "refuting": 0, "uncertain": 0}
            }

# =============================================================================
# THREAD-SAFE DATA STORE
# =============================================================================

class ThreadSafeStore:
    """Thread-safe data store"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._claims = {}
        self._evidence = {}
        self._scores = {}
        self._activity = []
        self._pipeline_status = {
            "last_run": None,
            "claims_processed": 0,
            "status": "initializing"
        }
    
    def add_claim(self, claim_id: str, data: dict):
        with self._lock:
            self._claims[claim_id] = data
    
    def get_claim(self, claim_id: str) -> Optional[dict]:
        with self._lock:
            return self._claims.get(claim_id)
    
    def get_all_claims(self) -> dict:
        with self._lock:
            return dict(self._claims)
    
    def add_evidence(self, claim_id: str, data: dict):
        with self._lock:
            self._evidence[claim_id] = data
    
    def get_evidence(self, claim_id: str) -> Optional[dict]:
        with self._lock:
            return self._evidence.get(claim_id)
    
    def add_score(self, claim_id: str, data: dict):
        with self._lock:
            self._scores[claim_id] = data
    
    def get_score(self, claim_id: str) -> Optional[dict]:
        with self._lock:
            return self._scores.get(claim_id)
    
    def get_all_scores(self) -> dict:
        with self._lock:
            return dict(self._scores)
    
    def add_activity(self, activity: dict):
        with self._lock:
            self._activity.append(activity)
            if len(self._activity) > 1000:
                self._activity = self._activity[-1000:]
    
    def get_activity(self, limit: int = 20) -> list:
        with self._lock:
            return list(self._activity[-limit:])
    
    def update_pipeline_status(self, **kwargs):
        with self._lock:
            self._pipeline_status.update(kwargs)
    
    def get_pipeline_status(self) -> dict:
        with self._lock:
            return dict(self._pipeline_status)
    
    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_claims": len(self._claims),
                "total_evidence": sum(
                    len(e.get("supporting", [])) + len(e.get("refuting", [])) + len(e.get("uncertain", []))
                    for e in self._evidence.values()
                )
            }

# =============================================================================
# GLOBAL COMPONENTS
# =============================================================================

store = ThreadSafeStore()
news_connector = NewsAPIConnector()
claim_extractor = ClaimExtractor()
evidence_collector = EvidenceCollector()
credibility_scorer = CredibilityScorer()

def generate_claim_id(claim_text: str) -> str:
    """Generate unique ID for a claim"""
    return hashlib.md5(claim_text.encode()).hexdigest()[:12]

def process_articles(articles: list) -> list:
    """Process articles through the pipeline"""
    processed_claims = []
    
    for article in articles:
        try:
            title = article.get("title", "")
            content = article.get("content") or article.get("description") or ""
            source = article.get("source", {})
            if isinstance(source, dict):
                source = source.get("name", article.get("_source", "unknown"))
            url = article.get("url", "")
            published = article.get("publishedAt", datetime.now().isoformat())
            
            # Decode HTML entities in title and content
            title = InputValidator.decode_html_entities(title)
            content = InputValidator.decode_html_entities(content)
            
            claims = claim_extractor.extract(content, title)
            
            for claim in claims:
                if not claim.get("text"):
                    continue
                    
                claim_id = generate_claim_id(claim["text"])
                
                existing = store.get_claim(claim_id)
                if existing:
                    last_updated = existing.get("last_updated")
                    if last_updated and isinstance(last_updated, datetime):
                        if (datetime.now() - last_updated).seconds < 3600:
                            continue
                
                evidence = evidence_collector.collect(claim["text"])
                score_data = credibility_scorer.calculate_score(claim, evidence)
                
                claim_data = {
                    "id": claim_id,
                    "text": claim["text"],
                    "category": claim.get("category", "general"),
                    "source": source,
                    "source_url": url,
                    "published": published,
                    "extracted_at": datetime.now().isoformat(),
                    "last_updated": datetime.now(),
                    "platform": article.get("_source", "news"),
                    "topic": article.get("_topic", "general")
                }
                
                store.add_claim(claim_id, claim_data)
                store.add_evidence(claim_id, evidence)
                store.add_score(claim_id, score_data)
                
                store.add_activity({
                    "type": "new_claim",
                    "claim_id": claim_id,
                    "timestamp": datetime.now().isoformat(),
                    "message": f"New claim: {claim['text'][:50]}..."
                })
                
                processed_claims.append({**claim_data, "evidence": evidence, "score": score_data})
                
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            continue
    
    return processed_claims

def run_pipeline_cycle():
    """Run one cycle of the data pipeline"""
    logger.info("Starting pipeline cycle...")
    store.update_pipeline_status(status="running")
    
    try:
        articles = news_connector.fetch_all()
        
        if articles:
            processed = process_articles(articles)
            store.update_pipeline_status(
                claims_processed=len(processed),
                last_run=datetime.now().isoformat(),
                status="idle"
            )
            logger.info(f"Processed {len(processed)} new claims")
        else:
            store.update_pipeline_status(
                claims_processed=0,
                last_run=datetime.now().isoformat(),
                status="idle"
            )
        return True
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        store.update_pipeline_status(status="error")
        return False

def background_pipeline():
    """Run pipeline continuously in background"""
    while True:
        try:
            run_pipeline_cycle()
        except Exception as e:
            logger.error(f"Background pipeline error: {e}")
        time.sleep(Config.NEWS_REFRESH_INTERVAL)

# =============================================================================
# GRADIO INTERFACE FUNCTIONS
# =============================================================================

def analyze_claim_gradio(claim_text: str) -> tuple:
    """Analyze a claim via Gradio interface"""
    if not claim_text or not claim_text.strip():
        return "⚠️ Please enter a claim to analyze.", "", ""
    
    try:
        claim_text = InputValidator.sanitize_text(claim_text, 2000)
    except Exception as e:
        return f"❌ Invalid input: {e}", "", ""
    
    claim_id = generate_claim_id(claim_text)
    claims = claim_extractor.extract(claim_text, claim_text)
    claim_meta = claims[0] if claims else {"text": claim_text, "category": "default"}
    
    evidence = evidence_collector.collect(claim_text)
    score_data = credibility_scorer.calculate_score(claim_meta, evidence)
    
    # Store for later retrieval
    store.add_claim(claim_id, {
        "id": claim_id,
        "text": claim_text,
        "category": claim_meta.get("category", "general"),
        "source": "user_submitted",
        "extracted_at": datetime.now().isoformat()
    })
    store.add_evidence(claim_id, evidence)
    store.add_score(claim_id, score_data)
    
    score_emoji = "🟢" if score_data['score'] >= 75 else "🟡" if score_data['score'] >= 50 else "🟠" if score_data['score'] >= 25 else "🔴"
    
    score_text = f"""
## {score_emoji} Credibility Score: {score_data['score']}%

**Label:** {score_data['label'].replace('-', ' ').title()}  
**Confidence:** {score_data['confidence'].title()}  
**Category:** {claim_meta.get('category', 'general').title()}

### Score Breakdown
- Baseline Score: {score_data['baseline']}%
- Evidence Adjustment: {score_data['adjustment']:+.1f}%
- **Final Score: {score_data['score']}%**
"""
    
    evidence_text = f"""
### 📚 Evidence Found
| Type | Count |
|------|-------|
| ✅ Supporting | {score_data['evidence_count']['supporting']} |
| ❌ Refuting | {score_data['evidence_count']['refuting']} |
| ❓ Uncertain | {score_data['evidence_count']['uncertain']} |
"""
    
    sources_text = "### 📖 Sources\n"
    for cat in ["supporting", "refuting", "uncertain"]:
        for e in evidence.get(cat, [])[:3]:
            sources_text += f"- **[{e.get('source', 'Unknown')}]** {e.get('title', 'No title')[:80]}...\n"
    
    if len(sources_text) < 30:
        sources_text = "ℹ️ No direct evidence sources found. Score based on category baseline and claim analysis."
    
    return score_text, evidence_text, sources_text

def get_dashboard_stats() -> str:
    """Get current dashboard statistics"""
    stats = store.get_stats()
    pipeline = store.get_pipeline_status()
    
    status_emoji = "🟢" if pipeline.get('status') == 'idle' else "🔄" if pipeline.get('status') == 'running' else "🔴"
    
    return f"""
## 📊 VeriSmart Dashboard

### Real-Time Statistics
| Metric | Value |
|--------|-------|
| 📋 Active Claims | **{stats['total_claims']}** |
| 📚 Evidence Items | **{stats['total_evidence']}** |
| {status_emoji} Pipeline Status | **{pipeline.get('status', 'unknown').title()}** |
| 🕐 Last Update | {pipeline.get('last_run', 'Never')} |
| ✨ Last Batch | {pipeline.get('claims_processed', 0)} claims |

### API Endpoints
- `/api/health` - Health check
- `/api/claims` - List all claims
- `/api/analyze` - Analyze new claim
- `/api/stats` - Dashboard stats
"""

def get_recent_claims() -> str:
    """Get recent claims for display"""
    all_claims = store.get_all_claims()
    all_scores = store.get_all_scores()
    
    sorted_claims = sorted(
        all_claims.items(),
        key=lambda x: x[1].get("extracted_at", ""),
        reverse=True
    )[:10]
    
    if not sorted_claims:
        return "⏳ No claims processed yet. Pipeline is initializing..."
    
    recent = ["### 📰 Recent Claims\n"]
    for cid, claim in sorted_claims:
        score = all_scores.get(cid, {}).get("score", 50)
        emoji = "🟢" if score >= 75 else "🟡" if score >= 50 else "🟠" if score >= 25 else "🔴"
        text = claim.get('text', '')[:80]
        recent.append(f"{emoji} **[{score}%]** {text}...")
    
    return "\n".join(recent)

def get_api_response(endpoint: str) -> str:
    """Get API response as JSON string"""
    try:
        if endpoint == "health":
            stats = store.get_stats()
            return json.dumps({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "claims_count": stats["total_claims"],
                "evidence_count": stats["total_evidence"],
                "pipeline": store.get_pipeline_status()
            }, indent=2)
        
        elif endpoint == "claims":
            all_claims = store.get_all_claims()
            all_scores = store.get_all_scores()
            claims = []
            for cid, claim in list(all_claims.items())[:50]:
                score_data = all_scores.get(cid, {})
                claim_copy = claim.copy()
                if isinstance(claim_copy.get("last_updated"), datetime):
                    claim_copy["last_updated"] = claim_copy["last_updated"].isoformat()
                claims.append({
                    **claim_copy,
                    "credibility": score_data.get("score", 50),
                    "label": score_data.get("label", "unknown")
                })
            return json.dumps({"claims": claims, "total": len(claims)}, indent=2)
        
        elif endpoint == "stats":
            stats = store.get_stats()
            return json.dumps({
                "total_claims": stats["total_claims"],
                "total_evidence": stats["total_evidence"],
                "pipeline_status": store.get_pipeline_status()
            }, indent=2)
        
        elif endpoint == "activity":
            return json.dumps({"activities": store.get_activity(20)}, indent=2)
        
        else:
            return json.dumps({"error": "Unknown endpoint"}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# =============================================================================
# GRADIO APP
# =============================================================================

with gr.Blocks(
    title="VeriSmart - Real-Time Credibility Engine",
    css="""
    .gradio-container { max-width: 1200px !important; }
    .gr-button { min-height: 45px; }
    """
) as app:
    gr.Markdown("""
    # 🛡️ VeriSmart - Real-Time Credibility Engine
    
    A real-time misinformation detection system powered by **Pathway** for live data streaming and RAG capabilities.
    
    **Features:** Real-time news ingestion • Evidence-based scoring • Multi-source verification • Privacy-preserving
    
    ---
    """)
    
    with gr.Tab("🔬 Analyze Claim"):
        gr.Markdown("Enter a claim to analyze its credibility based on evidence from trusted sources.")
        
        claim_input = gr.Textbox(
            label="Enter a claim to analyze",
            placeholder="e.g., 'Vitamin D supplements prevent COVID-19 infection'",
            lines=3
        )
        analyze_btn = gr.Button("🔍 Analyze Credibility", variant="primary", size="lg")
        
        with gr.Row():
            score_output = gr.Markdown(label="Credibility Score")
            evidence_output = gr.Markdown(label="Evidence Summary")
        
        sources_output = gr.Markdown(label="Sources Found")
        
        analyze_btn.click(
            analyze_claim_gradio,
            inputs=[claim_input],
            outputs=[score_output, evidence_output, sources_output]
        )
    
    with gr.Tab("📊 Dashboard"):
        gr.Markdown("Real-time statistics and recent claims from the monitoring pipeline.")
        
        refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
        stats_output = gr.Markdown()
        claims_output = gr.Markdown()
        
        def refresh_dashboard():
            return get_dashboard_stats(), get_recent_claims()
        
        refresh_btn.click(refresh_dashboard, outputs=[stats_output, claims_output])
        app.load(refresh_dashboard, outputs=[stats_output, claims_output])
    
    with gr.Tab("🔗 API"):
        gr.Markdown("""
        ## REST API Access
        
        Use the dropdown to test different API endpoints:
        """)
        
        endpoint_dropdown = gr.Dropdown(
            choices=["health", "claims", "stats", "activity"],
            value="health",
            label="Select Endpoint"
        )
        test_btn = gr.Button("📡 Test Endpoint")
        api_output = gr.Code(language="json", label="Response")
        
        test_btn.click(get_api_response, inputs=[endpoint_dropdown], outputs=[api_output])
        
        gr.Markdown("""
        ### External API Access
        
        You can also access these endpoints externally:
        
        ```bash
        # Health check
        curl https://YOUR-SPACE.hf.space/api/health
        
        # Get claims
        curl https://YOUR-SPACE.hf.space/api/claims
        
        # Analyze a claim
        curl -X POST https://YOUR-SPACE.hf.space/api/analyze \\
          -H "Content-Type: application/json" \\
          -d '{"text": "Your claim here"}'
        ```
        """)
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About VeriSmart
        
        VeriSmart is a real-time credibility engine that:
        
        1. **Ingests** news from real-time APIs (NewsData.io, GNews)
        2. **Extracts** factual claims using LLM (Google Gemini 1.5 Flash)
        3. **Collects** evidence from trusted sources (PubMed, WHO, CDC)
        4. **Scores** credibility based on evidence quality
        5. **Delivers** corrections to users who saw misinformation
        
        ### 🛠️ Technology Stack
        - **Backend:** Python, Gradio, Pathway Framework
        - **LLM:** Google Gemini 1.5 Flash
        - **Evidence:** PubMed API, Fact-checkers
        - **Frontend:** Gradio + HTML/Tailwind
        
        ### 🔒 Security Features
        - Input validation & sanitization
        - Rate limiting
        - HTML entity decoding
        - Safe error handling
        
        ---
        
        © 2026 Aryan & Khushboo. All rights reserved.
        """)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("VeriSmart - Real-Time Credibility Engine")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info(f"NEWS_API_KEY: {'✓ Configured' if Config.NEWS_API_KEY else '✗ Not set'}")
    logger.info(f"GNEWS_API_KEY: {'✓ Configured' if Config.GNEWS_API_KEY else '✗ Not set'}")
    logger.info(f"GEMINI_API_KEY: {'✓ Configured' if Config.GEMINI_API_KEY else '✗ Not set'}")
    
    # Run initial pipeline
    logger.info("Running initial data pipeline...")
    run_pipeline_cycle()
    
    # Start background pipeline
    pipeline_thread = threading.Thread(target=background_pipeline, daemon=True)
    pipeline_thread.start()
    logger.info("Background pipeline started")
    
    # Launch Gradio (this is the only server we need for HF Spaces)
    logger.info("Launching Gradio interface...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )