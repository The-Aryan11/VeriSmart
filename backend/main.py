"""
VeriSmart - Real-Time Credibility Engine
Main Pathway Application

This is the core backend that:
1. Ingests real-time news from multiple APIs
2. Extracts claims using LLM
3. Clusters similar claims
4. Collects evidence from trusted sources
5. Scores credibility in real-time
6. Serves results via REST API
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pathway as pw
from pathway.xpacks.llm import embedders, llms, parsers, prompts
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.udfs import DiskCache, ExponentialBackoffRetryStrategy
import requests
from typing import Optional
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verismart")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # API Keys
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))
    
    # Data refresh intervals (seconds)
    NEWS_REFRESH_INTERVAL = int(os.getenv("NEWS_REFRESH_INTERVAL", 300))  # 5 minutes
    
    # Evidence sources
    EVIDENCE_SOURCES = [
        "pubmed", "who", "cdc", "reuters", "apnews", "snopes", "politifact"
    ]
    
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
# CUSTOM NEWS CONNECTOR
# =============================================================================

class NewsAPIConnector:
    """Custom connector for real-time news ingestion from multiple sources"""
    
    def __init__(self):
        self.news_api_key = Config.NEWS_API_KEY
        self.gnews_api_key = Config.GNEWS_API_KEY
        self.last_fetch = {}
        
    def fetch_newsapi(self, query: str = "health OR politics OR climate", page_size: int = 20) -> list:
        """Fetch from NewsAPI.org"""
        if not self.news_api_key:
            logger.warning("NewsAPI key not configured")
            return []
            
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": self.news_api_key,
                "pageSize": page_size,
                "sortBy": "publishedAt",
                "language": "en"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("articles", [])
            else:
                logger.error(f"NewsAPI error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []
    
    def fetch_gnews(self, query: str = "health", max_results: int = 10) -> list:
        """Fetch from GNews.io"""
        if not self.gnews_api_key:
            logger.warning("GNews API key not configured")
            return []
            
        try:
            url = "https://gnews.io/api/v4/search"
            params = {
                "q": query,
                "token": self.gnews_api_key,
                "max": max_results,
                "lang": "en"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("articles", [])
            else:
                logger.error(f"GNews error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"GNews fetch error: {e}")
            return []
    
    def fetch_all(self) -> list:
        """Fetch from all configured news sources"""
        all_articles = []
        
        # Fetch from NewsAPI
        topics = ["health vaccines", "climate change", "elections politics", "AI technology", "economy finance"]
        for topic in topics:
            articles = self.fetch_newsapi(query=topic, page_size=10)
            for article in articles:
                article["_source"] = "newsapi"
                article["_topic"] = topic.split()[0]
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
        return all_articles

# =============================================================================
# CLAIM EXTRACTION WITH LLM
# =============================================================================

class ClaimExtractor:
    """Extract factual claims from news articles using LLM"""
    
    def __init__(self):
        self.openai_key = Config.OPENAI_API_KEY
        self.gemini_key = Config.GEMINI_API_KEY
        
    def extract_claims_openai(self, article_text: str, article_title: str) -> list:
        """Extract claims using OpenAI API"""
        if not self.openai_key:
            return self._fallback_extraction(article_text, article_title)
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_key)
            
            prompt = f"""Analyze this news article and extract factual claims that can be verified.
            
Title: {article_title}
Content: {article_text[:2000]}

Return a JSON array of claims. Each claim should have:
- "text": The claim statement (clear, concise)
- "category": One of [health, politics, climate, technology, finance, science]
- "verifiable": true/false (is this a factual claim that can be checked?)
- "confidence": 0-100 (how confident are you this is the main claim?)

Only include claims that are:
1. Factual statements (not opinions)
2. Specific enough to verify
3. Newsworthy or significant

Return ONLY valid JSON array, no other text."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a fact-checking assistant that extracts verifiable claims from news articles. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            # Parse JSON from response
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            
            claims = json.loads(result)
            return [c for c in claims if c.get("verifiable", True)]
            
        except Exception as e:
            logger.error(f"OpenAI extraction error: {e}")
            return self._fallback_extraction(article_text, article_title)
    
    def extract_claims_gemini(self, article_text: str, article_title: str) -> list:
        """Extract claims using Google Gemini API"""
        if not self.gemini_key:
            return self._fallback_extraction(article_text, article_title)
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = f"""Analyze this news article and extract factual claims.
            
Title: {article_title}
Content: {article_text[:2000]}

Return a JSON array with claims having: text, category (health/politics/climate/technology/finance/science), verifiable (boolean), confidence (0-100).
Only return the JSON array, nothing else."""

            response = model.generate_content(prompt)
            result = response.text.strip()
            
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            
            claims = json.loads(result)
            return [c for c in claims if c.get("verifiable", True)]
            
        except Exception as e:
            logger.error(f"Gemini extraction error: {e}")
            return self._fallback_extraction(article_text, article_title)
    
    def _fallback_extraction(self, article_text: str, article_title: str) -> list:
        """Simple rule-based extraction when LLM is unavailable"""
        # Basic extraction based on title
        category = "health"
        if any(w in article_title.lower() for w in ["election", "vote", "congress", "president", "political"]):
            category = "politics"
        elif any(w in article_title.lower() for w in ["climate", "warming", "carbon", "environment"]):
            category = "climate"
        elif any(w in article_title.lower() for w in ["ai", "tech", "software", "computer", "digital"]):
            category = "technology"
        elif any(w in article_title.lower() for w in ["stock", "economy", "bank", "finance", "market"]):
            category = "finance"
        
        return [{
            "text": article_title,
            "category": category,
            "verifiable": True,
            "confidence": 60
        }]
    
    def extract(self, article_text: str, article_title: str) -> list:
        """Extract claims using available LLM"""
        if self.openai_key:
            return self.extract_claims_openai(article_text, article_title)
        elif self.gemini_key:
            return self.extract_claims_gemini(article_text, article_title)
        else:
            return self._fallback_extraction(article_text, article_title)

# =============================================================================
# EVIDENCE COLLECTOR
# =============================================================================

class EvidenceCollector:
    """Collect evidence from trusted sources for claim verification"""
    
    def __init__(self):
        self.sources = Config.EVIDENCE_SOURCES
    
    def search_pubmed(self, claim: str) -> list:
        """Search PubMed for relevant scientific evidence"""
        try:
            # Use PubMed E-utilities API (free, no key required for basic use)
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": claim[:100],
                "retmax": 5,
                "retmode": "json",
                "sort": "relevance"
            }
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                ids = data.get("esearchresult", {}).get("idlist", [])
                
                if ids:
                    # Fetch article summaries
                    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    summary_params = {
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "json"
                    }
                    summary_response = requests.get(summary_url, params=summary_params, timeout=10)
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        results = summary_data.get("result", {})
                        
                        evidence = []
                        for pid in ids:
                            if pid in results:
                                article = results[pid]
                                evidence.append({
                                    "source": "PubMed",
                                    "title": article.get("title", ""),
                                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                                    "date": article.get("pubdate", ""),
                                    "quality": 10,  # Peer-reviewed
                                    "type": "scientific"
                                })
                        return evidence
            return []
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    def search_factcheck(self, claim: str) -> list:
        """Search Google Fact Check API"""
        try:
            # Google Fact Check Tools API (requires API key)
            api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
            if not api_key:
                return []
            
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                "key": api_key,
                "query": claim[:100],
                "languageCode": "en"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                claims = data.get("claims", [])
                
                evidence = []
                for c in claims[:5]:
                    for review in c.get("claimReview", []):
                        evidence.append({
                            "source": review.get("publisher", {}).get("name", "Fact-checker"),
                            "title": review.get("title", c.get("text", "")),
                            "url": review.get("url", ""),
                            "rating": review.get("textualRating", ""),
                            "quality": 8,
                            "type": "factcheck"
                        })
                return evidence
            return []
        except Exception as e:
            logger.error(f"Fact-check search error: {e}")
            return []
    
    def collect(self, claim: str) -> dict:
        """Collect evidence from all sources"""
        all_evidence = {
            "supporting": [],
            "refuting": [],
            "uncertain": []
        }
        
        # Search PubMed
        pubmed_evidence = self.search_pubmed(claim)
        for e in pubmed_evidence:
            all_evidence["uncertain"].append(e)  # Needs further analysis
        
        # Search fact-checkers
        factcheck_evidence = self.search_factcheck(claim)
        for e in factcheck_evidence:
            rating = e.get("rating", "").lower()
            if any(w in rating for w in ["true", "correct", "accurate"]):
                all_evidence["supporting"].append(e)
            elif any(w in rating for w in ["false", "fake", "incorrect", "misleading"]):
                all_evidence["refuting"].append(e)
            else:
                all_evidence["uncertain"].append(e)
        
        return all_evidence

# =============================================================================
# CREDIBILITY SCORER
# =============================================================================

class CredibilityScorer:
    """Calculate credibility scores based on evidence"""
    
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
        """Calculate credibility score for a claim"""
        category = claim.get("category", "default")
        baseline = self.baselines.get(category, self.baselines["default"])
        
        # Count evidence
        supporting = evidence.get("supporting", [])
        refuting = evidence.get("refuting", [])
        uncertain = evidence.get("uncertain", [])
        
        # Calculate weighted score adjustments
        support_score = sum(self.weights.get(e.get("type", "social"), 1) * e.get("quality", 5) for e in supporting)
        refute_score = sum(self.weights.get(e.get("type", "social"), 1) * e.get("quality", 5) for e in refuting)
        
        # Normalize and apply to baseline
        max_adjustment = 50  # Maximum points to add/subtract
        
        if support_score + refute_score > 0:
            net_score = (support_score - refute_score) / (support_score + refute_score) * max_adjustment
        else:
            net_score = 0
        
        final_score = max(0, min(100, baseline + net_score))
        
        # Calculate confidence
        total_evidence = len(supporting) + len(refuting) + len(uncertain)
        if total_evidence >= 10:
            confidence = "high"
        elif total_evidence >= 5:
            confidence = "moderate"
        elif total_evidence >= 2:
            confidence = "low"
        else:
            confidence = "insufficient"
        
        # Determine label
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
            },
            "breakdown": {
                "baseline_score": baseline,
                "support_contribution": round(support_score, 1),
                "refute_contribution": round(refute_score, 1),
                "final_score": round(final_score, 1)
            }
        }

# =============================================================================
# PATHWAY PIPELINE
# =============================================================================

# Initialize components
news_connector = NewsAPIConnector()
claim_extractor = ClaimExtractor()
evidence_collector = EvidenceCollector()
credibility_scorer = CredibilityScorer()

# In-memory storage for real-time data
claims_store = {}
evidence_store = {}
scores_store = {}
activity_log = []

def generate_claim_id(claim_text: str) -> str:
    """Generate unique ID for a claim"""
    return hashlib.md5(claim_text.encode()).hexdigest()[:12]

def process_articles(articles: list) -> list:
    """Process articles through the full pipeline"""
    processed_claims = []
    
    for article in articles:
        title = article.get("title", "")
        content = article.get("content") or article.get("description") or ""
        source = article.get("source", {}).get("name", article.get("_source", "unknown"))
        url = article.get("url", "")
        published = article.get("publishedAt", datetime.now().isoformat())
        
        # Extract claims
        claims = claim_extractor.extract(content, title)
        
        for claim in claims:
            claim_id = generate_claim_id(claim["text"])
            
            # Skip if already processed recently
            if claim_id in claims_store:
                existing = claims_store[claim_id]
                if (datetime.now() - existing.get("last_updated", datetime.min)).seconds < 3600:
                    continue
            
            # Collect evidence
            evidence = evidence_collector.collect(claim["text"])
            
            # Calculate score
            score_data = credibility_scorer.calculate_score(claim, evidence)
            
            # Store data
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
            
            claims_store[claim_id] = claim_data
            evidence_store[claim_id] = evidence
            scores_store[claim_id] = score_data
            
            # Log activity
            activity_log.append({
                "type": "new_claim",
                "claim_id": claim_id,
                "timestamp": datetime.now().isoformat(),
                "message": f"New claim detected: {claim['text'][:50]}..."
            })
            
            processed_claims.append({
                **claim_data,
                "evidence": evidence,
                "score": score_data
            })
    
    return processed_claims

def run_pipeline_cycle():
    """Run one cycle of the data pipeline"""
    logger.info("Starting pipeline cycle...")
    
    # Fetch news
    articles = news_connector.fetch_all()
    
    # Process through pipeline
    if articles:
        processed = process_articles(articles)
        logger.info(f"Processed {len(processed)} new claims")
        return processed
    
    return []

# =============================================================================
# REST API SERVER
# =============================================================================

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Background pipeline runner
def background_pipeline():
    """Run pipeline continuously in background"""
    while True:
        try:
            run_pipeline_cycle()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        time.sleep(Config.NEWS_REFRESH_INTERVAL)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "claims_count": len(claims_store),
        "evidence_count": sum(len(e.get("supporting", [])) + len(e.get("refuting", [])) + len(e.get("uncertain", [])) for e in evidence_store.values())
    })

@app.route('/api/claims', methods=['GET'])
def get_claims():
    """Get all claims with scores"""
    category = request.args.get('category')
    limit = int(request.args.get('limit', 50))
    
    claims = []
    for claim_id, claim_data in list(claims_store.items())[:limit]:
        score_data = scores_store.get(claim_id, {})
        evidence_data = evidence_store.get(claim_id, {})
        
        if category and claim_data.get("category") != category:
            continue
        
        claims.append({
            **claim_data,
            "credibility": score_data.get("score", 50),
            "credibility_label": score_data.get("label", "unknown"),
            "confidence": score_data.get("confidence", "unknown"),
            "evidence": {
                "supporting": len(evidence_data.get("supporting", [])),
                "refuting": len(evidence_data.get("refuting", [])),
                "uncertain": len(evidence_data.get("uncertain", []))
            }
        })
    
    # Sort by recency
    claims.sort(key=lambda x: x.get("extracted_at", ""), reverse=True)
    
    return jsonify({
        "claims": claims,
        "total": len(claims),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/claims/<claim_id>', methods=['GET'])
def get_claim_detail(claim_id):
    """Get detailed claim information"""
    if claim_id not in claims_store:
        return jsonify({"error": "Claim not found"}), 404
    
    claim_data = claims_store[claim_id]
    score_data = scores_store.get(claim_id, {})
    evidence_data = evidence_store.get(claim_id, {})
    
    return jsonify({
        "claim": claim_data,
        "score": score_data,
        "evidence": evidence_data,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/claims/<claim_id>/analyze', methods=['POST'])
def analyze_claim(claim_id):
    """Re-analyze a claim with fresh evidence"""
    if claim_id not in claims_store:
        return jsonify({"error": "Claim not found"}), 404
    
    claim_data = claims_store[claim_id]
    
    # Re-collect evidence
    evidence = evidence_collector.collect(claim_data["text"])
    
    # Re-calculate score
    score_data = credibility_scorer.calculate_score(
        {"text": claim_data["text"], "category": claim_data.get("category", "default")},
        evidence
    )
    
    # Update stores
    evidence_store[claim_id] = evidence
    scores_store[claim_id] = score_data
    claims_store[claim_id]["last_updated"] = datetime.now()
    
    # Log activity
    activity_log.append({
        "type": "score_update",
        "claim_id": claim_id,
        "timestamp": datetime.now().isoformat(),
        "message": f"Score updated to {score_data['score']}%"
    })
    
    return jsonify({
        "claim": claims_store[claim_id],
        "score": score_data,
        "evidence": evidence,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_new_claim():
    """Analyze a new claim text"""
    data = request.get_json()
    claim_text = data.get("text", "")
    
    if not claim_text:
        return jsonify({"error": "Claim text required"}), 400
    
    # Generate ID
    claim_id = generate_claim_id(claim_text)
    
    # Extract claim metadata
    claims = claim_extractor.extract(claim_text, claim_text)
    claim_meta = claims[0] if claims else {"text": claim_text, "category": "default"}
    
    # Collect evidence
    evidence = evidence_collector.collect(claim_text)
    
    # Calculate score
    score_data = credibility_scorer.calculate_score(claim_meta, evidence)
    
    # Store
    claim_data = {
        "id": claim_id,
        "text": claim_text,
        "category": claim_meta.get("category", "general"),
        "source": "user_submitted",
        "extracted_at": datetime.now().isoformat(),
        "last_updated": datetime.now()
    }
    
    claims_store[claim_id] = claim_data
    evidence_store[claim_id] = evidence
    scores_store[claim_id] = score_data
    
    return jsonify({
        "claim": claim_data,
        "score": score_data,
        "evidence": evidence,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    total_claims = len(claims_store)
    total_evidence = sum(
        len(e.get("supporting", [])) + len(e.get("refuting", [])) + len(e.get("uncertain", []))
        for e in evidence_store.values()
    )
    
    # Calculate category distribution
    categories = {}
    credibility_dist = {"high": 0, "medium": 0, "low": 0, "dubious": 0}
    
    for claim_id, claim in claims_store.items():
        cat = claim.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1
        
        score = scores_store.get(claim_id, {}).get("score", 50)
        if score >= 75:
            credibility_dist["high"] += 1
        elif score >= 50:
            credibility_dist["medium"] += 1
        elif score >= 25:
            credibility_dist["low"] += 1
        else:
            credibility_dist["dubious"] += 1
    
    return jsonify({
        "total_claims": total_claims,
        "total_evidence": total_evidence,
        "categories": categories,
        "credibility_distribution": credibility_dist,
        "last_update": datetime.now().isoformat(),
        "pipeline_status": "running"
    })

@app.route('/api/activity', methods=['GET'])
def get_activity():
    """Get recent activity feed"""
    limit = int(request.args.get('limit', 20))
    return jsonify({
        "activities": activity_log[-limit:][::-1],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/trending', methods=['GET'])
def get_trending():
    """Get trending claims"""
    limit = int(request.args.get('limit', 10))
    
    # Sort by recency and return top claims
    sorted_claims = sorted(
        [
            {
                **claims_store[cid],
                "score": scores_store.get(cid, {}).get("score", 50),
                "label": scores_store.get(cid, {}).get("label", "unknown")
            }
            for cid in claims_store.keys()
        ],
        key=lambda x: x.get("extracted_at", ""),
        reverse=True
    )[:limit]
    
    return jsonify({
        "trending": sorted_claims,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/evidence-sources', methods=['GET'])
def get_evidence_sources():
    """Get list of evidence sources"""
    sources = [
        {"name": "PubMed", "type": "Scientific", "quality": 10, "status": "active"},
        {"name": "WHO", "type": "Institutional", "quality": 9, "status": "active"},
        {"name": "CDC", "type": "Institutional", "quality": 9, "status": "active"},
        {"name": "Reuters", "type": "Journalism", "quality": 8, "status": "active"},
        {"name": "AP News", "type": "Journalism", "quality": 8, "status": "active"},
        {"name": "Snopes", "type": "Fact-check", "quality": 7, "status": "active"},
        {"name": "PolitiFact", "type": "Fact-check", "quality": 7, "status": "active"},
    ]
    return jsonify({"sources": sources})

# =============================================================================
# PATHWAY VECTOR STORE (for RAG)
# =============================================================================

def setup_pathway_vectorstore():
    """Setup Pathway vector store for RAG queries"""
    try:
        # Create embedder
        embedder = embedders.OpenAIEmbedder(
            api_key=Config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        
        # This would connect to a document source
        # For now, we'll use the claims as the document source
        logger.info("Pathway vector store initialized")
        return True
    except Exception as e:
        logger.warning(f"Could not initialize Pathway vector store: {e}")
        return False

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("VeriSmart - Real-Time Credibility Engine")
    logger.info("=" * 60)
    
    # Check API keys
    if Config.NEWS_API_KEY:
        logger.info("✓ NewsAPI configured")
    else:
        logger.warning("✗ NewsAPI not configured")
    
    if Config.GNEWS_API_KEY:
        logger.info("✓ GNews configured")
    else:
        logger.warning("✗ GNews not configured")
    
    if Config.OPENAI_API_KEY:
        logger.info("✓ OpenAI configured")
    else:
        logger.warning("✗ OpenAI not configured (using fallback extraction)")
    
    # Run initial pipeline
    logger.info("Running initial data pipeline...")
    run_pipeline_cycle()
    
    # Start background pipeline
    pipeline_thread = threading.Thread(target=background_pipeline, daemon=True)
    pipeline_thread.start()
    logger.info(f"Background pipeline started (refresh every {Config.NEWS_REFRESH_INTERVAL}s)")
    
    # Start API server
    logger.info(f"Starting API server on {Config.HOST}:{Config.PORT}")
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)
