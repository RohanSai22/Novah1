"""
Analysis Agent - Advanced Data Synthesis and Analysis
Handles comprehensive analysis, synthesis, and insight generation from multiple data sources
"""

import asyncio
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import statistics
import hashlib

# NLP and analysis libraries
try:
    import spacy
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Data processing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisMetrics:
    """Metrics for analysis quality and depth"""
    data_sources: int
    total_data_points: int
    unique_insights: int
    confidence_score: float
    completeness_score: float
    coherence_score: float
    bias_score: float
    recency_score: float
    cross_references: int
    analysis_depth: str  # shallow, medium, deep, comprehensive

@dataclass
class InsightResult:
    """Single insight result"""
    insight: str
    evidence: List[str]
    confidence: float
    sources: List[str]
    category: str
    impact_level: str  # low, medium, high, critical
    
@dataclass
class AnalysisReport:
    """Comprehensive analysis report"""
    query: str
    executive_summary: str
    key_insights: List[InsightResult]
    data_synthesis: Dict[str, Any]
    trends_identified: List[str]
    gaps_identified: List[str]
    recommendations: List[str]
    methodology: str
    limitations: List[str]
    metrics: AnalysisMetrics
    visualizations: List[str]
    timestamp: str

class AnalysisAgent:
    """Advanced Analysis Agent for comprehensive data synthesis"""
    
    def __init__(self):
        self.name = "Analysis Agent"
        self.version = "2.0.0"
        self.capabilities = [
            "Multi-source data synthesis",
            "Trend analysis",
            "Pattern recognition",
            "Sentiment analysis",
            "Topic modeling",
            "Cross-reference validation",
            "Insight generation",
            "Gap analysis",
            "Bias detection",
            "Quality assessment"
        ]
        
        # Initialize NLP components
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.stop_words = set()
        
        if NLP_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self.stop_words = set(stopwords.words('english'))
                
                # Load spaCy model
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                    
            except Exception as e:
                logger.error(f"Error initializing NLP components: {e}")
        
        # Analysis cache
        self.analysis_cache = {}
        self.insight_templates = self._load_insight_templates()
        
    def _load_insight_templates(self) -> Dict[str, List[str]]:
        """Load templates for generating insights"""
        return {
            "trend": [
                "Analysis reveals a {direction} trend in {topic} over {timeframe}",
                "Data indicates {magnitude} {direction} movement in {metric}",
                "Significant {pattern} pattern observed across {sources} sources"
            ],
            "comparison": [
                "Comparative analysis shows {entity1} {comparison} {entity2} by {metric}",
                "{entity1} demonstrates {advantage} compared to {entity2}",
                "Key differentiator between {entity1} and {entity2} is {factor}"
            ],
            "correlation": [
                "Strong correlation identified between {variable1} and {variable2}",
                "Analysis suggests {relationship} relationship between {factor1} and {factor2}",
                "{correlation_strength} correlation observed in {context}"
            ],
            "prediction": [
                "Based on current trends, {prediction} is likely within {timeframe}",
                "Predictive analysis suggests {outcome} with {confidence}% confidence",
                "Forward-looking indicators point to {forecast}"
            ],
            "risk": [
                "Potential risk identified: {risk_factor} could impact {target}",
                "Analysis reveals {severity} risk level for {scenario}",
                "Risk mitigation should focus on {mitigation_area}"
            ]
        }
    
    async def analyze(self, context: Dict[str, Any]) -> AnalysisReport:
        """Main analysis entry point"""
        query = context.get("query", "")
        previous_results = context.get("previous_results", {})
        synthesis_mode = context.get("synthesis_mode", "comprehensive")
        
        logger.info(f"Starting {synthesis_mode} analysis for: {query}")
        
        try:
            # Extract and normalize data
            normalized_data = await self._extract_and_normalize_data(previous_results)
            
            # Perform comprehensive analysis
            analysis_results = await self._perform_comprehensive_analysis(
                query, normalized_data, synthesis_mode
            )
            
            # Generate insights
            insights = await self._generate_insights(
                query, normalized_data, analysis_results
            )
            
            # Create visualizations
            visualizations = await self._create_analysis_visualizations(
                query, normalized_data, analysis_results
            )
            
            # Calculate metrics
            metrics = self._calculate_analysis_metrics(
                normalized_data, analysis_results, insights
            )
            
            # Compile final report
            report = AnalysisReport(
                query=query,
                executive_summary=self._generate_executive_summary(query, insights, metrics),
                key_insights=insights,
                data_synthesis=analysis_results,
                trends_identified=analysis_results.get("trends", []),
                gaps_identified=analysis_results.get("gaps", []),
                recommendations=self._generate_recommendations(insights, analysis_results),
                methodology=self._describe_methodology(synthesis_mode),
                limitations=self._identify_limitations(normalized_data, analysis_results),
                metrics=metrics,
                visualizations=visualizations,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Analysis completed: {len(insights)} insights generated")
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    async def _extract_and_normalize_data(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize data from previous results"""
        normalized_data = {
            "search_results": [],
            "web_content": [],
            "structured_data": [],
            "metadata": {},
            "sources": set(),
            "timestamps": [],
            "text_content": "",
            "entities": [],
            "sentiment_data": []
        }
        
        for task_id, result in previous_results.items():
            if not result or isinstance(result, dict) and result.get("status") == "failed":
                continue
                
            # Process search results
            if "search" in task_id and isinstance(result, dict):
                if "results" in result:
                    normalized_data["search_results"].extend(result["results"])
                if "sources" in result:
                    normalized_data["sources"].update(result["sources"])
            
            # Process web content
            elif "web" in task_id and isinstance(result, dict):
                if "content" in result:
                    normalized_data["web_content"].append(result["content"])
                    normalized_data["text_content"] += f" {result['content']}"
                if "screenshots" in result:
                    normalized_data["metadata"]["screenshots"] = result["screenshots"]
            
            # Process structured data
            elif "coding" in task_id or "analysis" in task_id:
                if isinstance(result, dict):
                    normalized_data["structured_data"].append(result)
            
            # Extract entities and sentiment if available
            if isinstance(result, dict):
                if "entities" in result:
                    normalized_data["entities"].extend(result["entities"])
                if "sentiment" in result:
                    normalized_data["sentiment_data"].append(result["sentiment"])
                if "timestamp" in result:
                    normalized_data["timestamps"].append(result["timestamp"])
        
        # Process text content with NLP
        if self.nlp_model and normalized_data["text_content"]:
            doc = self.nlp_model(normalized_data["text_content"][:1000000])  # Limit text size
            normalized_data["entities"].extend([(ent.text, ent.label_) for ent in doc.ents])
        
        normalized_data["sources"] = list(normalized_data["sources"])
        
        return normalized_data
    
    async def _perform_comprehensive_analysis(
        self, query: str, data: Dict[str, Any], mode: str
    ) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        
        analysis_results = {
            "topic_analysis": {},
            "sentiment_analysis": {},
            "trend_analysis": {},
            "entity_analysis": {},
            "content_analysis": {},
            "source_analysis": {},
            "temporal_analysis": {},
            "gaps": [],
            "patterns": [],
            "anomalies": []
        }
        
        # Topic modeling and analysis
        if data["text_content"]:
            analysis_results["topic_analysis"] = await self._perform_topic_analysis(
                data["text_content"]
            )
        
        # Sentiment analysis
        if data["text_content"] and self.sentiment_analyzer:
            analysis_results["sentiment_analysis"] = self._perform_sentiment_analysis(
                data["text_content"]
            )
        
        # Entity analysis
        if data["entities"]:
            analysis_results["entity_analysis"] = self._analyze_entities(data["entities"])
        
        # Source credibility analysis
        analysis_results["source_analysis"] = self._analyze_sources(data["sources"])
        
        # Content quality analysis
        analysis_results["content_analysis"] = self._analyze_content_quality(
            data["text_content"]
        )
        
        # Temporal analysis
        if data["timestamps"]:
            analysis_results["temporal_analysis"] = self._analyze_temporal_patterns(
                data["timestamps"]
            )
        
        # Pattern detection
        analysis_results["patterns"] = self._detect_patterns(data)
        
        # Gap analysis
        analysis_results["gaps"] = self._identify_information_gaps(query, data)
        
        # Trend analysis
        analysis_results["trend_analysis"] = self._analyze_trends(data)
        
        return analysis_results
    
    async def _perform_topic_analysis(self, text_content: str) -> Dict[str, Any]:
        """Perform topic modeling and analysis"""
        try:
            # Prepare text for analysis
            sentences = sent_tokenize(text_content)
            if len(sentences) < 5:
                return {"topics": [], "keywords": [], "themes": []}
            
            # TF-IDF analysis
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top keywords
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            top_keywords = [
                (feature_names[i], tfidf_scores[i]) 
                for i in tfidf_scores.argsort()[-20:][::-1]
            ]
            
            # Topic clustering
            n_topics = min(5, len(sentences) // 3)
            if n_topics > 1:
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=10
                )
                lda.fit(tfidf_matrix)
                
                topics = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
                    topics.append({
                        "id": topic_idx,
                        "words": top_words,
                        "weight": float(topic.sum())
                    })
            else:
                topics = []
            
            return {
                "topics": topics,
                "keywords": [{"word": word, "score": float(score)} for word, score in top_keywords],
                "themes": self._extract_themes(text_content),
                "topic_count": len(topics)
            }
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            return {"topics": [], "keywords": [], "themes": []}
    
    def _perform_sentiment_analysis(self, text_content: str) -> Dict[str, Any]:
        """Perform sentiment analysis"""
        try:
            if not self.sentiment_analyzer:
                return {"overall_sentiment": "neutral", "confidence": 0.5}
            
            # Analyze overall sentiment
            scores = self.sentiment_analyzer.polarity_scores(text_content)
            
            # Analyze sentence-level sentiment
            sentences = sent_tokenize(text_content)
            sentence_sentiments = []
            
            for sentence in sentences[:50]:  # Limit to first 50 sentences
                sent_scores = self.sentiment_analyzer.polarity_scores(sentence)
                sentence_sentiments.append({
                    "sentence": sentence[:100],  # Truncate for storage
                    "sentiment": max(sent_scores, key=sent_scores.get),
                    "confidence": max(sent_scores.values())
                })
            
            # Determine overall sentiment
            compound = scores['compound']
            if compound >= 0.05:
                overall_sentiment = "positive"
            elif compound <= -0.05:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            return {
                "overall_sentiment": overall_sentiment,
                "scores": scores,
                "confidence": abs(compound),
                "sentence_analysis": sentence_sentiments,
                "sentiment_distribution": {
                    "positive": len([s for s in sentence_sentiments if s["sentiment"] == "pos"]),
                    "negative": len([s for s in sentence_sentiments if s["sentiment"] == "neg"]),
                    "neutral": len([s for s in sentence_sentiments if s["sentiment"] == "neu"])
                }
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"overall_sentiment": "neutral", "confidence": 0.5}
    
    def _analyze_entities(self, entities: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze named entities"""
        entity_counts = Counter()
        entity_types = defaultdict(list)
        
        for entity_text, entity_type in entities:
            entity_counts[entity_text] += 1
            entity_types[entity_type].append(entity_text)
        
        # Get most common entities
        top_entities = entity_counts.most_common(20)
        
        # Analyze entity types
        type_distribution = {
            entity_type: len(set(entities))  # Unique entities per type
            for entity_type, entities in entity_types.items()
        }
        
        return {
            "total_entities": len(entities),
            "unique_entities": len(entity_counts),
            "top_entities": [{"entity": entity, "count": count} for entity, count in top_entities],
            "entity_types": dict(entity_types),
            "type_distribution": type_distribution,
            "most_common_type": max(type_distribution, key=type_distribution.get) if type_distribution else None
        }
    
    def _analyze_sources(self, sources: List[str]) -> Dict[str, Any]:
        """Analyze source credibility and diversity"""
        if not sources:
            return {"source_count": 0, "diversity_score": 0.0}
        
        # Extract domains
        domains = []
        for source in sources:
            if isinstance(source, str):
                if "://" in source:
                    domain = source.split("://")[1].split("/")[0]
                else:
                    domain = source.split("/")[0]
                domains.append(domain)
        
        domain_counts = Counter(domains)
        unique_domains = len(domain_counts)
        
        # Calculate diversity score (Shannon entropy)
        total_sources = len(sources)
        diversity_score = 0.0
        if total_sources > 1:
            for count in domain_counts.values():
                p = count / total_sources
                if p > 0:
                    diversity_score -= p * np.log2(p)
            diversity_score = diversity_score / np.log2(unique_domains) if unique_domains > 1 else 0.0
        
        # Assess credibility based on domain types
        credible_indicators = ['.edu', '.gov', '.org']
        news_indicators = ['news', 'times', 'post', 'guardian', 'reuters', 'bbc']
        
        credibility_score = 0.0
        for domain in domains:
            if any(indicator in domain for indicator in credible_indicators):
                credibility_score += 0.3
            elif any(indicator in domain for indicator in news_indicators):
                credibility_score += 0.2
            else:
                credibility_score += 0.1
        
        credibility_score = min(1.0, credibility_score / len(domains))
        
        return {
            "source_count": len(sources),
            "unique_domains": unique_domains,
            "diversity_score": float(diversity_score),
            "credibility_score": float(credibility_score),
            "domain_distribution": dict(domain_counts),
            "top_domains": domain_counts.most_common(10)
        }
    
    def _analyze_content_quality(self, text_content: str) -> Dict[str, Any]:
        """Analyze content quality metrics"""
        if not text_content:
            return {"quality_score": 0.0, "readability": "unknown"}
        
        try:
            # Basic metrics
            word_count = len(text_content.split())
            sentence_count = len(sent_tokenize(text_content))
            char_count = len(text_content)
            
            # Readability metrics
            try:
                flesch_score = flesch_reading_ease(text_content)
                fk_grade = flesch_kincaid_grade(text_content)
            except:
                flesch_score = 50.0  # Average
                fk_grade = 8.0
            
            # Determine readability level
            if flesch_score >= 90:
                readability = "very_easy"
            elif flesch_score >= 80:
                readability = "easy"
            elif flesch_score >= 70:
                readability = "fairly_easy"
            elif flesch_score >= 60:
                readability = "standard"
            elif flesch_score >= 50:
                readability = "fairly_difficult"
            elif flesch_score >= 30:
                readability = "difficult"
            else:
                readability = "very_difficult"
            
            # Calculate quality score
            quality_factors = {
                "length": min(1.0, word_count / 500),  # Optimal around 500 words
                "readability": max(0.0, min(1.0, (100 - abs(flesch_score - 65)) / 100)),
                "structure": min(1.0, sentence_count / 20) if sentence_count > 0 else 0.0
            }
            
            quality_score = sum(quality_factors.values()) / len(quality_factors)
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "character_count": char_count,
                "flesch_score": float(flesch_score),
                "flesch_kincaid_grade": float(fk_grade),
                "readability": readability,
                "quality_score": float(quality_score),
                "quality_factors": quality_factors
            }
            
        except Exception as e:
            logger.error(f"Content quality analysis failed: {e}")
            return {"quality_score": 0.5, "readability": "unknown"}
    
    def _analyze_temporal_patterns(self, timestamps: List[str]) -> Dict[str, Any]:
        """Analyze temporal patterns in data"""
        if not timestamps:
            return {"recency_score": 0.5, "temporal_distribution": {}}
        
        try:
            # Parse timestamps
            parsed_times = []
            for ts in timestamps:
                try:
                    if isinstance(ts, str):
                        parsed_times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    elif isinstance(ts, datetime):
                        parsed_times.append(ts)
                except:
                    continue
            
            if not parsed_times:
                return {"recency_score": 0.5, "temporal_distribution": {}}
            
            # Calculate recency score
            now = datetime.now()
            recency_scores = []
            for ts in parsed_times:
                # Make ts timezone-naive for comparison
                if ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)
                
                days_old = (now - ts).days
                if days_old <= 1:
                    score = 1.0
                elif days_old <= 7:
                    score = 0.9
                elif days_old <= 30:
                    score = 0.7
                elif days_old <= 180:
                    score = 0.5
                else:
                    score = 0.3
                recency_scores.append(score)
            
            avg_recency = statistics.mean(recency_scores) if recency_scores else 0.5
            
            # Temporal distribution
            time_ranges = {
                "last_24h": 0,
                "last_week": 0,
                "last_month": 0,
                "last_year": 0,
                "older": 0
            }
            
            for ts in parsed_times:
                if ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)
                    
                days_old = (now - ts).days
                if days_old <= 1:
                    time_ranges["last_24h"] += 1
                elif days_old <= 7:
                    time_ranges["last_week"] += 1
                elif days_old <= 30:
                    time_ranges["last_month"] += 1
                elif days_old <= 365:
                    time_ranges["last_year"] += 1
                else:
                    time_ranges["older"] += 1
            
            return {
                "recency_score": float(avg_recency),
                "temporal_distribution": time_ranges,
                "oldest_source": min(parsed_times).isoformat() if parsed_times else None,
                "newest_source": max(parsed_times).isoformat() if parsed_times else None,
                "total_timespan_days": (max(parsed_times) - min(parsed_times)).days if len(parsed_times) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return {"recency_score": 0.5, "temporal_distribution": {}}
    
    def _detect_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect patterns in the data"""
        patterns = []
        
        # Pattern 1: Source concentration
        if data["sources"]:
            source_domains = [src.split("://")[1].split("/")[0] if "://" in src else src.split("/")[0] 
                             for src in data["sources"]]
            domain_counts = Counter(source_domains)
            
            if len(domain_counts) > 1:
                max_count = max(domain_counts.values())
                total_count = len(data["sources"])
                
                if max_count / total_count > 0.5:
                    patterns.append({
                        "type": "source_concentration",
                        "description": f"High concentration of sources from {domain_counts.most_common(1)[0][0]}",
                        "confidence": 0.8,
                        "impact": "medium"
                    })
        
        # Pattern 2: Content repetition
        if data["text_content"]:
            sentences = sent_tokenize(data["text_content"])
            if len(sentences) > 10:
                # Simple repetition detection
                sentence_hashes = [hashlib.md5(sent.lower().encode()).hexdigest() for sent in sentences]
                hash_counts = Counter(sentence_hashes)
                
                repeated_sentences = sum(1 for count in hash_counts.values() if count > 1)
                repetition_rate = repeated_sentences / len(sentences)
                
                if repetition_rate > 0.3:
                    patterns.append({
                        "type": "content_repetition",
                        "description": f"High content repetition detected ({repetition_rate:.1%})",
                        "confidence": 0.7,
                        "impact": "low"
                    })
        
        # Pattern 3: Entity clustering
        if data["entities"]:
            entity_types = defaultdict(list)
            for entity_text, entity_type in data["entities"]:
                entity_types[entity_type].append(entity_text)
            
            for entity_type, entities in entity_types.items():
                if len(entities) > 5:
                    unique_entities = len(set(entities))
                    if unique_entities < len(entities) * 0.7:  # High repetition
                        patterns.append({
                            "type": "entity_clustering",
                            "description": f"High concentration of {entity_type} entities",
                            "confidence": 0.6,
                            "impact": "medium"
                        })
        
        return patterns
    
    def _identify_information_gaps(self, query: str, data: Dict[str, Any]) -> List[str]:
        """Identify gaps in information coverage"""
        gaps = []
        
        # Check for temporal gaps
        if not data["timestamps"]:
            gaps.append("No temporal information available - cannot assess recency")
        
        # Check for source diversity
        if len(data["sources"]) < 3:
            gaps.append("Limited source diversity - only few sources consulted")
        
        # Check for content depth
        word_count = len(data["text_content"].split()) if data["text_content"] else 0
        if word_count < 200:
            gaps.append("Limited content depth - insufficient textual information")
        
        # Check for structured data
        if not data["structured_data"]:
            gaps.append("No structured data available - missing quantitative analysis")
        
        # Check for specific query terms coverage
        query_terms = set(query.lower().split())
        if data["text_content"]:
            content_terms = set(data["text_content"].lower().split())
            missing_terms = query_terms - content_terms
            if len(missing_terms) > len(query_terms) * 0.3:
                gaps.append(f"Query terms not well covered: {', '.join(list(missing_terms)[:5])}")
        
        return gaps
    
    def _analyze_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        trends = {
            "identified_trends": [],
            "trend_confidence": 0.0,
            "trend_direction": "unknown",
            "trend_strength": "weak"
        }
        
        # Temporal trends
        if data["timestamps"]:
            try:
                parsed_times = []
                for ts in data["timestamps"]:
                    try:
                        if isinstance(ts, str):
                            parsed_times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    except:
                        continue
                
                if len(parsed_times) > 2:
                    # Simple trend analysis based on publication frequency
                    time_diffs = [(parsed_times[i] - parsed_times[i-1]).days 
                                 for i in range(1, len(parsed_times))]
                    
                    if time_diffs:
                        avg_interval = statistics.mean(time_diffs)
                        if avg_interval < 7:
                            trends["identified_trends"].append("High publication frequency - trending topic")
                            trends["trend_strength"] = "strong"
                        elif avg_interval < 30:
                            trends["identified_trends"].append("Moderate publication frequency")
                            trends["trend_strength"] = "medium"
            except Exception as e:
                logger.error(f"Temporal trend analysis failed: {e}")
        
        # Entity trends
        if data["entities"]:
            entity_counts = Counter(entity[0] for entity in data["entities"])
            top_entities = entity_counts.most_common(5)
            
            if top_entities and top_entities[0][1] > 3:
                trends["identified_trends"].append(f"High focus on: {top_entities[0][0]}")
        
        # Set overall trend confidence
        trends["trend_confidence"] = min(0.8, len(trends["identified_trends"]) * 0.2)
        
        return trends
    
    def _extract_themes(self, text_content: str) -> List[str]:
        """Extract main themes from text content"""
        if not text_content:
            return []
        
        # Simple theme extraction based on frequent noun phrases
        themes = []
        
        if self.nlp_model:
            try:
                doc = self.nlp_model(text_content[:10000])  # Limit text size
                
                # Extract noun phrases
                noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                               if len(chunk.text.split()) <= 3]
                
                # Count and filter
                phrase_counts = Counter(noun_phrases)
                common_phrases = [phrase for phrase, count in phrase_counts.most_common(20) 
                                 if count > 2 and len(phrase) > 3]
                
                themes = common_phrases[:10]
                
            except Exception as e:
                logger.error(f"Theme extraction failed: {e}")
        
        # Fallback: simple keyword extraction
        if not themes:
            words = word_tokenize(text_content.lower())
            filtered_words = [word for word in words 
                             if word.isalpha() and len(word) > 4 and word not in self.stop_words]
            word_counts = Counter(filtered_words)
            themes = [word for word, count in word_counts.most_common(10) if count > 2]
        
        return themes[:10]
    
    async def _generate_insights(
        self, query: str, data: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> List[InsightResult]:
        """Generate insights from analysis results"""
        insights = []
        
        # Insight 1: Topic-based insights
        if analysis_results.get("topic_analysis", {}).get("topics"):
            topics = analysis_results["topic_analysis"]["topics"]
            for i, topic in enumerate(topics[:3]):  # Top 3 topics
                insight = InsightResult(
                    insight=f"Key theme identified: {', '.join(topic['words'][:3])}",
                    evidence=[f"Topic modeling identified {len(topic['words'])} related terms"],
                    confidence=min(0.9, topic['weight'] / max(t['weight'] for t in topics)),
                    sources=data["sources"][:3],
                    category="thematic",
                    impact_level="medium"
                )
                insights.append(insight)
        
        # Insight 2: Sentiment-based insights
        if analysis_results.get("sentiment_analysis", {}).get("overall_sentiment"):
            sentiment_data = analysis_results["sentiment_analysis"]
            sentiment = sentiment_data["overall_sentiment"]
            confidence = sentiment_data.get("confidence", 0.5)
            
            if confidence > 0.6:
                insight = InsightResult(
                    insight=f"Overall sentiment analysis reveals {sentiment} sentiment towards {query}",
                    evidence=[f"Sentiment analysis confidence: {confidence:.2%}"],
                    confidence=float(confidence),
                    sources=data["sources"][:2],
                    category="sentiment",
                    impact_level="low" if sentiment == "neutral" else "medium"
                )
                insights.append(insight)
        
        # Insight 3: Source diversity insights
        source_analysis = analysis_results.get("source_analysis", {})
        if source_analysis.get("diversity_score", 0) > 0.7:
            insight = InsightResult(
                insight=f"High source diversity detected with {source_analysis.get('unique_domains', 0)} unique domains",
                evidence=[f"Diversity score: {source_analysis.get('diversity_score', 0):.2f}"],
                confidence=0.8,
                sources=data["sources"],
                category="methodological",
                impact_level="medium"
            )
            insights.append(insight)
        elif source_analysis.get("diversity_score", 0) < 0.3:
            insight = InsightResult(
                insight="Limited source diversity may indicate information bias",
                evidence=[f"Only {source_analysis.get('unique_domains', 0)} unique domains consulted"],
                confidence=0.7,
                sources=data["sources"],
                category="methodological",
                impact_level="high"
            )
            insights.append(insight)
        
        # Insight 4: Temporal insights
        temporal_analysis = analysis_results.get("temporal_analysis", {})
        recency_score = temporal_analysis.get("recency_score", 0.5)
        
        if recency_score > 0.8:
            insight = InsightResult(
                insight="Information sources are very recent and up-to-date",
                evidence=[f"Average recency score: {recency_score:.2%}"],
                confidence=0.9,
                sources=data["sources"][:2],
                category="temporal",
                impact_level="medium"
            )
            insights.append(insight)
        elif recency_score < 0.4:
            insight = InsightResult(
                insight="Information sources may be outdated",
                evidence=[f"Low recency score: {recency_score:.2%}"],
                confidence=0.8,
                sources=data["sources"][:2],
                category="temporal",
                impact_level="high"
            )
            insights.append(insight)
        
        # Insight 5: Entity-based insights
        entity_analysis = analysis_results.get("entity_analysis", {})
        if entity_analysis.get("top_entities"):
            top_entity = entity_analysis["top_entities"][0]
            insight = InsightResult(
                insight=f"Most prominent entity: '{top_entity['entity']}' (mentioned {top_entity['count']} times)",
                evidence=[f"Entity frequency analysis across {entity_analysis.get('total_entities', 0)} mentions"],
                confidence=0.7,
                sources=data["sources"][:3],
                category="entity",
                impact_level="medium"
            )
            insights.append(insight)
        
        # Insight 6: Pattern-based insights
        patterns = analysis_results.get("patterns", [])
        for pattern in patterns[:2]:  # Top 2 patterns
            insight = InsightResult(
                insight=pattern["description"],
                evidence=[f"Pattern detected with {pattern['confidence']:.1%} confidence"],
                confidence=float(pattern["confidence"]),
                sources=data["sources"][:2],
                category="pattern",
                impact_level=pattern["impact"]
            )
            insights.append(insight)
        
        # Sort insights by confidence and impact
        insights.sort(key=lambda x: (
            {"high": 3, "medium": 2, "low": 1, "critical": 4}.get(x.impact_level, 1),
            x.confidence
        ), reverse=True)
        
        return insights[:10]  # Return top 10 insights
    
    async def _create_analysis_visualizations(
        self, query: str, data: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Create visualizations for analysis results"""
        visualizations = []
        
        try:
            # Create output directory
            viz_dir = "analysis_visualizations"
            import os
            os.makedirs(viz_dir, exist_ok=True)
            
            # Visualization 1: Source Distribution
            if data["sources"]:
                source_domains = [src.split("://")[1].split("/")[0] if "://" in src else src.split("/")[0] 
                                 for src in data["sources"]]
                domain_counts = Counter(source_domains)
                
                plt.figure(figsize=(10, 6))
                domains, counts = zip(*domain_counts.most_common(10))
                plt.bar(domains, counts)
                plt.title("Source Distribution")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                viz_path = f"{viz_dir}/source_distribution.png"
                plt.savefig(viz_path)
                plt.close()
                visualizations.append(viz_path)
            
            # Visualization 2: Sentiment Distribution
            sentiment_analysis = analysis_results.get("sentiment_analysis", {})
            if sentiment_analysis.get("sentiment_distribution"):
                sent_dist = sentiment_analysis["sentiment_distribution"]
                
                plt.figure(figsize=(8, 6))
                labels = list(sent_dist.keys())
                sizes = list(sent_dist.values())
                plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                plt.title("Sentiment Distribution")
                
                viz_path = f"{viz_dir}/sentiment_distribution.png"
                plt.savefig(viz_path)
                plt.close()
                visualizations.append(viz_path)
            
            # Visualization 3: Word Cloud
            if data["text_content"]:
                try:
                    wordcloud = WordCloud(
                        width=800, height=400, 
                        background_color='white',
                        max_words=100
                    ).generate(data["text_content"])
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title("Key Terms Word Cloud")
                    
                    viz_path = f"{viz_dir}/wordcloud.png"
                    plt.savefig(viz_path, bbox_inches='tight')
                    plt.close()
                    visualizations.append(viz_path)
                    
                except Exception as e:
                    logger.error(f"Word cloud generation failed: {e}")
            
            # Visualization 4: Entity Type Distribution
            entity_analysis = analysis_results.get("entity_analysis", {})
            if entity_analysis.get("type_distribution"):
                type_dist = entity_analysis["type_distribution"]
                
                plt.figure(figsize=(10, 6))
                types, counts = zip(*list(type_dist.items())[:10])
                plt.bar(types, counts)
                plt.title("Entity Type Distribution")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                viz_path = f"{viz_dir}/entity_types.png"
                plt.savefig(viz_path)
                plt.close()
                visualizations.append(viz_path)
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
        
        return visualizations
    
    def _calculate_analysis_metrics(
        self, data: Dict[str, Any], analysis_results: Dict[str, Any], insights: List[InsightResult]
    ) -> AnalysisMetrics:
        """Calculate comprehensive analysis metrics"""
        
        # Count data sources and points
        data_sources = len(data["sources"])
        total_data_points = (
            len(data["search_results"]) +
            len(data["web_content"]) +
            len(data["structured_data"]) +
            len(data["entities"])
        )
        
        # Calculate confidence score
        if insights:
            confidence_score = statistics.mean([insight.confidence for insight in insights])
        else:
            confidence_score = 0.5
        
        # Calculate completeness score
        completeness_factors = [
            1.0 if data["text_content"] else 0.0,
            1.0 if data["sources"] else 0.0,
            1.0 if data["entities"] else 0.0,
            1.0 if analysis_results.get("sentiment_analysis") else 0.0,
            1.0 if analysis_results.get("topic_analysis") else 0.0
        ]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # Calculate coherence score
        coherence_factors = [
            analysis_results.get("content_analysis", {}).get("quality_score", 0.5),
            1.0 - (len(analysis_results.get("gaps", [])) * 0.2),  # Fewer gaps = higher coherence
            min(1.0, len(insights) * 0.1)  # More insights = higher coherence
        ]
        coherence_score = max(0.0, sum(coherence_factors) / len(coherence_factors))
        
        # Calculate bias score (lower is better)
        source_analysis = analysis_results.get("source_analysis", {})
        diversity_score = source_analysis.get("diversity_score", 0.5)
        bias_score = 1.0 - diversity_score  # High diversity = low bias
        
        # Calculate recency score
        temporal_analysis = analysis_results.get("temporal_analysis", {})
        recency_score = temporal_analysis.get("recency_score", 0.5)
        
        # Count cross-references (simplified)
        cross_references = max(0, data_sources - 1)  # Each additional source is a cross-reference
        
        # Determine analysis depth
        if total_data_points >= 50 and len(insights) >= 8:
            analysis_depth = "comprehensive"
        elif total_data_points >= 20 and len(insights) >= 5:
            analysis_depth = "deep"
        elif total_data_points >= 10 and len(insights) >= 3:
            analysis_depth = "medium"
        else:
            analysis_depth = "shallow"
        
        return AnalysisMetrics(
            data_sources=data_sources,
            total_data_points=total_data_points,
            unique_insights=len(insights),
            confidence_score=float(confidence_score),
            completeness_score=float(completeness_score),
            coherence_score=float(coherence_score),
            bias_score=float(bias_score),
            recency_score=float(recency_score),
            cross_references=cross_references,
            analysis_depth=analysis_depth
        )
    
    def _generate_executive_summary(
        self, query: str, insights: List[InsightResult], metrics: AnalysisMetrics
    ) -> str:
        """Generate executive summary"""
        
        high_impact_insights = [i for i in insights if i.impact_level in ["high", "critical"]]
        
        summary_parts = [
            f"Comprehensive analysis of '{query}' based on {metrics.data_sources} sources and {metrics.total_data_points} data points.",
            f"Analysis depth: {metrics.analysis_depth.title()}.",
            f"Generated {metrics.unique_insights} unique insights with {metrics.confidence_score:.1%} average confidence."
        ]
        
        if high_impact_insights:
            summary_parts.append(f"Key findings: {high_impact_insights[0].insight}")
        
        if metrics.bias_score > 0.5:
            summary_parts.append("Note: Some potential bias detected in source selection.")
        
        if metrics.recency_score < 0.5:
            summary_parts.append("Note: Some information sources may be outdated.")
        
        return " ".join(summary_parts)
    
    def _generate_recommendations(
        self, insights: List[InsightResult], analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Source diversity recommendations
        source_analysis = analysis_results.get("source_analysis", {})
        if source_analysis.get("diversity_score", 0) < 0.5:
            recommendations.append("Increase source diversity by consulting additional domain types")
        
        # Temporal recommendations
        temporal_analysis = analysis_results.get("temporal_analysis", {})
        if temporal_analysis.get("recency_score", 0) < 0.6:
            recommendations.append("Seek more recent sources to ensure current information")
        
        # Gap-based recommendations
        gaps = analysis_results.get("gaps", [])
        for gap in gaps[:3]:
            recommendations.append(f"Address information gap: {gap}")
        
        # Insight-based recommendations
        high_confidence_insights = [i for i in insights if i.confidence > 0.8]
        if high_confidence_insights:
            recommendations.append(f"Focus investigation on: {high_confidence_insights[0].insight}")
        
        # Quality recommendations
        content_analysis = analysis_results.get("content_analysis", {})
        if content_analysis.get("quality_score", 0) < 0.6:
            recommendations.append("Seek higher quality, more comprehensive sources")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _describe_methodology(self, synthesis_mode: str) -> str:
        """Describe the analysis methodology used"""
        base_methods = [
            "Multi-source data extraction and normalization",
            "Natural language processing and entity recognition",
            "Topic modeling and thematic analysis",
            "Sentiment analysis and bias detection",
            "Temporal pattern analysis",
            "Source credibility assessment"
        ]
        
        if synthesis_mode == "comprehensive":
            additional_methods = [
                "Advanced statistical analysis",
                "Cross-reference validation",
                "Deep insight generation",
                "Quality metrics calculation"
            ]
            base_methods.extend(additional_methods)
        
        return f"Analysis methodology: {'; '.join(base_methods)}."
    
    def _identify_limitations(
        self, data: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Identify analysis limitations"""
        limitations = []
        
        # Data limitations
        if len(data["sources"]) < 5:
            limitations.append("Limited number of sources may affect comprehensiveness")
        
        if not data["text_content"] or len(data["text_content"].split()) < 500:
            limitations.append("Limited textual content may affect analysis depth")
        
        # Temporal limitations
        temporal_analysis = analysis_results.get("temporal_analysis", {})
        if temporal_analysis.get("recency_score", 0) < 0.5:
            limitations.append("Older sources may not reflect current state")
        
        # Methodological limitations
        if not NLP_AVAILABLE:
            limitations.append("Advanced NLP analysis limited due to missing dependencies")
        
        # Bias limitations
        source_analysis = analysis_results.get("source_analysis", {})
        if source_analysis.get("diversity_score", 0) < 0.4:
            limitations.append("Low source diversity may introduce selection bias")
        
        # Language limitations
        limitations.append("Analysis primarily designed for English-language content")
        
        return limitations[:5]  # Return top 5 limitations


# Example usage
if __name__ == "__main__":
    async def test_analysis_agent():
        """Test the analysis agent"""
        agent = AnalysisAgent()
        
        # Mock data for testing
        mock_context = {
            "query": "artificial intelligence trends",
            "previous_results": {
                "search_001": {
                    "results": [
                        {"title": "AI Trends 2024", "url": "https://example.com/ai-trends", "snippet": "AI is transforming industries..."},
                        {"title": "Machine Learning Advances", "url": "https://tech.com/ml", "snippet": "Recent ML breakthroughs..."}
                    ],
                    "sources": ["example.com", "tech.com"],
                    "timestamp": datetime.now().isoformat()
                },
                "web_001": {
                    "content": "Artificial intelligence continues to evolve rapidly. Machine learning models are becoming more sophisticated. Natural language processing has made significant strides. Computer vision applications are expanding across industries.",
                    "screenshots": ["screenshot1.png"],
                    "timestamp": datetime.now().isoformat()
                }
            },
            "synthesis_mode": "comprehensive"
        }
        
        try:
            report = await agent.analyze(mock_context)
            
            print("=== Analysis Report ===")
            print(f"Query: {report.query}")
            print(f"Executive Summary: {report.executive_summary}")
            print(f"\nKey Insights ({len(report.key_insights)}):")
            for i, insight in enumerate(report.key_insights[:3], 1):
                print(f"{i}. {insight.insight} (Confidence: {insight.confidence:.1%})")
            
            print(f"\nMetrics:")
            print(f"- Analysis Depth: {report.metrics.analysis_depth}")
            print(f"- Confidence Score: {report.metrics.confidence_score:.1%}")
            print(f"- Quality Score: {report.metrics.completeness_score:.1%}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    import asyncio
    asyncio.run(test_analysis_agent())
