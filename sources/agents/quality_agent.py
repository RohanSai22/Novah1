"""
Quality Agent - Advanced Quality Validation and Verification System
This agent validates results, fact-checks information, and ensures quality standards
"""

import asyncio
import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics
from urllib.parse import urlparse
import difflib

from sources.utility import pretty_print, animate_thinking
from sources.logger import Logger

@dataclass
class QualityMetric:
    """Individual quality metric assessment"""
    metric_name: str
    score: float  # 0.0 to 1.0
    confidence: float
    details: str
    evidence: List[str]
    weight: float = 1.0

@dataclass
class SourceCredibility:
    """Source credibility assessment"""
    url: str
    domain: str
    credibility_score: float
    reputation_factors: Dict[str, float]
    content_quality: float
    freshness_score: float
    authority_indicators: List[str]

@dataclass
class FactCheckResult:
    """Fact-checking result for a specific claim"""
    claim: str
    verification_status: str  # verified, disputed, unverified, false
    confidence: float
    supporting_sources: List[str]
    contradicting_sources: List[str]
    context: str

@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_quality_score: float
    metrics: List[QualityMetric]
    source_credibility: List[SourceCredibility]
    fact_check_results: List[FactCheckResult]
    bias_analysis: Dict[str, Any]
    completeness_assessment: Dict[str, Any]
    accuracy_indicators: Dict[str, Any]
    recency_analysis: Dict[str, Any]
    recommendations: List[str]
    limitations: List[str]
    confidence_interval: Tuple[float, float]
    timestamp: datetime

class QualityAgent:
    """
    Advanced Quality Agent for comprehensive validation and verification
    """
    
    def __init__(self):
        self.logger = Logger("quality_agent.log")
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.75,
            'acceptable': 0.6,
            'poor': 0.4,
            'unacceptable': 0.0
        }
        
        # Credible domains (simplified list - in production, use comprehensive database)
        self.credible_domains = {
            'high_credibility': {
                'academic.edu', 'gov', 'ieee.org', 'nature.com', 'science.org',
                'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com', 'arxiv.org'
            },
            'medium_credibility': {
                'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'economist.com',
                'bbc.com', 'npr.org', 'pbs.org', 'cnn.com', 'nytimes.com'
            },
            'business_credible': {
                'sec.gov', 'federalreserve.gov', 'worldbank.org', 'imf.org',
                'fortune.com', 'forbes.com', 'marketwatch.com', 'yahoo.com'
            }
        }
        
        # Bias indicators
        self.bias_indicators = {
            'political_bias': [
                'liberal', 'conservative', 'left-wing', 'right-wing',
                'democrat', 'republican', 'partisan'
            ],
            'commercial_bias': [
                'sponsored', 'advertisement', 'promoted', 'affiliate',
                'partner', 'disclosure', 'paid content'
            ],
            'emotional_bias': [
                'shocking', 'unbelievable', 'amazing', 'devastating',
                'outrageous', 'scandal', 'crisis', 'breaking'
            ]
        }
        
        # Fact-checking keywords
        self.fact_check_indicators = {
            'claims': [
                'according to', 'study shows', 'research indicates',
                'data reveals', 'statistics show', 'evidence suggests'
            ],
            'uncertainty': [
                'allegedly', 'reportedly', 'claims', 'suggests',
                'appears to', 'seems to', 'might', 'could'
            ],
            'strong_assertions': [
                'proves', 'demonstrates', 'confirms', 'establishes',
                'shows definitively', 'conclusively'
            ]
        }

    async def validate_comprehensive(self, context: Dict[str, Any]) -> QualityReport:
        """
        Perform comprehensive quality validation of research results
        """
        query = context.get("query", "")
        previous_results = context.get("previous_results", {})
        validation_level = context.get("validation_level", "comprehensive")
        
        self.logger.info(f"Starting quality validation for: {query}")
        animate_thinking("Validating research quality...")
        
        # Extract all data for analysis
        consolidated_data = await self._consolidate_validation_data(previous_results)
        
        # Perform comprehensive quality assessments
        source_credibility = await self._assess_source_credibility(consolidated_data)
        fact_check_results = await self._perform_fact_checking(consolidated_data, query)
        bias_analysis = await self._analyze_bias(consolidated_data)
        completeness_assessment = await self._assess_completeness(consolidated_data, query)
        accuracy_indicators = await self._assess_accuracy(consolidated_data)
        recency_analysis = await self._analyze_recency(consolidated_data)
        
        # Calculate individual quality metrics
        metrics = await self._calculate_quality_metrics(
            source_credibility, fact_check_results, bias_analysis,
            completeness_assessment, accuracy_indicators, recency_analysis
        )
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(metrics)
        
        # Generate recommendations and limitations
        recommendations = await self._generate_quality_recommendations(metrics, overall_score)
        limitations = await self._identify_quality_limitations(consolidated_data, metrics)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(metrics)
        
        report = QualityReport(
            overall_quality_score=overall_score,
            metrics=metrics,
            source_credibility=source_credibility,
            fact_check_results=fact_check_results,
            bias_analysis=bias_analysis,
            completeness_assessment=completeness_assessment,
            accuracy_indicators=accuracy_indicators,
            recency_analysis=recency_analysis,
            recommendations=recommendations,
            limitations=limitations,
            confidence_interval=confidence_interval,
            timestamp=datetime.now()
        )
        
        self.logger.info(f"Quality validation completed. Overall score: {overall_score:.3f}")
        return report

    async def _consolidate_validation_data(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate data for quality validation"""
        consolidated = {
            'sources': [],
            'urls': [],
            'content_blocks': [],
            'metadata': [],
            'timestamps': [],
            'search_results': [],
            'claims': [],
            'statistics': []
        }
        
        for task_id, result in previous_results.items():
            if not result or (isinstance(result, dict) and result.get("status") == "failed"):
                continue
            
            # Extract sources and URLs
            if isinstance(result, dict):
                if "sources" in result:
                    if isinstance(result["sources"], list):
                        consolidated['sources'].extend(result["sources"])
                    
                if "url" in result:
                    consolidated['urls'].append(result["url"])
                    
                if "urls" in result:
                    consolidated['urls'].extend(result["urls"])
                
                # Extract content
                if "content" in result:
                    consolidated['content_blocks'].append(result["content"])
                    
                if "text" in result:
                    consolidated['content_blocks'].append(result["text"])
                
                # Extract metadata
                if "metadata" in result:
                    consolidated['metadata'].append(result["metadata"])
                
                # Extract timestamps
                if "timestamp" in result:
                    consolidated['timestamps'].append(result["timestamp"])
                
                # Extract search results
                if "results" in result and isinstance(result["results"], list):
                    consolidated['search_results'].extend(result["results"])
        
        # Extract claims and statistics from content
        all_content = " ".join(consolidated['content_blocks'])
        consolidated['claims'] = self._extract_claims(all_content)
        consolidated['statistics'] = self._extract_statistics(all_content)
        
        return consolidated

    async def _assess_source_credibility(self, data: Dict[str, Any]) -> List[SourceCredibility]:
        """Assess credibility of all sources"""
        credibility_assessments = []
        
        all_urls = list(set(data.get('urls', []) + data.get('sources', [])))
        
        for url in all_urls:
            if not url or not isinstance(url, str):
                continue
                
            try:
                domain = urlparse(url).netloc.lower()
                domain = domain.replace('www.', '')
                
                # Assess credibility factors
                credibility_score = self._calculate_domain_credibility(domain)
                reputation_factors = self._analyze_domain_reputation(domain)
                content_quality = await self._assess_content_quality_for_url(url, data)
                freshness_score = self._assess_url_freshness(url, data)
                authority_indicators = self._identify_authority_indicators(domain, url)
                
                assessment = SourceCredibility(
                    url=url,
                    domain=domain,
                    credibility_score=credibility_score,
                    reputation_factors=reputation_factors,
                    content_quality=content_quality,
                    freshness_score=freshness_score,
                    authority_indicators=authority_indicators
                )
                
                credibility_assessments.append(assessment)
                
            except Exception as e:
                self.logger.warning(f"Error assessing credibility for {url}: {str(e)}")
                continue
        
        return credibility_assessments

    def _calculate_domain_credibility(self, domain: str) -> float:
        """Calculate domain credibility score"""
        base_score = 0.5  # Neutral starting point
        
        # Check against credible domain lists
        for credibility_level, domains in self.credible_domains.items():
            if any(cred_domain in domain for cred_domain in domains):
                if credibility_level == 'high_credibility':
                    return 0.95
                elif credibility_level == 'medium_credibility':
                    return 0.8
                elif credibility_level == 'business_credible':
                    return 0.75
        
        # Domain characteristics analysis
        if domain.endswith('.edu'):
            base_score += 0.3
        elif domain.endswith('.gov'):
            base_score += 0.4
        elif domain.endswith('.org'):
            base_score += 0.1
        elif domain.endswith('.com'):
            base_score += 0.05
        
        # Length and complexity (longer, more specific domains often more credible)
        if len(domain) > 15:
            base_score += 0.05
        
        # Subdomain analysis
        subdomains = domain.split('.')
        if len(subdomains) > 2:
            if any(academic in domain for academic in ['scholar', 'research', 'journal']):
                base_score += 0.15
        
        return min(1.0, base_score)

    def _analyze_domain_reputation(self, domain: str) -> Dict[str, float]:
        """Analyze domain reputation factors"""
        factors = {
            'academic_authority': 0.0,
            'media_reputation': 0.0,
            'government_official': 0.0,
            'commercial_reliability': 0.0,
            'technical_authority': 0.0
        }
        
        # Academic authority
        academic_indicators = ['edu', 'scholar', 'research', 'university', 'institute']
        if any(indicator in domain for indicator in academic_indicators):
            factors['academic_authority'] = 0.9
        
        # Government official
        if '.gov' in domain or 'official' in domain:
            factors['government_official'] = 0.95
        
        # Media reputation
        reputable_media = ['reuters', 'bloomberg', 'wsj', 'ft', 'economist', 'bbc', 'npr']
        if any(media in domain for media in reputable_media):
            factors['media_reputation'] = 0.85
        
        # Technical authority
        tech_indicators = ['ieee', 'acm', 'arxiv', 'technical', 'engineering']
        if any(indicator in domain for indicator in tech_indicators):
            factors['technical_authority'] = 0.9
        
        # Commercial reliability
        reliable_commercial = ['fortune', 'forbes', 'marketwatch', 'sec.gov']
        if any(commercial in domain for commercial in reliable_commercial):
            factors['commercial_reliability'] = 0.8
        
        return factors

    async def _assess_content_quality_for_url(self, url: str, data: Dict[str, Any]) -> float:
        """Assess content quality for a specific URL"""
        # Find content associated with this URL
        relevant_content = []
        
        for content_block in data.get('content_blocks', []):
            if isinstance(content_block, str) and len(content_block) > 100:
                relevant_content.append(content_block)
        
        if not relevant_content:
            return 0.5  # Neutral score if no content found
        
        # Analyze content quality factors
        combined_content = " ".join(relevant_content)
        
        quality_score = 0.0
        factors_count = 0
        
        # Length and depth
        if len(combined_content) > 1000:
            quality_score += 0.2
            factors_count += 1
        
        # Presence of citations or references
        citation_patterns = [r'\[\d+\]', r'\(\d{4}\)', r'et al\.', r'doi:', r'http']
        citation_count = sum(len(re.findall(pattern, combined_content)) for pattern in citation_patterns)
        if citation_count > 0:
            quality_score += min(0.3, citation_count * 0.05)
            factors_count += 1
        
        # Technical vocabulary (indicates expertise)
        technical_words = len(re.findall(r'\b[a-zA-Z]{8,}\b', combined_content))
        if technical_words > 10:
            quality_score += 0.15
            factors_count += 1
        
        # Structured content (headings, lists, etc.)
        structure_indicators = combined_content.count('\n') + combined_content.count('•') + combined_content.count('-')
        if structure_indicators > 5:
            quality_score += 0.1
            factors_count += 1
        
        # Normalize score
        if factors_count > 0:
            return min(1.0, 0.5 + (quality_score / factors_count))
        else:
            return 0.5

    def _assess_url_freshness(self, url: str, data: Dict[str, Any]) -> float:
        """Assess freshness/recency of URL content"""
        # Look for date indicators in metadata
        current_year = datetime.now().year
        freshness_score = 0.5  # Default neutral
        
        # Check timestamps in data
        for timestamp in data.get('timestamps', []):
            try:
                if isinstance(timestamp, str):
                    # Try to parse various date formats
                    for date_format in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y']:
                        try:
                            date_obj = datetime.strptime(timestamp[:len(date_format)], date_format)
                            years_old = current_year - date_obj.year
                            
                            if years_old == 0:
                                freshness_score = 1.0
                            elif years_old == 1:
                                freshness_score = 0.8
                            elif years_old <= 3:
                                freshness_score = 0.6
                            elif years_old <= 5:
                                freshness_score = 0.4
                            else:
                                freshness_score = 0.2
                            break
                        except ValueError:
                            continue
            except Exception:
                continue
        
        return freshness_score

    def _identify_authority_indicators(self, domain: str, url: str) -> List[str]:
        """Identify authority indicators for a source"""
        indicators = []
        
        # Domain-based indicators
        if '.edu' in domain:
            indicators.append('Educational Institution')
        if '.gov' in domain:
            indicators.append('Government Source')
        if 'research' in domain:
            indicators.append('Research Organization')
        if 'journal' in domain:
            indicators.append('Academic Journal')
        
        # URL path indicators
        if '/research/' in url:
            indicators.append('Research Section')
        if '/publication/' in url or '/paper/' in url:
            indicators.append('Academic Publication')
        if '/official/' in url:
            indicators.append('Official Documentation')
        
        # Content-based indicators (simplified)
        authority_keywords = ['peer-reviewed', 'published', 'citation', 'methodology', 'references']
        
        return indicators

    async def _perform_fact_checking(self, data: Dict[str, Any], query: str) -> List[FactCheckResult]:
        """Perform automated fact-checking on claims"""
        fact_check_results = []
        
        claims = data.get('claims', [])
        all_content = " ".join(data.get('content_blocks', []))
        
        for claim in claims:
            if len(claim) < 10:  # Skip very short claims
                continue
            
            # Analyze claim characteristics
            verification_status = self._assess_claim_verifiability(claim, all_content)
            confidence = self._calculate_claim_confidence(claim, all_content)
            supporting_sources = self._find_supporting_sources(claim, data)
            contradicting_sources = self._find_contradicting_sources(claim, data)
            context = self._extract_claim_context(claim, all_content)
            
            result = FactCheckResult(
                claim=claim,
                verification_status=verification_status,
                confidence=confidence,
                supporting_sources=supporting_sources,
                contradicting_sources=contradicting_sources,
                context=context
            )
            
            fact_check_results.append(result)
        
        return fact_check_results[:10]  # Limit to top 10 claims

    def _extract_claims(self, content: str) -> List[str]:
        """Extract factual claims from content"""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Look for claim indicators
            claim_indicators = [
                'according to', 'study shows', 'research indicates', 'data reveals',
                'statistics show', 'found that', 'discovered that', 'analysis shows'
            ]
            
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                claims.append(sentence)
            
            # Look for statistical claims
            if re.search(r'\d+(?:\.\d+)?%', sentence) or re.search(r'\$[\d,]+', sentence):
                claims.append(sentence)
        
        return claims[:20]  # Limit to first 20 claims

    def _extract_statistics(self, content: str) -> List[str]:
        """Extract statistical information from content"""
        statistics = []
        
        # Percentage patterns
        percentages = re.findall(r'\d+(?:\.\d+)?%', content)
        statistics.extend(percentages)
        
        # Financial figures
        financial = re.findall(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?', content)
        statistics.extend(financial)
        
        # General numbers with context
        numbers_with_context = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?\s+(?:people|users|companies|percent|million|billion)', content)
        statistics.extend(numbers_with_context)
        
        return list(set(statistics))  # Remove duplicates

    def _assess_claim_verifiability(self, claim: str, content: str) -> str:
        """Assess whether a claim can be verified"""
        claim_lower = claim.lower()
        
        # Strong verification indicators
        strong_indicators = ['study shows', 'research proves', 'data confirms', 'officially announced']
        if any(indicator in claim_lower for indicator in strong_indicators):
            return 'verified'
        
        # Uncertainty indicators
        uncertainty_indicators = ['allegedly', 'reportedly', 'claims', 'suggests', 'appears']
        if any(indicator in claim_lower for indicator in uncertainty_indicators):
            return 'unverified'
        
        # Look for supporting evidence in content
        claim_words = set(claim.lower().split())
        content_words = set(content.lower().split())
        overlap_ratio = len(claim_words & content_words) / len(claim_words)
        
        if overlap_ratio > 0.7:
            return 'verified'
        elif overlap_ratio > 0.4:
            return 'partially_verified'
        else:
            return 'unverified'

    def _calculate_claim_confidence(self, claim: str, content: str) -> float:
        """Calculate confidence level for a claim"""
        confidence_factors = []
        
        # Source citation presence
        if re.search(r'according to|source:|study by', claim.lower()):
            confidence_factors.append(0.8)
        
        # Statistical backing
        if re.search(r'\d+(?:\.\d+)?%|\d+(?:,\d{3})*', claim):
            confidence_factors.append(0.7)
        
        # Content support
        claim_words = set(claim.lower().split())
        content_words = set(content.lower().split())
        support_ratio = len(claim_words & content_words) / len(claim_words)
        confidence_factors.append(support_ratio)
        
        # Specificity (more specific claims often more reliable)
        if len(claim.split()) > 10:
            confidence_factors.append(0.6)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5

    def _find_supporting_sources(self, claim: str, data: Dict[str, Any]) -> List[str]:
        """Find sources that support a claim"""
        supporting_sources = []
        claim_words = set(claim.lower().split())
        
        # Check content blocks for support
        for i, content_block in enumerate(data.get('content_blocks', [])):
            if isinstance(content_block, str):
                content_words = set(content_block.lower().split())
                overlap = len(claim_words & content_words) / len(claim_words)
                
                if overlap > 0.5:  # Significant overlap
                    # Try to find associated source
                    sources = data.get('sources', [])
                    if i < len(sources):
                        supporting_sources.append(sources[i])
        
        return supporting_sources

    def _find_contradicting_sources(self, claim: str, data: Dict[str, Any]) -> List[str]:
        """Find sources that contradict a claim"""
        # Simplified implementation - in practice, this would require
        # advanced NLP and semantic analysis
        contradicting_sources = []
        
        # Look for contradictory keywords
        contradiction_indicators = ['however', 'but', 'contrary', 'dispute', 'disagree', 'refute']
        
        for content_block in data.get('content_blocks', []):
            if isinstance(content_block, str):
                if any(indicator in content_block.lower() for indicator in contradiction_indicators):
                    # This is a simplified approach
                    contradicting_sources.append("Content with contradictory indicators found")
                    break
        
        return contradicting_sources

    def _extract_claim_context(self, claim: str, content: str) -> str:
        """Extract context around a claim"""
        # Find the claim in content and extract surrounding context
        claim_index = content.lower().find(claim.lower())
        
        if claim_index != -1:
            start = max(0, claim_index - 200)
            end = min(len(content), claim_index + len(claim) + 200)
            context = content[start:end].strip()
            return context
        
        return "Context not found"

    async def _analyze_bias(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential bias in sources and content"""
        bias_analysis = {
            'overall_bias_score': 0.0,
            'bias_types': defaultdict(float),
            'bias_indicators': [],
            'source_diversity': 0.0,
            'perspective_balance': 0.0
        }
        
        all_content = " ".join(data.get('content_blocks', []))
        
        # Analyze each bias type
        total_bias_score = 0.0
        bias_count = 0
        
        for bias_type, keywords in self.bias_indicators.items():
            bias_score = sum(all_content.lower().count(keyword) for keyword in keywords)
            if bias_score > 0:
                normalized_score = min(1.0, bias_score / 100)  # Normalize
                bias_analysis['bias_types'][bias_type] = normalized_score
                total_bias_score += normalized_score
                bias_count += 1
        
        # Calculate overall bias score (lower is better)
        if bias_count > 0:
            bias_analysis['overall_bias_score'] = total_bias_score / bias_count
        
        # Analyze source diversity
        sources = data.get('sources', [])
        unique_domains = set()
        for source in sources:
            if isinstance(source, str):
                try:
                    domain = urlparse(source).netloc.lower().replace('www.', '')
                    unique_domains.add(domain)
                except:
                    continue
        
        # Source diversity score (more diverse sources = lower bias risk)
        bias_analysis['source_diversity'] = min(1.0, len(unique_domains) / 5)
        
        # Perspective balance (simplified)
        positive_indicators = ['positive', 'growth', 'success', 'improvement', 'benefit']
        negative_indicators = ['negative', 'decline', 'failure', 'problem', 'risk']
        
        positive_count = sum(all_content.lower().count(word) for word in positive_indicators)
        negative_count = sum(all_content.lower().count(word) for word in negative_indicators)
        
        if positive_count + negative_count > 0:
            balance_ratio = min(positive_count, negative_count) / max(positive_count, negative_count)
            bias_analysis['perspective_balance'] = balance_ratio
        else:
            bias_analysis['perspective_balance'] = 1.0
        
        return bias_analysis

    async def _assess_completeness(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Assess completeness of research relative to query"""
        query_words = set(query.lower().split())
        all_content = " ".join(data.get('content_blocks', [])).lower()
        
        # Query coverage
        covered_words = sum(1 for word in query_words if word in all_content)
        coverage_ratio = covered_words / len(query_words) if query_words else 0
        
        # Content depth
        total_content_length = len(all_content)
        depth_score = min(1.0, total_content_length / 5000)  # Normalize to 5k chars
        
        # Source breadth
        source_count = len(set(data.get('sources', [])))
        breadth_score = min(1.0, source_count / 5)  # Normalize to 5 sources
        
        # Topic coverage (simplified)
        topic_keywords = [
            'background', 'history', 'current', 'future', 'impact',
            'analysis', 'comparison', 'benefits', 'challenges', 'conclusion'
        ]
        topic_coverage = sum(1 for keyword in topic_keywords if keyword in all_content)
        topic_score = topic_coverage / len(topic_keywords)
        
        completeness_score = statistics.mean([coverage_ratio, depth_score, breadth_score, topic_score])
        
        return {
            'completeness_score': completeness_score,
            'query_coverage': coverage_ratio,
            'content_depth': depth_score,
            'source_breadth': breadth_score,
            'topic_coverage': topic_score,
            'missing_aspects': self._identify_missing_aspects(query, all_content)
        }

    def _identify_missing_aspects(self, query: str, content: str) -> List[str]:
        """Identify potentially missing aspects of the query"""
        missing_aspects = []
        
        # Common research aspects to check
        research_aspects = {
            'historical context': ['history', 'historical', 'background', 'origin'],
            'current status': ['current', 'present', 'today', 'now', 'recent'],
            'future outlook': ['future', 'forecast', 'prediction', 'outlook', 'trend'],
            'challenges': ['challenge', 'problem', 'issue', 'difficulty', 'obstacle'],
            'benefits': ['benefit', 'advantage', 'positive', 'gain', 'improvement'],
            'comparison': ['compare', 'versus', 'vs', 'relative', 'alternative'],
            'statistical data': ['data', 'statistics', 'numbers', 'percentage', 'rate']
        }
        
        for aspect, keywords in research_aspects.items():
            if not any(keyword in content.lower() for keyword in keywords):
                missing_aspects.append(aspect)
        
        return missing_aspects

    async def _assess_accuracy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess accuracy indicators in the data"""
        accuracy_indicators = {
            'citation_count': 0,
            'statistical_backing': 0,
            'cross_reference_potential': 0,
            'source_consistency': 0.0,
            'temporal_consistency': 0.0
        }
        
        all_content = " ".join(data.get('content_blocks', []))
        
        # Citation analysis
        citation_patterns = [r'\[\d+\]', r'\(\d{4}\)', r'et al\.', r'doi:', r'source:']
        for pattern in citation_patterns:
            accuracy_indicators['citation_count'] += len(re.findall(pattern, all_content))
        
        # Statistical backing
        stats = data.get('statistics', [])
        accuracy_indicators['statistical_backing'] = len(stats)
        
        # Cross-reference potential
        sources = data.get('sources', [])
        accuracy_indicators['cross_reference_potential'] = len(set(sources))
        
        # Source consistency (simplified)
        if len(sources) > 1:
            # Check for consistent information across sources
            content_blocks = data.get('content_blocks', [])
            if len(content_blocks) > 1:
                # Simplified consistency check using content length similarity
                lengths = [len(block) for block in content_blocks if isinstance(block, str)]
                if lengths:
                    avg_length = statistics.mean(lengths)
                    length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
                    consistency = 1.0 - min(1.0, length_variance / (avg_length ** 2))
                    accuracy_indicators['source_consistency'] = consistency
        
        return accuracy_indicators

    async def _analyze_recency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze recency and temporal relevance of data"""
        current_year = datetime.now().year
        timestamps = data.get('timestamps', [])
        
        recency_analysis = {
            'average_age': 0.0,
            'freshness_score': 0.0,
            'temporal_distribution': {},
            'outdated_content_ratio': 0.0
        }
        
        if timestamps:
            ages = []
            year_distribution = defaultdict(int)
            
            for timestamp in timestamps:
                try:
                    if isinstance(timestamp, str):
                        # Extract year from timestamp
                        year_match = re.search(r'20\d{2}', timestamp)
                        if year_match:
                            year = int(year_match.group())
                            age = current_year - year
                            ages.append(age)
                            year_distribution[year] += 1
                except:
                    continue
            
            if ages:
                recency_analysis['average_age'] = statistics.mean(ages)
                
                # Freshness score (inverse of age, normalized)
                max_age = max(ages)
                freshness_scores = [(max_age - age) / max_age for age in ages]
                recency_analysis['freshness_score'] = statistics.mean(freshness_scores)
                
                # Temporal distribution
                recency_analysis['temporal_distribution'] = dict(year_distribution)
                
                # Outdated content ratio (content older than 5 years)
                outdated_count = sum(1 for age in ages if age > 5)
                recency_analysis['outdated_content_ratio'] = outdated_count / len(ages)
        
        return recency_analysis

    async def _calculate_quality_metrics(self, source_credibility: List[SourceCredibility],
                                       fact_check_results: List[FactCheckResult],
                                       bias_analysis: Dict[str, Any],
                                       completeness_assessment: Dict[str, Any],
                                       accuracy_indicators: Dict[str, Any],
                                       recency_analysis: Dict[str, Any]) -> List[QualityMetric]:
        """Calculate comprehensive quality metrics"""
        metrics = []
        
        # Source Credibility Metric
        if source_credibility:
            avg_credibility = statistics.mean([s.credibility_score for s in source_credibility])
            metrics.append(QualityMetric(
                metric_name="Source Credibility",
                score=avg_credibility,
                confidence=0.9,
                details=f"Average credibility across {len(source_credibility)} sources",
                evidence=[f"{s.domain}: {s.credibility_score:.2f}" for s in source_credibility[:3]],
                weight=0.25
            ))
        
        # Information Accuracy Metric
        citation_score = min(1.0, accuracy_indicators.get('citation_count', 0) / 10)
        statistical_score = min(1.0, accuracy_indicators.get('statistical_backing', 0) / 5)
        accuracy_score = (citation_score + statistical_score) / 2
        
        metrics.append(QualityMetric(
            metric_name="Information Accuracy",
            score=accuracy_score,
            confidence=0.8,
            details=f"Based on citations and statistical backing",
            evidence=[f"Citations: {accuracy_indicators.get('citation_count', 0)}",
                     f"Statistics: {accuracy_indicators.get('statistical_backing', 0)}"],
            weight=0.2
        ))
        
        # Content Completeness Metric
        completeness_score = completeness_assessment.get('completeness_score', 0.0)
        metrics.append(QualityMetric(
            metric_name="Content Completeness",
            score=completeness_score,
            confidence=0.85,
            details="Comprehensive coverage assessment",
            evidence=[f"Query coverage: {completeness_assessment.get('query_coverage', 0):.2f}",
                     f"Content depth: {completeness_assessment.get('content_depth', 0):.2f}"],
            weight=0.2
        ))
        
        # Bias Assessment Metric (inverted - lower bias = higher quality)
        bias_score = 1.0 - bias_analysis.get('overall_bias_score', 0.0)
        metrics.append(QualityMetric(
            metric_name="Bias Assessment",
            score=bias_score,
            confidence=0.7,
            details="Low bias indicates higher quality",
            evidence=[f"Source diversity: {bias_analysis.get('source_diversity', 0):.2f}",
                     f"Perspective balance: {bias_analysis.get('perspective_balance', 0):.2f}"],
            weight=0.15
        ))
        
        # Recency/Freshness Metric
        freshness_score = recency_analysis.get('freshness_score', 0.5)
        metrics.append(QualityMetric(
            metric_name="Content Freshness",
            score=freshness_score,
            confidence=0.75,
            details="Temporal relevance of information",
            evidence=[f"Average age: {recency_analysis.get('average_age', 0):.1f} years",
                     f"Outdated ratio: {recency_analysis.get('outdated_content_ratio', 0):.2f}"],
            weight=0.1
        ))
        
        # Fact Verification Metric
        if fact_check_results:
            verified_count = sum(1 for r in fact_check_results if r.verification_status == 'verified')
            verification_score = verified_count / len(fact_check_results)
            
            metrics.append(QualityMetric(
                metric_name="Fact Verification",
                score=verification_score,
                confidence=0.8,
                details=f"Verification status of {len(fact_check_results)} claims",
                evidence=[f"Verified claims: {verified_count}/{len(fact_check_results)}"],
                weight=0.1
            ))
        
        return metrics

    def _calculate_overall_quality_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate weighted overall quality score"""
        if not metrics:
            return 0.0
        
        weighted_sum = sum(metric.score * metric.weight for metric in metrics)
        total_weight = sum(metric.weight for metric in metrics)
        
        return round(weighted_sum / total_weight if total_weight > 0 else 0.0, 3)

    async def _generate_quality_recommendations(self, metrics: List[QualityMetric], 
                                              overall_score: float) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Overall quality recommendations
        if overall_score >= 0.9:
            recommendations.append("Excellent quality research - suitable for high-stakes decision making")
        elif overall_score >= 0.75:
            recommendations.append("Good quality research - minor improvements could enhance reliability")
        elif overall_score >= 0.6:
            recommendations.append("Acceptable quality - consider additional verification for critical decisions")
        else:
            recommendations.append("Quality concerns identified - additional research recommended")
        
        # Metric-specific recommendations
        for metric in metrics:
            if metric.score < 0.6:
                if metric.metric_name == "Source Credibility":
                    recommendations.append("Consider seeking additional sources from more credible domains")
                elif metric.metric_name == "Information Accuracy":
                    recommendations.append("Verify key claims with additional authoritative sources")
                elif metric.metric_name == "Content Completeness":
                    recommendations.append("Research additional aspects to provide more comprehensive coverage")
                elif metric.metric_name == "Bias Assessment":
                    recommendations.append("Seek diverse perspectives to balance potential bias")
                elif metric.metric_name == "Content Freshness":
                    recommendations.append("Update research with more recent information")
        
        return recommendations[:5]  # Top 5 recommendations

    async def _identify_quality_limitations(self, data: Dict[str, Any], 
                                          metrics: List[QualityMetric]) -> List[str]:
        """Identify limitations affecting quality assessment"""
        limitations = []
        
        # Data availability limitations
        source_count = len(data.get('sources', []))
        if source_count < 3:
            limitations.append(f"Limited number of sources ({source_count}) may affect assessment accuracy")
        
        # Content volume limitations
        content_length = sum(len(str(block)) for block in data.get('content_blocks', []))
        if content_length < 2000:
            limitations.append("Limited content volume may affect quality assessment depth")
        
        # Temporal limitations
        timestamps = data.get('timestamps', [])
        if not timestamps:
            limitations.append("No temporal information available for freshness assessment")
        
        # Metric-specific limitations
        low_confidence_metrics = [m for m in metrics if m.confidence < 0.7]
        if low_confidence_metrics:
            limitations.append(f"Low confidence in {len(low_confidence_metrics)} quality metrics")
        
        # Fact-checking limitations
        claims = data.get('claims', [])
        if len(claims) > 20:
            limitations.append("Large number of claims may limit comprehensive fact-checking")
        
        return limitations[:5]  # Top 5 limitations

    def _calculate_confidence_interval(self, metrics: List[QualityMetric]) -> Tuple[float, float]:
        """Calculate confidence interval for overall quality score"""
        if not metrics:
            return (0.0, 0.0)
        
        scores = [metric.score for metric in metrics]
        confidences = [metric.confidence for metric in metrics]
        
        # Simple confidence interval based on score variance and metric confidence
        mean_score = statistics.mean(scores)
        mean_confidence = statistics.mean(confidences)
        
        if len(scores) > 1:
            score_variance = statistics.variance(scores)
            confidence_adjustment = mean_confidence * 0.1  # 10% adjustment based on confidence
            
            lower_bound = max(0.0, mean_score - score_variance - confidence_adjustment)
            upper_bound = min(1.0, mean_score + score_variance + confidence_adjustment)
        else:
            # Single metric case
            confidence_margin = (1.0 - mean_confidence) * 0.2
            lower_bound = max(0.0, mean_score - confidence_margin)
            upper_bound = min(1.0, mean_score + confidence_margin)
        
        return (round(lower_bound, 3), round(upper_bound, 3))

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_quality_agent():
        """Test the quality agent"""
        agent = QualityAgent()
        
        # Mock previous results
        mock_results = {
            "search_001": {
                "results": [
                    {"title": "AI Research Study", "url": "https://academic.edu/ai-study"},
                    {"title": "Market Analysis", "url": "https://reuters.com/ai-market"}
                ],
                "sources": ["https://academic.edu/ai-study", "https://reuters.com/ai-market"]
            },
            "web_001": {
                "content": "According to recent research published in 2024, artificial intelligence market growth shows 25% annual increase. Study by MIT indicates strong correlation between AI adoption and productivity gains.",
                "url": "https://academic.edu/ai-study",
                "metadata": {"date": "2024-01-15"}
            }
        }
        
        context = {
            "query": "artificial intelligence market trends and growth analysis",
            "previous_results": mock_results,
            "validation_level": "comprehensive"
        }
        
        # Run quality validation
        report = await agent.validate_comprehensive(context)
        
        # Print results
        print(f"\n=== Quality Validation Report ===")
        print(f"Overall Quality Score: {report.overall_quality_score}")
        print(f"Confidence Interval: {report.confidence_interval}")
        print(f"Number of Metrics: {len(report.metrics)}")
        print(f"Source Credibility Assessments: {len(report.source_credibility)}")
        print(f"Fact Check Results: {len(report.fact_check_results)}")
        
        # Print quality metrics
        print(f"\n=== Quality Metrics ===")
        for metric in report.metrics:
            print(f"{metric.metric_name}: {metric.score:.3f} (confidence: {metric.confidence:.2f})")
        
        # Print recommendations
        print(f"\n=== Recommendations ===")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
    
    # Run the test
    asyncio.run(test_quality_agent())
