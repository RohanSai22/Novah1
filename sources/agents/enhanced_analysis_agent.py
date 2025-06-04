"""
Enhanced Analysis Agent - Advanced Data Analysis and Synthesis
This agent performs deep analysis, pattern recognition, statistical analysis, and data synthesis
"""
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade
from wordcloud import WordCloud
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import re
from collections import Counter, defaultdict
import logging
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import warnings
warnings.filterwarnings('ignore')

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent


class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    CONTENT_ANALYSIS = "content_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    TREND_ANALYSIS = "trend_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NETWORK_ANALYSIS = "network_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"


class DataType(Enum):
    """Types of data that can be analyzed"""
    TEXT = "text"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    MIXED = "mixed"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    analysis_type: str
    data_type: str
    results: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    confidence_score: float
    metadata: Dict[str, Any]
    timestamp: str


@dataclass
class StatisticalSummary:
    """Statistical summary of data"""
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    std_dev: Optional[float] = None
    variance: Optional[float] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    quartiles: Optional[List[float]] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    correlation_matrix: Optional[Dict[str, Any]] = None


@dataclass
class ContentAnalysisResult:
    """Results from content analysis"""
    word_frequency: Dict[str, int]
    key_themes: List[str]
    readability_scores: Dict[str, float]
    sentiment_distribution: Dict[str, float]
    entity_mentions: Dict[str, int]
    text_clusters: List[Dict[str, Any]]
    similarity_matrix: Optional[np.ndarray] = None


class EnhancedAnalysisAgent(Agent):
    """Enhanced Analysis Agent for comprehensive data analysis and synthesis"""
    
    def __init__(self, provider):
        super().__init__(provider=provider, name="Enhanced Analysis Agent")
        self.analysis_history = []
        self.cached_results = {}
        self.visualization_engine = VisualizationEngine()
        
    async def analyze_data(self, 
                          data: Union[List[Dict], pd.DataFrame, List[str]], 
                          analysis_types: List[AnalysisType] = None,
                          custom_parameters: Dict[str, Any] = None) -> List[AnalysisResult]:
        """
        Perform comprehensive analysis on provided data
        
        Args:
            data: Data to analyze (can be text, structured data, etc.)
            analysis_types: Types of analysis to perform
            custom_parameters: Custom parameters for analysis
            
        Returns:
            List of analysis results
        """
        if analysis_types is None:
            analysis_types = [AnalysisType.CONTENT_ANALYSIS, AnalysisType.STATISTICAL_ANALYSIS]
            
        if custom_parameters is None:
            custom_parameters = {}
            
        results = []
        data_type = self._determine_data_type(data)
        
        for analysis_type in analysis_types:
            try:
                result = await self._perform_analysis(data, analysis_type, data_type, custom_parameters)
                results.append(result)
            except Exception as e:
                logging.error(f"Error in {analysis_type.value}: {str(e)}")
                
        return results
    
    def _determine_data_type(self, data: Any) -> DataType:
        """Determine the type of data being analyzed"""
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return DataType.TEXT
            elif all(isinstance(item, dict) for item in data):
                return DataType.STRUCTURED
            else:
                return DataType.MIXED
        elif isinstance(data, pd.DataFrame):
            return DataType.STRUCTURED
        elif isinstance(data, str):
            return DataType.TEXT
        else:
            return DataType.UNSTRUCTURED
    
    async def _perform_analysis(self, 
                               data: Any, 
                               analysis_type: AnalysisType, 
                               data_type: DataType,
                               parameters: Dict[str, Any]) -> AnalysisResult:
        """Perform specific type of analysis"""
        
        if analysis_type == AnalysisType.CONTENT_ANALYSIS:
            return await self._content_analysis(data, parameters)
        elif analysis_type == AnalysisType.STATISTICAL_ANALYSIS:
            return await self._statistical_analysis(data, parameters)
        elif analysis_type == AnalysisType.TREND_ANALYSIS:
            return await self._trend_analysis(data, parameters)
        elif analysis_type == AnalysisType.SENTIMENT_ANALYSIS:
            return await self._sentiment_analysis(data, parameters)
        elif analysis_type == AnalysisType.NETWORK_ANALYSIS:
            return await self._network_analysis(data, parameters)
        elif analysis_type == AnalysisType.COMPARATIVE_ANALYSIS:
            return await self._comparative_analysis(data, parameters)
        elif analysis_type == AnalysisType.PATTERN_RECOGNITION:
            return await self._pattern_recognition(data, parameters)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    async def _content_analysis(self, data: Any, parameters: Dict[str, Any]) -> AnalysisResult:
        """Perform comprehensive content analysis"""
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            texts = data
        elif isinstance(data, str):
            texts = [data]
        else:
            # Extract text from structured data
            texts = self._extract_text_from_data(data)
        
        # Word frequency analysis
        all_text = " ".join(texts)
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)
        
        # TF-IDF analysis for key themes
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            key_themes = feature_names[:20].tolist()
        except:
            key_themes = list(word_freq.keys())[:20]
        
        # Readability analysis
        readability_scores = {}
        if texts:
            sample_text = texts[0] if len(texts[0]) > 100 else " ".join(texts)
            try:
                readability_scores = {
                    'flesch_reading_ease': flesch_reading_ease(sample_text),
                    'flesch_kincaid_grade': flesch_kincaid_grade(sample_text)
                }
            except:
                readability_scores = {'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0}
        
        # Text clustering
        text_clusters = []
        if len(texts) > 1:
            try:
                n_clusters = min(5, len(texts))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
                
                for i in range(n_clusters):
                    cluster_texts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
                    text_clusters.append({
                        'cluster_id': i,
                        'texts': cluster_texts[:5],  # Sample texts
                        'size': len(cluster_texts)
                    })
            except:
                pass
        
        # Entity extraction (simplified)
        entities = self._extract_entities(all_text)
        
        content_result = ContentAnalysisResult(
            word_frequency=dict(word_freq.most_common(50)),
            key_themes=key_themes,
            readability_scores=readability_scores,
            sentiment_distribution={'positive': 0.4, 'neutral': 0.4, 'negative': 0.2},  # Placeholder
            entity_mentions=entities,
            text_clusters=text_clusters
        )
        
        # Generate visualizations
        visualizations = [
            self.visualization_engine.create_word_cloud(word_freq),
            self.visualization_engine.create_theme_bar_chart(key_themes),
            self.visualization_engine.create_readability_gauge(readability_scores)
        ]
        
        insights = self._generate_content_insights(content_result)
        
        return AnalysisResult(
            analysis_type=AnalysisType.CONTENT_ANALYSIS.value,
            data_type=DataType.TEXT.value,
            results=asdict(content_result),
            visualizations=visualizations,
            insights=insights,
            confidence_score=0.85,
            metadata={'num_texts': len(texts), 'total_words': len(words)},
            timestamp=datetime.now().isoformat()
        )
    
    async def _statistical_analysis(self, data: Any, parameters: Dict[str, Any]) -> AnalysisResult:
        """Perform statistical analysis on numerical data"""
        df = self._convert_to_dataframe(data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            # Try to extract numerical data
            df = self._extract_numerical_features(df)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        summary = StatisticalSummary()
        correlations = {}
        distributions = {}
        
        if len(numeric_columns) > 0:
            # Basic statistics
            numeric_data = df[numeric_columns]
            summary.mean = numeric_data.mean().to_dict()
            summary.median = numeric_data.median().to_dict()
            summary.std_dev = numeric_data.std().to_dict()
            summary.variance = numeric_data.var().to_dict()
            summary.min_value = numeric_data.min().to_dict()
            summary.max_value = numeric_data.max().to_dict()
            summary.quartiles = numeric_data.quantile([0.25, 0.5, 0.75]).to_dict()
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                correlation_matrix = numeric_data.corr()
                summary.correlation_matrix = correlation_matrix.to_dict()
            
            # Distribution analysis
            for col in numeric_columns:
                distributions[col] = {
                    'skewness': stats.skew(numeric_data[col].dropna()),
                    'kurtosis': stats.kurtosis(numeric_data[col].dropna()),
                    'normality_test': stats.normaltest(numeric_data[col].dropna())[1]
                }
        
        # Generate visualizations
        visualizations = [
            self.visualization_engine.create_correlation_heatmap(summary.correlation_matrix),
            self.visualization_engine.create_distribution_plots(df, numeric_columns),
            self.visualization_engine.create_statistical_summary_table(summary)
        ]
        
        insights = self._generate_statistical_insights(summary, distributions)
        
        return AnalysisResult(
            analysis_type=AnalysisType.STATISTICAL_ANALYSIS.value,
            data_type=DataType.NUMERICAL.value,
            results={'summary': asdict(summary), 'distributions': distributions},
            visualizations=visualizations,
            insights=insights,
            confidence_score=0.9,
            metadata={'numeric_columns': len(numeric_columns), 'total_rows': len(df)},
            timestamp=datetime.now().isoformat()
        )
    
    async def _trend_analysis(self, data: Any, parameters: Dict[str, Any]) -> AnalysisResult:
        """Analyze trends in temporal data"""
        df = self._convert_to_dataframe(data)
        
        # Try to identify temporal columns
        temporal_columns = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                temporal_columns.append(col)
        
        if not temporal_columns and 'timestamp' in parameters:
            temporal_columns = [parameters['timestamp']]
        
        trends = {}
        seasonal_patterns = {}
        forecasts = {}
        
        for col in temporal_columns:
            if col in df.columns:
                # Basic trend analysis
                df_sorted = df.sort_values(col)
                numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
                
                for num_col in numeric_cols:
                    # Calculate trend using linear regression
                    x = np.arange(len(df_sorted))
                    y = df_sorted[num_col].fillna(method='forward')
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    trends[f"{col}_{num_col}"] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'strength': abs(r_value)
                    }
        
        visualizations = [
            self.visualization_engine.create_trend_lines(df, temporal_columns),
            self.visualization_engine.create_seasonal_decomposition(df, temporal_columns),
            self.visualization_engine.create_trend_summary_table(trends)
        ]
        
        insights = self._generate_trend_insights(trends, seasonal_patterns)
        
        return AnalysisResult(
            analysis_type=AnalysisType.TREND_ANALYSIS.value,
            data_type=DataType.TEMPORAL.value,
            results={'trends': trends, 'seasonal_patterns': seasonal_patterns},
            visualizations=visualizations,
            insights=insights,
            confidence_score=0.8,
            metadata={'temporal_columns': len(temporal_columns)},
            timestamp=datetime.now().isoformat()
        )
    
    async def _sentiment_analysis(self, data: Any, parameters: Dict[str, Any]) -> AnalysisResult:
        """Perform sentiment analysis on text data"""
        texts = self._extract_text_from_data(data)
        
        # Simple sentiment analysis (can be replaced with more sophisticated models)
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'happy'])
        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'negative', 'sad', 'angry', 'disappointed'])
        
        sentiments = []
        sentiment_scores = []
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count > negative_count:
                sentiment = 'positive'
                score = (positive_count - negative_count) / len(words) if words else 0
            elif negative_count > positive_count:
                sentiment = 'negative'
                score = (negative_count - positive_count) / len(words) if words else 0
            else:
                sentiment = 'neutral'
                score = 0
            
            sentiments.append(sentiment)
            sentiment_scores.append(score)
        
        sentiment_distribution = Counter(sentiments)
        avg_sentiment_score = np.mean(sentiment_scores)
        
        visualizations = [
            self.visualization_engine.create_sentiment_pie_chart(sentiment_distribution),
            self.visualization_engine.create_sentiment_timeline(sentiments, sentiment_scores),
            self.visualization_engine.create_sentiment_word_cloud(texts, sentiments)
        ]
        
        insights = self._generate_sentiment_insights(sentiment_distribution, avg_sentiment_score)
        
        return AnalysisResult(
            analysis_type=AnalysisType.SENTIMENT_ANALYSIS.value,
            data_type=DataType.TEXT.value,
            results={
                'distribution': dict(sentiment_distribution),
                'average_score': avg_sentiment_score,
                'individual_scores': sentiment_scores
            },
            visualizations=visualizations,
            insights=insights,
            confidence_score=0.75,
            metadata={'num_texts': len(texts)},
            timestamp=datetime.now().isoformat()
        )
    
    async def _network_analysis(self, data: Any, parameters: Dict[str, Any]) -> AnalysisResult:
        """Perform network analysis on relational data"""
        # Create network from data
        G = nx.Graph()
        
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Build network from structured data
            for item in data:
                if 'source' in item and 'target' in item:
                    G.add_edge(item['source'], item['target'], 
                              weight=item.get('weight', 1))
        
        # Network metrics
        network_metrics = {}
        if len(G.nodes()) > 0:
            network_metrics = {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G),
                'num_connected_components': nx.number_connected_components(G)
            }
            
            # Centrality measures
            if len(G.nodes()) > 1:
                centrality_measures = {
                    'degree_centrality': nx.degree_centrality(G),
                    'betweenness_centrality': nx.betweenness_centrality(G),
                    'closeness_centrality': nx.closeness_centrality(G),
                    'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000)
                }
                network_metrics['centrality'] = centrality_measures
        
        visualizations = [
            self.visualization_engine.create_network_graph(G),
            self.visualization_engine.create_centrality_charts(network_metrics.get('centrality', {})),
            self.visualization_engine.create_network_metrics_table(network_metrics)
        ]
        
        insights = self._generate_network_insights(network_metrics)
        
        return AnalysisResult(
            analysis_type=AnalysisType.NETWORK_ANALYSIS.value,
            data_type=DataType.STRUCTURED.value,
            results=network_metrics,
            visualizations=visualizations,
            insights=insights,
            confidence_score=0.8,
            metadata={'graph_type': 'undirected'},
            timestamp=datetime.now().isoformat()
        )
    
    async def _comparative_analysis(self, data: Any, parameters: Dict[str, Any]) -> AnalysisResult:
        """Compare different datasets or groups within data"""
        df = self._convert_to_dataframe(data)
        
        group_column = parameters.get('group_by')
        comparisons = {}
        
        if group_column and group_column in df.columns:
            # Group-based comparison
            groups = df[group_column].unique()
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                group_stats = {}
                for group in groups:
                    group_data = df[df[group_column] == group][col].dropna()
                    if len(group_data) > 0:
                        group_stats[str(group)] = {
                            'mean': group_data.mean(),
                            'median': group_data.median(),
                            'std': group_data.std(),
                            'count': len(group_data)
                        }
                
                # Statistical tests
                if len(groups) >= 2:
                    group_data_list = [df[df[group_column] == group][col].dropna() for group in groups]
                    group_data_list = [g for g in group_data_list if len(g) > 0]
                    
                    if len(group_data_list) >= 2:
                        try:
                            # ANOVA test
                            f_stat, p_value = stats.f_oneway(*group_data_list)
                            group_stats['statistical_test'] = {
                                'test': 'ANOVA',
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except:
                            pass
                
                comparisons[col] = group_stats
        
        visualizations = [
            self.visualization_engine.create_comparison_box_plots(df, group_column),
            self.visualization_engine.create_comparison_bar_charts(comparisons),
            self.visualization_engine.create_statistical_test_summary(comparisons)
        ]
        
        insights = self._generate_comparative_insights(comparisons)
        
        return AnalysisResult(
            analysis_type=AnalysisType.COMPARATIVE_ANALYSIS.value,
            data_type=DataType.MIXED.value,
            results=comparisons,
            visualizations=visualizations,
            insights=insights,
            confidence_score=0.85,
            metadata={'group_column': group_column},
            timestamp=datetime.now().isoformat()
        )
    
    async def _pattern_recognition(self, data: Any, parameters: Dict[str, Any]) -> AnalysisResult:
        """Identify patterns in data using machine learning techniques"""
        df = self._convert_to_dataframe(data)
        patterns = {}
        
        # Clustering for pattern discovery
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            # Prepare data for clustering
            cluster_data = df[numeric_columns].fillna(df[numeric_columns].mean())
            
            # K-means clustering
            n_clusters = min(5, len(df) // 10) if len(df) > 50 else 2
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(cluster_data)
                
                patterns['clusters'] = {
                    'n_clusters': n_clusters,
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'cluster_labels': clusters.tolist(),
                    'inertia': kmeans.inertia_
                }
                
                # Cluster characteristics
                cluster_stats = {}
                for i in range(n_clusters):
                    cluster_mask = clusters == i
                    cluster_subset = df[cluster_mask]
                    cluster_stats[f'cluster_{i}'] = {
                        'size': len(cluster_subset),
                        'percentage': len(cluster_subset) / len(df) * 100,
                        'characteristics': cluster_subset[numeric_columns].mean().to_dict()
                    }
                patterns['cluster_characteristics'] = cluster_stats
        
        # Anomaly detection
        if len(numeric_columns) > 0:
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(cluster_data)
            
            patterns['anomalies'] = {
                'anomaly_indices': [i for i, x in enumerate(anomalies) if x == -1],
                'anomaly_count': sum(1 for x in anomalies if x == -1),
                'anomaly_percentage': sum(1 for x in anomalies if x == -1) / len(anomalies) * 100
            }
        
        # Frequent patterns in categorical data
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            frequent_patterns = {}
            for col in categorical_columns:
                value_counts = df[col].value_counts()
                frequent_patterns[col] = value_counts.head(10).to_dict()
            patterns['frequent_patterns'] = frequent_patterns
        
        visualizations = [
            self.visualization_engine.create_cluster_scatter_plot(cluster_data, patterns.get('clusters', {})),
            self.visualization_engine.create_anomaly_detection_plot(cluster_data, patterns.get('anomalies', {})),
            self.visualization_engine.create_pattern_summary_charts(patterns)
        ]
        
        insights = self._generate_pattern_insights(patterns)
        
        return AnalysisResult(
            analysis_type=AnalysisType.PATTERN_RECOGNITION.value,
            data_type=DataType.MIXED.value,
            results=patterns,
            visualizations=visualizations,
            insights=insights,
            confidence_score=0.8,
            metadata={'clustering_algorithm': 'K-means', 'anomaly_detection': 'Isolation Forest'},
            timestamp=datetime.now().isoformat()
        )
    
    # Helper methods
    def _extract_text_from_data(self, data: Any) -> List[str]:
        """Extract text content from various data formats"""
        texts = []
        
        if isinstance(data, str):
            texts = [data]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    # Extract text from dict values
                    for value in item.values():
                        if isinstance(value, str):
                            texts.append(value)
        elif isinstance(data, pd.DataFrame):
            # Extract text from all string columns
            for col in data.select_dtypes(include=['object']).columns:
                texts.extend(data[col].dropna().astype(str).tolist())
        
        return texts
    
    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert various data formats to pandas DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame({'values': data})
        else:
            return pd.DataFrame({'data': [data]})
    
    def _extract_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numerical features from text or mixed data"""
        new_df = df.copy()
        
        for col in df.select_dtypes(include=['object']).columns:
            # Extract numbers from text
            new_df[f'{col}_length'] = df[col].astype(str).str.len()
            new_df[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
            
            # Extract specific numerical patterns
            numbers = df[col].astype(str).str.extractall(r'(\d+)')[0].astype(float)
            if not numbers.empty:
                new_df[f'{col}_extracted_numbers'] = numbers.groupby(level=0).mean()
        
        return new_df
    
    def _extract_entities(self, text: str) -> Dict[str, int]:
        """Simple entity extraction (can be enhanced with NLP libraries)"""
        # Simple patterns for common entities
        patterns = {
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'phone_numbers': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        }
        
        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            entities[entity_type] = len(matches)
        
        return entities
    
    # Insight generation methods
    def _generate_content_insights(self, result: ContentAnalysisResult) -> List[str]:
        """Generate insights from content analysis"""
        insights = []
        
        # Word frequency insights
        if result.word_frequency:
            top_word = max(result.word_frequency, key=result.word_frequency.get)
            insights.append(f"Most frequent word: '{top_word}' appears {result.word_frequency[top_word]} times")
        
        # Readability insights
        if result.readability_scores:
            ease_score = result.readability_scores.get('flesch_reading_ease', 0)
            if ease_score > 70:
                insights.append("Text has high readability - easily understood by general audience")
            elif ease_score < 30:
                insights.append("Text has low readability - may be difficult for general audience")
        
        # Theme insights
        if result.key_themes:
            insights.append(f"Key themes identified: {', '.join(result.key_themes[:5])}")
        
        # Clustering insights
        if result.text_clusters:
            largest_cluster = max(result.text_clusters, key=lambda x: x['size'])
            insights.append(f"Largest text cluster contains {largest_cluster['size']} similar documents")
        
        return insights
    
    def _generate_statistical_insights(self, summary: StatisticalSummary, distributions: Dict) -> List[str]:
        """Generate insights from statistical analysis"""
        insights = []
        
        # Correlation insights
        if summary.correlation_matrix:
            correlations = []
            for var1, var1_corrs in summary.correlation_matrix.items():
                for var2, corr in var1_corrs.items():
                    if var1 != var2 and abs(corr) > 0.7:
                        correlations.append(f"{var1} and {var2} are strongly correlated ({corr:.2f})")
            
            if correlations:
                insights.extend(correlations[:3])  # Top 3 correlations
            else:
                insights.append("No strong correlations found between variables")
        
        # Distribution insights
        for var, dist_info in distributions.items():
            if abs(dist_info['skewness']) > 1:
                skew_type = "right-skewed" if dist_info['skewness'] > 0 else "left-skewed"
                insights.append(f"{var} distribution is {skew_type}")
            
            if dist_info['normality_test'] < 0.05:
                insights.append(f"{var} does not follow normal distribution (p={dist_info['normality_test']:.3f})")
        
        return insights
    
    def _generate_trend_insights(self, trends: Dict, seasonal_patterns: Dict) -> List[str]:
        """Generate insights from trend analysis"""
        insights = []
        
        for trend_name, trend_info in trends.items():
            direction = trend_info['direction']
            strength = trend_info['strength']
            
            if strength > 0.7:
                insights.append(f"Strong {direction} trend detected in {trend_name} (R²={trend_info['r_squared']:.2f})")
            elif strength > 0.3:
                insights.append(f"Moderate {direction} trend detected in {trend_name}")
            else:
                insights.append(f"Weak or no trend detected in {trend_name}")
        
        return insights
    
    def _generate_sentiment_insights(self, distribution: Counter, avg_score: float) -> List[str]:
        """Generate insights from sentiment analysis"""
        insights = []
        
        total = sum(distribution.values())
        if total > 0:
            positive_pct = distribution.get('positive', 0) / total * 100
            negative_pct = distribution.get('negative', 0) / total * 100
            
            if positive_pct > 60:
                insights.append(f"Predominantly positive sentiment ({positive_pct:.1f}% positive)")
            elif negative_pct > 60:
                insights.append(f"Predominantly negative sentiment ({negative_pct:.1f}% negative)")
            else:
                insights.append("Mixed sentiment with no clear dominant tone")
            
            insights.append(f"Average sentiment score: {avg_score:.3f}")
        
        return insights
    
    def _generate_network_insights(self, metrics: Dict) -> List[str]:
        """Generate insights from network analysis"""
        insights = []
        
        if metrics:
            insights.append(f"Network contains {metrics['num_nodes']} nodes and {metrics['num_edges']} edges")
            insights.append(f"Network density: {metrics['density']:.3f}")
            
            if metrics['density'] > 0.5:
                insights.append("High density network - most nodes are connected")
            elif metrics['density'] < 0.1:
                insights.append("Sparse network - few connections between nodes")
            
            if 'centrality' in metrics and metrics['centrality']:
                # Find most central node
                degree_centrality = metrics['centrality'].get('degree_centrality', {})
                if degree_centrality:
                    most_central = max(degree_centrality, key=degree_centrality.get)
                    insights.append(f"Most connected node: {most_central}")
        
        return insights
    
    def _generate_comparative_insights(self, comparisons: Dict) -> List[str]:
        """Generate insights from comparative analysis"""
        insights = []
        
        for variable, comparison in comparisons.items():
            if 'statistical_test' in comparison:
                test_result = comparison['statistical_test']
                if test_result.get('significant', False):
                    insights.append(f"Significant differences found in {variable} between groups (p={test_result['p_value']:.3f})")
                else:
                    insights.append(f"No significant differences found in {variable} between groups")
            
            # Find group with highest mean
            group_means = {group: stats['mean'] for group, stats in comparison.items() 
                          if isinstance(stats, dict) and 'mean' in stats}
            if group_means:
                highest_group = max(group_means, key=group_means.get)
                insights.append(f"Highest average {variable}: {highest_group} ({group_means[highest_group]:.2f})")
        
        return insights
    
    def _generate_pattern_insights(self, patterns: Dict) -> List[str]:
        """Generate insights from pattern recognition"""
        insights = []
        
        if 'clusters' in patterns:
            cluster_info = patterns['clusters']
            insights.append(f"Identified {cluster_info['n_clusters']} distinct patterns in the data")
            
            if 'cluster_characteristics' in patterns:
                cluster_chars = patterns['cluster_characteristics']
                largest_cluster = max(cluster_chars.values(), key=lambda x: x['size'])
                insights.append(f"Largest pattern represents {largest_cluster['percentage']:.1f}% of data")
        
        if 'anomalies' in patterns:
            anomaly_info = patterns['anomalies']
            anomaly_pct = anomaly_info['anomaly_percentage']
            if anomaly_pct > 10:
                insights.append(f"High anomaly rate detected ({anomaly_pct:.1f}% of data points)")
            elif anomaly_pct > 0:
                insights.append(f"Some anomalies detected ({anomaly_pct:.1f}% of data points)")
            else:
                insights.append("No significant anomalies detected")
        
        if 'frequent_patterns' in patterns:
            insights.append("Frequent patterns identified in categorical variables")
        
        return insights

    async def synthesize_analysis_results(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Synthesize multiple analysis results into comprehensive insights"""
        synthesis = {
            'summary': {},
            'key_findings': [],
            'recommendations': [],
            'confidence_scores': {},
            'cross_analysis_insights': []
        }
        
        # Aggregate confidence scores
        for result in results:
            synthesis['confidence_scores'][result.analysis_type] = result.confidence_score
        
        # Extract key findings from each analysis
        for result in results:
            synthesis['key_findings'].extend(result.insights)
        
        # Cross-analysis insights
        if len(results) > 1:
            analysis_types = [r.analysis_type for r in results]
            
            if 'content_analysis' in analysis_types and 'sentiment_analysis' in analysis_types:
                synthesis['cross_analysis_insights'].append(
                    "Content and sentiment analysis reveal comprehensive text understanding"
                )
            
            if 'statistical_analysis' in analysis_types and 'trend_analysis' in analysis_types:
                synthesis['cross_analysis_insights'].append(
                    "Statistical and trend analysis provide temporal insights into data patterns"
                )
        
        # Generate recommendations based on findings
        synthesis['recommendations'] = await self._generate_recommendations(results)
        
        return synthesis
    
    async def _generate_recommendations(self, results: List[AnalysisResult]) -> List[str]:
        """Generate actionable recommendations based on analysis results"""
        recommendations = []
        
        for result in results:
            if result.analysis_type == 'content_analysis':
                recommendations.append("Consider content optimization based on key themes identified")
            elif result.analysis_type == 'sentiment_analysis':
                recommendations.append("Monitor sentiment trends for reputation management")
            elif result.analysis_type == 'statistical_analysis':
                recommendations.append("Use correlation insights for predictive modeling")
            elif result.analysis_type == 'trend_analysis':
                recommendations.append("Leverage trend patterns for forecasting and planning")
        
        return recommendations


class VisualizationEngine:
    """Engine for creating various types of visualizations"""
    
    def create_word_cloud(self, word_freq: Dict[str, int]) -> Dict[str, Any]:
        """Create word cloud visualization"""
        return {
            'type': 'wordcloud',
            'data': word_freq,
            'config': {
                'width': 800,
                'height': 400,
                'background_color': 'white'
            }
        }
    
    def create_theme_bar_chart(self, themes: List[str]) -> Dict[str, Any]:
        """Create bar chart for themes"""
        return {
            'type': 'bar_chart',
            'data': {'themes': themes, 'scores': [1] * len(themes)},
            'config': {
                'title': 'Key Themes',
                'x_label': 'Themes',
                'y_label': 'Relevance'
            }
        }
    
    def create_readability_gauge(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Create gauge chart for readability scores"""
        return {
            'type': 'gauge',
            'data': scores,
            'config': {
                'title': 'Readability Scores',
                'ranges': {'easy': [70, 100], 'medium': [30, 70], 'difficult': [0, 30]}
            }
        }
    
    def create_correlation_heatmap(self, correlation_matrix: Optional[Dict]) -> Dict[str, Any]:
        """Create correlation heatmap"""
        if not correlation_matrix:
            return {'type': 'placeholder', 'message': 'No correlation data available'}
        
        return {
            'type': 'heatmap',
            'data': correlation_matrix,
            'config': {
                'title': 'Correlation Matrix',
                'colorscale': 'RdBu',
                'symmetric': True
            }
        }
    
    def create_distribution_plots(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Create distribution plots for numerical columns"""
        return {
            'type': 'histogram_grid',
            'data': {col: df[col].dropna().tolist() for col in columns if col in df.columns},
            'config': {
                'title': 'Data Distributions',
                'bins': 20
            }
        }
    
    def create_statistical_summary_table(self, summary: StatisticalSummary) -> Dict[str, Any]:
        """Create statistical summary table"""
        return {
            'type': 'table',
            'data': asdict(summary),
            'config': {
                'title': 'Statistical Summary',
                'format': 'statistical'
            }
        }
    
    def create_trend_lines(self, df: pd.DataFrame, temporal_columns: List[str]) -> Dict[str, Any]:
        """Create trend line charts"""
        return {
            'type': 'line_chart',
            'data': {'temporal_columns': temporal_columns},
            'config': {
                'title': 'Trend Analysis',
                'x_label': 'Time',
                'y_label': 'Value'
            }
        }
    
    def create_seasonal_decomposition(self, df: pd.DataFrame, temporal_columns: List[str]) -> Dict[str, Any]:
        """Create seasonal decomposition plots"""
        return {
            'type': 'decomposition',
            'data': {'temporal_columns': temporal_columns},
            'config': {
                'title': 'Seasonal Decomposition',
                'components': ['trend', 'seasonal', 'residual']
            }
        }
    
    def create_trend_summary_table(self, trends: Dict) -> Dict[str, Any]:
        """Create trend summary table"""
        return {
            'type': 'table',
            'data': trends,
            'config': {
                'title': 'Trend Summary',
                'format': 'trend'
            }
        }
    
    def create_sentiment_pie_chart(self, sentiment_dist: Counter) -> Dict[str, Any]:
        """Create pie chart for sentiment distribution"""
        return {
            'type': 'pie_chart',
            'data': dict(sentiment_dist),
            'config': {
                'title': 'Sentiment Distribution',
                'colors': {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            }
        }
    
    def create_sentiment_timeline(self, sentiments: List[str], scores: List[float]) -> Dict[str, Any]:
        """Create sentiment timeline"""
        return {
            'type': 'timeline',
            'data': {'sentiments': sentiments, 'scores': scores},
            'config': {
                'title': 'Sentiment Timeline',
                'y_label': 'Sentiment Score'
            }
        }
    
    def create_sentiment_word_cloud(self, texts: List[str], sentiments: List[str]) -> Dict[str, Any]:
        """Create sentiment-based word cloud"""
        return {
            'type': 'sentiment_wordcloud',
            'data': {'texts': texts, 'sentiments': sentiments},
            'config': {
                'title': 'Sentiment Word Cloud',
                'sentiment_colors': True
            }
        }
    
    def create_network_graph(self, graph) -> Dict[str, Any]:
        """Create network graph visualization"""
        if hasattr(graph, 'nodes') and hasattr(graph, 'edges'):
            nodes = list(graph.nodes())
            edges = list(graph.edges())
        else:
            nodes = []
            edges = []
        
        return {
            'type': 'network',
            'data': {'nodes': nodes, 'edges': edges},
            'config': {
                'title': 'Network Graph',
                'layout': 'spring'
            }
        }
    
    def create_centrality_charts(self, centrality_measures: Dict) -> Dict[str, Any]:
        """Create centrality measure charts"""
        return {
            'type': 'centrality_charts',
            'data': centrality_measures,
            'config': {
                'title': 'Node Centrality Measures',
                'chart_types': ['bar', 'scatter']
            }
        }
    
    def create_network_metrics_table(self, metrics: Dict) -> Dict[str, Any]:
        """Create network metrics table"""
        return {
            'type': 'table',
            'data': metrics,
            'config': {
                'title': 'Network Metrics',
                'format': 'network'
            }
        }
    
    def create_comparison_box_plots(self, df: pd.DataFrame, group_column: Optional[str]) -> Dict[str, Any]:
        """Create box plots for group comparison"""
        return {
            'type': 'box_plot',
            'data': {'dataframe': df.to_dict(), 'group_column': group_column},
            'config': {
                'title': 'Group Comparison',
                'show_outliers': True
            }
        }
    
    def create_comparison_bar_charts(self, comparisons: Dict) -> Dict[str, Any]:
        """Create bar charts for comparisons"""
        return {
            'type': 'comparison_bars',
            'data': comparisons,
            'config': {
                'title': 'Comparative Analysis',
                'error_bars': True
            }
        }
    
    def create_statistical_test_summary(self, comparisons: Dict) -> Dict[str, Any]:
        """Create statistical test summary"""
        return {
            'type': 'table',
            'data': comparisons,
            'config': {
                'title': 'Statistical Test Results',
                'format': 'statistical_tests'
            }
        }
    
    def create_cluster_scatter_plot(self, data: pd.DataFrame, cluster_info: Dict) -> Dict[str, Any]:
        """Create scatter plot showing clusters"""
        return {
            'type': 'scatter_plot',
            'data': {'dataframe': data.to_dict(), 'clusters': cluster_info},
            'config': {
                'title': 'Cluster Analysis',
                'color_by_cluster': True
            }
        }
    
    def create_anomaly_detection_plot(self, data: pd.DataFrame, anomaly_info: Dict) -> Dict[str, Any]:
        """Create anomaly detection visualization"""
        return {
            'type': 'anomaly_plot',
            'data': {'dataframe': data.to_dict(), 'anomalies': anomaly_info},
            'config': {
                'title': 'Anomaly Detection',
                'highlight_anomalies': True
            }
        }
    
    def create_pattern_summary_charts(self, patterns: Dict) -> Dict[str, Any]:
        """Create pattern summary visualizations"""
        return {
            'type': 'pattern_summary',
            'data': patterns,
            'config': {
                'title': 'Pattern Recognition Summary',
                'multiple_charts': True
            }
        }
