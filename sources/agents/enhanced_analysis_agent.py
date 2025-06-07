"""
Enhanced Analysis Agent - Advanced Data Analysis and Synthesis
This agent performs deep analysis, pattern recognition, statistical analysis, and data synthesis
"""
import asyncio
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease
from typing import Dict, List, Any
from sources.agents.agent import Agent, ExecutionManager # Import Agent
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    analysis_type: str
    summary: str
    key_points: List[str]
    confidence_score: float
    data: Dict[str, Any]

class EnhancedAnalysisAgent(Agent):
    """Enhanced Analysis Agent for comprehensive data analysis and synthesis."""
    
    def __init__(self, name="Enhanced Analysis Agent", prompt_path="prompts/base/analysis_agent.txt", provider=None, verbose=False):
        # *** CRITICAL FIX: Call the parent class constructor ***
        super().__init__(name, prompt_path, provider, verbose)
        self.role = "analysis"
        self.type = "enhanced_analysis_agent"
        self.logger = logger

    async def execute(self, prompt: str, execution_manager: 'ExecutionManager' = None) -> Dict[str, Any]:
        """Main execution entry point for the orchestrator."""
        self.logger.info(f"Starting analysis task for prompt: {prompt}")
        if not execution_manager or not execution_manager.execution_state:
            return {"success": False, "error": "Execution context not provided."}

        search_results = execution_manager.execution_state.get('search', {}).get('results', [])
        web_content_list = execution_manager.execution_state.get('browser', {}).get('extracted_content', [])
        
        all_text = " ".join([res.get('snippet', '') for res in search_results if res.get('snippet')])
        for content_item in web_content_list:
            all_text += " " + content_item.get('content', '')

        if not all_text.strip():
            summary = "No content available to analyze from previous steps."
            analysis_result = {"success": True, "summary": summary, "key_insights": []}
        else:
            analysis_result = self.perform_text_analysis(all_text)
            analysis_result["success"] = True

        execution_manager.update_state({"execution": {"agent_progress": {self.agent_name: {"status": "completed", "output": analysis_result['summary']}}}})
        
        return analysis_result

    def perform_text_analysis(self, text: str) -> Dict[str, Any]:
        """Performs NLP analysis on a block of text."""
        if not text:
            return {"summary": "No text provided.", "word_count": 0, "readability": 0, "key_themes": []}

        word_count = len(text.split())
        try:
            readability_score = flesch_reading_ease(text)
        except:
            readability_score = 0 # Fails on very short text

        try:
            vectorizer = TfidfVectorizer(max_features=5, stop_words='english')
            vectorizer.fit_transform([text])
            key_themes = vectorizer.get_feature_names_out()
        except ValueError:
            key_themes = []
        
        summary = f"Analyzed {word_count} words. Readability score is {readability_score:.2f}. Key themes: {', '.join(key_themes)}."

        return {
            "summary": summary,
            "word_count": word_count,
            "readability": readability_score,
            "key_themes": list(key_themes)
        }
