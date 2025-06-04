"""
Advanced Task Orchestrator - The Central Brain of the Agent System
This orchestrator manages dynamic task planning, agent routing, and intelligent execution
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime

from sources.utility import pretty_print, animate_thinking
from sources.logger import Logger

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class ExecutionMode(Enum):
    FAST = "fast"
    DEEP_RESEARCH = "deep_research"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class Task:
    id: str
    description: str
    agent_type: str
    priority: int
    complexity: TaskComplexity
    dependencies: List[str]
    context: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    result: Any = None
    error: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ExecutionPlan:
    id: str
    query: str
    mode: ExecutionMode
    complexity: TaskComplexity
    tasks: List[Task]
    estimated_duration: int
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class TaskOrchestrator:
    """
    Advanced Task Orchestrator that manages the entire agent execution pipeline
    """
    
    def __init__(self, agents_registry: Dict, llm_provider=None):
        self.agents_registry = agents_registry
        self.llm_provider = llm_provider
        self.logger = Logger("task_orchestrator.log")
        
        # Execution state
        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_history: List[ExecutionPlan] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
          # Performance metrics
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        self.execution_stats = {
            "total_plans": 0,
            "successful_plans": 0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "tasks_executed": 0,
            "total_execution_time": 0.0,
            "fast_mode_executions": 0,
            "deep_mode_executions": 0,
            "average_duration": 0
        }
        
        # Configuration
        self.max_concurrent_tasks = 3
        self.task_timeout = 300  # 5 minutes
        self.quality_threshold = 0.7
        
        # Orchestrator start time for uptime calculation
        self.orchestrator_start_time = time.time()
        
    async def analyze_query_complexity(self, query: str) -> Tuple[TaskComplexity, ExecutionMode]:
        """
        Analyze query to determine complexity and recommended execution mode
        """
        query_lower = query.lower()
        
        # Complexity indicators
        complex_keywords = [
            "comprehensive", "detailed", "in-depth", "research", "analyze", "report",
            "compare", "evaluate", "investigate", "study", "examine", "assess"
        ]
        
        expert_keywords = [
            "financial analysis", "investment recommendation", "market research",
            "technical analysis", "academic research", "scientific study"
        ]
        
        # Mode indicators
        deep_research_indicators = [
            "report", "comprehensive", "detailed analysis", "investment",
            "research paper", "market analysis", "in-depth study"
        ]
        
        # Calculate complexity
        if any(keyword in query_lower for keyword in expert_keywords):
            complexity = TaskComplexity.EXPERT
        elif any(keyword in query_lower for keyword in complex_keywords):
            complexity = TaskComplexity.COMPLEX
        elif len(query.split()) > 10:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.SIMPLE
            
        # Determine execution mode
        if any(indicator in query_lower for indicator in deep_research_indicators):
            mode = ExecutionMode.DEEP_RESEARCH
        else:
            mode = ExecutionMode.FAST
            
        self.logger.info(f"Query analysis: complexity={complexity.value}, mode={mode.value}")
        return complexity, mode
    
    async def create_execution_plan(self, query: str) -> ExecutionPlan:
        """
        Create a dynamic execution plan based on query analysis
        """
        complexity, mode = await self.analyze_query_complexity(query)
        
        plan_id = f"plan_{int(time.time())}"
        tasks = []
        
        if mode == ExecutionMode.FAST:
            tasks = await self._create_fast_mode_tasks(query, complexity)
        else:
            tasks = await self._create_deep_research_tasks(query, complexity)
        
        # Estimate duration based on task complexity
        estimated_duration = self._estimate_execution_duration(tasks)
        
        plan = ExecutionPlan(
            id=plan_id,
            query=query,
            mode=mode,
                    complexity=complexity,
            tasks=tasks,
            estimated_duration=estimated_duration
        )
        
        self.current_plan = plan
        self.execution_history.append(plan)
        self.execution_stats["total_plans"] += 1
        
        pretty_print(f"Created execution plan: {len(tasks)} tasks, estimated {estimated_duration}s", "success")
        return plan
    
    async def _create_fast_mode_tasks(self, query: str, complexity: TaskComplexity) -> List[Task]:
        """Create tasks for fast execution mode"""
        tasks = []
        
        # Task 1: Quick search
        search_task = Task(
            id="search_001",
            description=f"Quick search for: {query}",
            agent_type="enhanced_search_agent",
            priority=1,
            complexity=complexity,
            dependencies=[],
            context={"query": query, "max_results": 5, "fast_mode": True}
        )
        tasks.append(search_task)
        
        # Task 2: Content analysis
        analysis_task = Task(
            id="analysis_001",
            description="Analyze and summarize findings",
            agent_type="casual_agent",
            priority=2,
            complexity=complexity,
                            dependencies=["search_001"],
            context={"query": query, "analysis_type": "quick_summary"}
        )
        tasks.append(analysis_task)
        
        return tasks

    async def _create_deep_research_tasks(self, query: str, complexity: TaskComplexity) -> List[Task]:
        """Create tasks for deep research mode"""
        tasks = []
        
        # Phase 1: Initial Search
        initial_search = Task(
            id="search_initial",
            description=f"Comprehensive initial search for: {query}",
            agent_type="enhanced_search_agent",
            priority=1,
            complexity=complexity,
            dependencies=[],
            context={
                "query": query, 
                "engines": ["duckduckgo", "brave", "bing"],
                "max_results": 15,
                "deep_mode": True
            }
        )
        tasks.append(initial_search)
        
        # Phase 2: Web Content Extraction
        web_extraction = Task(
            id="web_extraction",
            description="Extract detailed content from top sources",
            agent_type="enhanced_web_agent",
            priority=2,
            complexity=complexity,
            dependencies=["search_initial"],
            context={
                "query": query,
                "max_sources": 8,
                "screenshot_analysis": True,
                "extract_media": True,
                "deep_mode": True
            }
        )
        tasks.append(web_extraction)
          # Phase 3: Specialized searches based on complexity
        if complexity == TaskComplexity.EXPERT:
            # Academic/Research search
            academic_search = Task(
                id="search_academic",
                description="Academic and research-focused search",
                agent_type="enhanced_search_agent",
                priority=2,
                complexity=complexity,
                dependencies=["search_initial"],
                context={
                    "query": f"academic research {query}",
                    "engines": ["duckduckgo", "startpage", "internet_archive"],
                    "max_results": 10,
                    "focus": "academic"
                }
            )
            tasks.append(academic_search)
            
            # Financial/Market search if relevant
            if any(keyword in query.lower() for keyword in ["price", "stock", "market", "finance", "company"]):
                financial_search = Task(
                    id="search_financial",
                    description="Financial and market data search",
                    agent_type="enhanced_search_agent",
                    priority=2,
                    complexity=complexity,
                    dependencies=["search_initial"],
                    context={
                        "query": query,
                        "engines": ["yahoo_finance"],
                        "max_results": 5,
                        "focus": "financial"
                    }
                )
                tasks.append(financial_search)
        
        # Phase 4: Data Analysis and Visualization
        if any(keyword in query.lower() for keyword in ["data", "analysis", "chart", "graph", "trend", "statistics"]):
            coding_analysis = Task(
                id="coding_analysis",
                description="Data analysis and visualization",
                agent_type="enhanced_coding_agent",
                priority=3,
                complexity=complexity,
                dependencies=["web_extraction"],
                context={
                    "query": query,
                    "create_visualizations": True,
                    "data_analysis": True,
                    "generate_notebook": True
                }
            )
            tasks.append(coding_analysis)
        
        # Phase 5: Deep Analysis
        deep_analysis = Task(
            id="analysis_deep",
            description="Comprehensive analysis and synthesis",
            agent_type="analysis_agent",
            priority=4,
            complexity=complexity,
            dependencies=["web_extraction"] + (["search_academic"] if complexity == TaskComplexity.EXPERT else []),
            context={
                "query": query,
                "synthesis_mode": "comprehensive",
                "include_citations": True,
                "cross_reference": True
            }
        )
        tasks.append(deep_analysis)
        
        # Phase 6: Quality Validation
        quality_check = Task(
            id="quality_validation",
            description="Validate and verify findings",
            agent_type="quality_agent",
            priority=5,
            complexity=complexity,
            dependencies=["analysis_deep"],
            context={
                "query": query,
                "fact_check": True,
                "source_verification": True,
                "bias_analysis": True
            }
        )
        tasks.append(quality_check)
        
        # Phase 7: Report Generation
        report_generation = Task(
            id="report_generation",
            description="Generate comprehensive research report",
            agent_type="enhanced_report_agent",
            priority=6,
            complexity=complexity,
            dependencies=["quality_validation"],
            context={
                "query": query,
                "format": "comprehensive",
                "include_visualizations": True,
                "generate_pdf": True,
                "executive_summary": True
            }
        )
        tasks.append(report_generation)
        
        return tasks
    
    def _estimate_execution_duration(self, tasks: List[Task]) -> int:
        """Estimate execution duration based on task complexity"""
        base_duration = 0
        
        for task in tasks:
            if task.complexity == TaskComplexity.SIMPLE:
                base_duration += 10
            elif task.complexity == TaskComplexity.MODERATE:
                base_duration += 30
            elif task.complexity == TaskComplexity.COMPLEX:
                base_duration += 60
            else:  # EXPERT
                base_duration += 120
                
        # Add overhead for task coordination
        return int(base_duration * 1.2)
    
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute the complete execution plan"""
        start_time = time.time()
        results = {}
        executed_tasks = set()
        
        try:
            pretty_print(f"Starting execution of {len(plan.tasks)} tasks", "info")
            
            # Execute tasks in dependency order
            while len(executed_tasks) < len(plan.tasks):
                ready_tasks = [
                    task for task in plan.tasks 
                    if task.id not in executed_tasks 
                    and all(dep in executed_tasks for dep in task.dependencies)
                ]
                
                if not ready_tasks:
                    remaining_tasks = [task for task in plan.tasks if task.id not in executed_tasks]
                    unmet_deps = []
                    for task in remaining_tasks:
                        unmet_deps.extend([dep for dep in task.dependencies if dep not in executed_tasks])
                    raise Exception(f"Circular dependency or missing tasks detected. Unmet dependencies: {unmet_deps}")
                
                # Sort by priority
                ready_tasks.sort(key=lambda x: x.priority)
                
                # Execute ready tasks
                for task in ready_tasks:
                    pretty_print(f"Executing task: {task.id} - {task.description}", "info")
                    
                    try:
                        task_result = await self._execute_task(task, results)
                        results[task.id] = task_result
                        executed_tasks.add(task.id)
                        
                        # Update execution stats
                        self.execution_stats["tasks_executed"] += 1
                        
                        pretty_print(f"Completed task: {task.id}", "success")
                        
                    except Exception as e:
                        pretty_print(f"Task {task.id} failed: {str(e)}", "error")
                        
                        # Handle critical task failure
                        if task.priority <= 2:  # Critical task
                            raise Exception(f"Critical task {task.id} failed: {str(e)}")
                        else:
                            # Non-critical task failure, continue
                            results[task.id] = {"error": str(e), "status": "failed"}
                            executed_tasks.add(task.id)
            
            execution_time = time.time() - start_time
            
            # Compile final results
            final_result = {
                "execution_mode": plan.mode.value,
                "total_tasks": len(plan.tasks),
                "successful_tasks": len([r for r in results.values() if not isinstance(r, dict) or r.get("status") != "failed"]),
                "execution_time": execution_time,
                "results": results,
                "summary": await self._compile_summary(results, plan),
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_stats["total_execution_time"] += execution_time
            
            pretty_print(f"Execution completed in {execution_time:.2f}s", "success")
            return final_result
            
        except Exception as e:
            pretty_print(f"Execution failed: {str(e)}", "error")
            raise
    
    async def _execute_task(self, task: Task, previous_results: Dict[str, Any]) -> Any:
        """Execute a single task"""
        
        # Prepare context with previous results
        context = task.context.copy()
        context["previous_results"] = {dep: previous_results.get(dep) for dep in task.dependencies}
        
        # Route to appropriate agent
        if task.agent_type == "enhanced_search_agent":
            from ..agents.enhanced_search_agent import EnhancedSearchAgent
            agent = EnhancedSearchAgent()
            return await agent.search(context["query"], context)
            
        elif task.agent_type == "enhanced_web_agent":
            from ..agents.enhanced_web_agent import EnhancedWebAgent
            agent = EnhancedWebAgent()
            return await agent.extract_content(context)
            
        elif task.agent_type == "enhanced_coding_agent":
            from ..agents.enhanced_coding_agent import EnhancedCodingAgent
            agent = EnhancedCodingAgent()
            return await agent.analyze_data(context)
            
        elif task.agent_type == "analysis_agent":
            return await self._execute_analysis_agent(context)
            
        elif task.agent_type == "quality_agent":
            return await self._execute_quality_agent(context)
            
        elif task.agent_type == "enhanced_report_agent":
            return await self._execute_report_agent(context)
            
        elif task.agent_type in ["search_agent", "casual_agent"]:
            # Fallback to existing agents
            return await self._execute_legacy_agent(task.agent_type, context)
            
        else:
            raise ValueError(f"Unknown agent type: {task.agent_type}")
    
    async def _execute_analysis_agent(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis agent task"""
        try:
            from ..agents.analysis_agent import AnalysisAgent
            agent = AnalysisAgent()
            
            # Call the main analysis method
            if hasattr(agent, 'analyze_comprehensive'):
                result = await agent.analyze_comprehensive(context)
                
                # Convert dataclass to dict for JSON serialization
                if hasattr(result, '__dict__'):
                    return asdict(result)
                else:
                    return result
            else:
                # Fallback to basic analysis
                return await self._basic_analysis_fallback(context)
                
        except ImportError as e:
            self.logger.warning(f"Analysis agent not available: {str(e)}")
            return await self._basic_analysis_fallback(context)
        except Exception as e:
            self.logger.error(f"Analysis agent execution failed: {str(e)}")
            raise
    
    async def _execute_quality_agent(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quality agent task"""
        try:
            from ..agents.quality_agent import QualityAgent
            agent = QualityAgent()
            
            # Call the main validation method
            if hasattr(agent, 'validate_comprehensive'):
                result = await agent.validate_comprehensive(context)
                
                # Convert dataclass to dict for JSON serialization
                if hasattr(result, '__dict__'):
                    return asdict(result)
                else:
                    return result
            else:
                # Fallback to basic quality check
                return await self._basic_quality_fallback(context)
                
        except ImportError as e:
            self.logger.warning(f"Quality agent not available: {str(e)}")
            return await self._basic_quality_fallback(context)
        except Exception as e:
            self.logger.error(f"Quality agent execution failed: {str(e)}")
            raise
    
    async def _basic_analysis_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis implementation"""
        query = context.get("query", "")
        previous_results = context.get("previous_results", {})
        
        # Compile all available data
        all_data = []
        sources_count = 0
        
        for result in previous_results.values():
            if isinstance(result, dict):
                if "data" in result:
                    all_data.extend(result["data"] if isinstance(result["data"], list) else [result["data"]])
                if "results" in result:
                    all_data.extend(result["results"] if isinstance(result["results"], list) else [result["results"]])
                if "sources" in result:
                    sources_count += len(result["sources"]) if isinstance(result["sources"], list) else 1
            elif isinstance(result, list):
                all_data.extend(result)
        
        # Basic analysis
        analysis = {
            "query": query,
            "executive_summary": f"Analysis of {len(all_data)} data points from {sources_count} sources reveals comprehensive insights into {query}.",
            "key_insights": [
                {
                    "category": "Data Summary",
                    "insight": f"Collected {len(all_data)} data points from {sources_count} sources",
                    "confidence": 0.8,
                    "evidence": [f"Total data points: {len(all_data)}"],
                    "impact_level": "medium"
                },
                {
                    "category": "Research Coverage",
                    "insight": f"Research covers multiple aspects of {query}",
                    "confidence": 0.75,
                    "evidence": [f"Sources analyzed: {sources_count}"],
                    "impact_level": "medium"
                }
            ],
            "patterns": [
                {
                    "pattern_type": "data_collection",
                    "description": "Successful data collection across multiple sources",
                    "frequency": sources_count,
                    "confidence": 0.8
                }
            ],
            "trends": {
                "data_availability": "Good",
                "source_diversity": "Moderate" if sources_count > 2 else "Limited"
            },
            "sentiment_analysis": {
                "dominant_sentiment": "neutral",
                "confidence": 0.6
            },
            "statistical_summary": {
                "data_points": len(all_data),
                "sources_count": sources_count,
                "analysis_completeness": 0.75
            },
            "confidence_score": 0.75,
            "completeness_score": 0.8,
            "data_quality_score": 0.85,
            "recommendations": [
                f"Data collection successful for {query}",
                "Consider additional validation for critical decisions",
                "Results provide good foundation for decision making"
            ],
            "limitations": [
                "Analysis based on available data only",
                "Real-time limitations may apply"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis
    
    async def _basic_quality_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback quality validation implementation"""
        previous_results = context.get("previous_results", {})
        
        # Basic quality assessment
        successful_tasks = sum(1 for r in previous_results.values() 
                              if not (isinstance(r, dict) and r.get("status") == "failed"))
        total_tasks = len(previous_results)
        
        # Count sources
        all_sources = set()
        for result in previous_results.values():
            if isinstance(result, dict) and "sources" in result:
                if isinstance(result["sources"], list):
                    all_sources.update(result["sources"])
                else:
                    all_sources.add(str(result["sources"]))
        
        source_credibility_score = min(1.0, len(all_sources) / 3)  # Normalize to 3 sources
        accuracy_score = successful_tasks / total_tasks if total_tasks > 0 else 0.5
        completeness_score = min(1.0, successful_tasks / 5)  # Normalize to 5 successful tasks
        
        quality_metrics = {
            "overall_quality_score": statistics.mean([source_credibility_score, accuracy_score, completeness_score]),
            "metrics": [
                {
                    "metric_name": "Source Credibility",
                    "score": source_credibility_score,
                    "confidence": 0.7,
                    "details": f"Based on {len(all_sources)} unique sources"
                },
                {
                    "metric_name": "Task Success Rate",
                    "score": accuracy_score,
                    "confidence": 0.9,
                    "details": f"{successful_tasks}/{total_tasks} tasks completed successfully"
                },
                {
                    "metric_name": "Data Completeness",
                    "score": completeness_score,
                    "confidence": 0.8,
                    "details": f"Completeness based on task success rate"
                }
            ],
            "source_credibility": [
                {
                    "domain": "multiple_sources",
                    "credibility_score": source_credibility_score,
                    "source_count": len(all_sources)
                }
            ],
            "fact_check_results": [],
            "bias_analysis": {
                "overall_bias_score": 0.3,  # Moderate bias assumed
                "source_diversity": source_credibility_score,
                "assessment": "Basic bias assessment completed"
            },
            "completeness_assessment": {
                "completeness_score": completeness_score,
                "task_success_rate": accuracy_score
            },
            "accuracy_indicators": {
                "task_success_count": successful_tasks,
                "source_count": len(all_sources)
            },
            "recency_analysis": {
                "freshness_score": 0.8,  # Assume recent data
                "assessment": "Basic recency assessment"
            },
            "recommendations": [
                f"Quality assessment based on {total_tasks} completed tasks",
                f"Data collected from {len(all_sources)} sources",
                "Consider additional validation for critical applications"
            ],
            "limitations": [
                "Basic quality assessment only",
                "Limited fact-checking capabilities in fallback mode",
                "Source credibility based on quantity, not detailed analysis"
            ],
            "confidence_interval": (0.6, 0.9),
            "timestamp": datetime.now().isoformat()
        }
        
        return quality_metrics
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get orchestrator execution statistics"""
        return {
            **self.execution_stats,
            "uptime": time.time() - self.orchestrator_start_time,
            "average_plan_size": (
                self.execution_stats["tasks_executed"] / self.execution_stats["total_plans"]
                if self.execution_stats["total_plans"] > 0 else 0
            ),
            "average_execution_time": (
                self.execution_stats["total_execution_time"] / self.execution_stats["total_plans"]
                if self.execution_stats["total_plans"] > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset execution statistics"""
        self.execution_stats = {
            "total_plans": 0,
            "tasks_executed": 0,
            "total_execution_time": 0.0,
            "fast_mode_executions": 0,
            "deep_mode_executions": 0
        }
        self.orchestrator_start_time = time.time()
        pretty_print("Orchestrator statistics reset", "info")

# Utility function for pretty printing
def pretty_print(message: str, level: str = "info"):
    """Enhanced pretty print with colors and timestamps"""
    colors = {
        "info": "\033[36m",      # Cyan
        "success": "\033[92m",   # Green
        "warning": "\033[93m",   # Yellow
        "error": "\033[91m",     # Red
        "reset": "\033[0m"       # Reset
    }
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    color = colors.get(level, colors["info"])
    reset = colors["reset"]
    
    print(f"{color}[{timestamp}] [{level.upper()}] {message}{reset}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_orchestrator():
        """Test the task orchestrator"""
        orchestrator = TaskOrchestrator()
        
        # Test fast mode
        print("\n=== Testing Fast Mode ===")
        fast_plan = await orchestrator.create_execution_plan(
            "What is artificial intelligence?", 
            ExecutionMode.FAST
        )
        print(f"Fast mode plan: {len(fast_plan.tasks)} tasks")
        
        # Test deep research mode
        print("\n=== Testing Deep Research Mode ===")
        deep_plan = await orchestrator.create_execution_plan(
            "Comprehensive analysis of renewable energy trends", 
            ExecutionMode.DEEP_RESEARCH
        )
        print(f"Deep research plan: {len(deep_plan.tasks)} tasks")
        
        # Print stats
        print("\n=== Orchestrator Stats ===")
        stats = orchestrator.get_execution_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    # Run the test
    asyncio.run(test_orchestrator())
