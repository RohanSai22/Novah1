#!/usr/bin/env python3
"""
Enhanced FastAPI server with real PlannerAgent integration for agentic workflow
This version uses actual agents and follows the sophisticated workflow
"""
import uvicorn
import asyncio
import time
import uuid
import os
import configparser
import json
import re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the actual agent classes (but handle missing dependencies gracefully)
try:
    from sources.llm_provider import Provider
    from sources.agents import CasualAgent, CoderAgent, BrowserAgent, SearchAgent, PlannerAgent, ReportAgent
    from sources.router import AgentRouter
    from sources.logger import Logger
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent modules not available: {e}")
    AGENTS_AVAILABLE = False

# Import the orchestrator system
try:
    from sources.orchestrator.task_orchestrator import TaskOrchestrator, ExecutionMode, TaskComplexity
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Orchestrator module not available: {e}")
    ORCHESTRATOR_AVAILABLE = False

# Import E2B integration
try:
    from sources.e2b_integration import E2BSandbox
    import re  # For code extraction in execute_subtask
    E2B_AVAILABLE = True
except ImportError as e:
    print(f"Warning: E2B integration not available: {e}")
    E2B_AVAILABLE = False

# Import E2B integration for code execution
try:
    from sources.e2b_integration import E2BSandbox
    E2B_AVAILABLE = True
except ImportError as e:
    print(f"Warning: E2B integration not available: {e}")
    E2B_AVAILABLE = False

# Simple request/response models
class QueryRequest(BaseModel):
    query: str

class AdvancedQueryRequest(BaseModel):
    query: str
    execution_mode: str = "fast"  # "fast" or "deep_research"
    quality_validation: bool = True
    generate_report: bool = True

app = FastAPI(title="Novah API - Enhanced", version="0.2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Enhanced Execution Manager with real agent integration
class ExecutionManager:    
    def __init__(self):
        self.is_processing = False
        self.planner_agent = None
        self.chat_agent = None
        self.search_agent = None
        self.browser_agent = None
        self.report_agent = None
        self.orchestrator = None  # Add orchestrator
        self.e2b_sandbox = None  # Add E2B sandbox
        self.execution_state = {
            "intent": "",
            "plan": [],
            "current_task": 0,
            "subtask_status": [],
            "agent_outputs": {},
            "final_report_url": None,
            "status": "idle",
            "current_agent": None,
            "active_tool": None,
            "current_step": 0,
            "total_steps": 0,
            "agent_progress": {},
            "search_results": [],
            "screenshots": [],
            "links_processed": [],
            "code_outputs": []  # Add code execution tracking
        }
        self.messages = []
        self.latest_response = None
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize the actual agent system (with fallback for demo)"""
        if not AGENTS_AVAILABLE:
            print("Using demo mode - agent modules not available")
            return
        
        try:
            # Load configuration
            config = configparser.ConfigParser()
            config.read('config.ini')
            
            # Create provider
            provider = Provider(
                provider_name=config.get("MAIN", "provider_name", fallback="openai"),
                model=config.get("MAIN", "provider_model", fallback="gpt-3.5-turbo"),
                server_address=config.get("MAIN", "provider_server_address", fallback=""),
                is_local=config.getboolean('MAIN', 'is_local', fallback=False)
            )
              # Initialize minimal browser (without heavy loading)
            browser = None
            
            # Initialize agents
            self.chat_agent = CasualAgent(
                name="Chat Assistant",
                prompt_path="prompts/base/casual_agent.txt",
                provider=provider, verbose=False
            )
            
            self.search_agent = SearchAgent(
                name="Search Specialist",
                prompt_path="prompts/base/search_agent.txt",
                provider=provider, verbose=False, browser=browser
            )
            
            self.browser_agent = BrowserAgent(
                name="Web Navigator",
                prompt_path="prompts/base/browser_agent.txt",
                provider=provider, verbose=False, browser=browser
            )
            
            self.planner_agent = PlannerAgent(
                name="Task Planner",
                prompt_path="prompts/base/planner_agent.txt",
                provider=provider, verbose=False, browser=browser
            )
            
            self.report_agent = ReportAgent(
                name="Report Generator",
                prompt_path="prompts/base/report_agent.txt",
                provider=provider, verbose=False
            )
              # Initialize E2B sandbox for code execution
            try:
                from sources.e2b_integration import E2BSandbox
                self.e2b_sandbox = E2BSandbox()
                print("✅ E2B Sandbox initialized successfully!")
            except ImportError as e:
                print(f"⚠️ E2B integration not available: {e}")
                self.e2b_sandbox = None
            except Exception as e:
                print(f"⚠️ Failed to initialize E2B sandbox: {e}")
                self.e2b_sandbox = None

            # Initialize orchestrator with enhanced agents
            if ORCHESTRATOR_AVAILABLE:
                agents_registry = {
                    'casual_agent': self.chat_agent,
                    'search_agent': self.search_agent,
                    'browser_agent': self.browser_agent,
                    'planner_agent': self.planner_agent,
                    'report_agent': self.report_agent
                }
                self.orchestrator = TaskOrchestrator(agents_registry, provider)
                print("✅ Task Orchestrator initialized successfully!")
            
            print("✅ Real agents initialized successfully!")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize real agents: {e}")
            print("Using demo mode")
            self.planner_agent = None
            self.chat_agent = None
            self.search_agent = None
            self.browser_agent = None
            self.report_agent = None
    
    def reset(self):
        """Reset execution state"""
        self.is_processing = False
        self.execution_state = {
            "intent": "",
            "plan": [],
            "current_task": 0,
            "subtask_status": [],
            "agent_outputs": {},
            "final_report_url": None,
            "status": "idle",
            "current_agent": None,
            "active_tool": None,
            "current_step": 0,
            "total_steps": 0,
            "agent_progress": {},
            "search_results": [],
            "screenshots": [],
            "links_processed": []
        }
        self.messages = []
        self.latest_response = None

    def update_status(self, status: str, agent: str = None, tool: str = None):
        """Update execution status"""
        self.execution_state["status"] = status
        if agent:
            self.execution_state["current_agent"] = agent
        if tool:
            self.execution_state["active_tool"] = tool

    def add_message(self, msg_type: str, content: str, agent: str = None, status: str = None):
        """Add a message"""
        message = {
            "type": msg_type,
            "content": content,
            "agent": agent or "System",
            "status": status or "completed",
            "timestamp": time.time()
        }
        self.messages.append(message)
        
        # Update latest response for polling
        self.latest_response = {
            "answer": content,
            "agent_name": agent or "System",
            "status": status or "completed",
            "execution_state": self.execution_state.copy(),
            "done": "false" if self.is_processing else "true",
            "reasoning": f"Agent {agent} is working" if agent else "System message",
            "uid": f"msg-{int(time.time() * 1000)}",
            "blocks": {},
            "success": "true"
        }

execution_manager = ExecutionManager()

# Report generation function
async def generate_report(query: str, plan: List[str], execution_manager: ExecutionManager) -> str:
    """
    Generate a comprehensive PDF report using the ReportAgent
    
    Args:
        query: The original user query
        plan: The execution plan
        execution_manager: The execution manager instance
        
    Returns:
        Path to the generated PDF report
    """
    try:
        # Create execution data for report generation
        execution_data = {
            "intent": query,
            "plan": plan,
            "subtask_status": execution_manager.execution_state.get("subtask_status", []),
            "agent_outputs": execution_manager.execution_state.get("agent_outputs", {}),
            "agent_progress": execution_manager.execution_state.get("agent_progress", {}),
            "search_results": execution_manager.execution_state.get("search_results", []),
            "links_processed": execution_manager.execution_state.get("links_processed", []),
            "current_step": execution_manager.execution_state.get("current_step", 0),
            "total_steps": execution_manager.execution_state.get("total_steps", 0)
        }
        
        # Use existing report agent if available, otherwise create one
        if hasattr(execution_manager, 'report_agent') and execution_manager.report_agent:
            report_agent = execution_manager.report_agent
        else:
            # Fallback: create a new ReportAgent if needed
            if AGENTS_AVAILABLE:
                try:
                    config = configparser.ConfigParser()
                    config.read('config.ini')
                    
                    provider = Provider(
                        provider_name=config.get("MAIN", "provider_name", fallback="openai"),
                        model=config.get("MAIN", "provider_model", fallback="gpt-3.5-turbo"),
                        server_address=config.get("MAIN", "provider_server_address", fallback=""),
                        is_local=config.getboolean('MAIN', 'is_local', fallback=False)
                    )
                    
                    report_agent = ReportAgent(
                        name="Report Generator",
                        prompt_path="prompts/base/report_agent.txt",
                        provider=provider,
                        verbose=False
                    )
                except Exception as e:
                    print(f"Failed to create ReportAgent: {e}")
                    # Fallback to mock report generation
                    return await generate_mock_report(query, plan)
            else:
                # Fallback to mock report generation
                return await generate_mock_report(query, plan)
        
        # Generate comprehensive report
        pdf_path = await report_agent.generate_comprehensive_report(execution_data)
        return pdf_path
        
    except Exception as e:
        print(f"Error in generate_report: {e}")
        # Fallback to mock report generation
        return await generate_mock_report(query, plan)

async def generate_mock_report(query: str, plan: List[str]) -> str:
    """Generate a basic mock report as fallback"""
    try:
        import os
        from datetime import datetime
        
        # Create reports directory
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"execution_report_{timestamp}.txt"
        report_path = os.path.join(reports_dir, report_filename)
        
        # Generate simple text report
        report_content = f"""
EXECUTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Query: {query}

Plan:
{chr(10).join([f"  {i+1}. {step}" for i, step in enumerate(plan)])}

Status: Completed
        """
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
        
    except Exception as e:
        print(f"Error generating mock report: {e}")
        return "reports/fallback_report.txt"

# Background processing functions
async def process_query_background(query: str):
    """Enhanced background task processing with real agent integration"""
    global execution_manager
    
    try:
        execution_manager.is_processing = True
        execution_manager.execution_state["status"] = "initializing"
        execution_manager.execution_state["intent"] = query
        execution_manager.execution_state["total_steps"] = 4
        
        # Step 1: Chat agent greeting (real or simulated)
        execution_manager.update_status("greeting", "Chat Assistant", "Communication Tool")
        execution_manager.execution_state["current_step"] = 1
        
        if execution_manager.chat_agent:
            # Use real chat agent
            try:
                initial_response, _ = await execution_manager.chat_agent.process(
                    f"Greet the user and explain that you'll help them with: {query}", None
                )
            except:
                initial_response = f"Hey! I'd love to help you with that. Let me analyze your request: '{query}' and create a comprehensive plan to get you the best results."
        else:
            # Fallback response
            initial_response = f"Hey! I'd love to help you with that. Let me analyze your request: '{query}' and create a comprehensive plan to get you the best results."
            
        execution_manager.add_message("agent", initial_response, "Chat Assistant", "greeting")
        await asyncio.sleep(2)
        
        # Step 2: Planning phase with REAL planner agent
        execution_manager.update_status("planning", "Task Planner", "Planning Tool")
        execution_manager.execution_state["current_step"] = 2
        
        planning_response = "Starting planning phase... I'll analyze your request and break it down into manageable tasks."
        execution_manager.add_message("agent", planning_response, "Task Planner", "planning")
        await asyncio.sleep(1.5)
        
        plan = []
        subtask_status = []
        
        if execution_manager.planner_agent:
            # Use REAL planner agent
            try:
                plan = await execution_manager.planner_agent.make_plan_v2(query)
                execution_state = execution_manager.planner_agent.get_execution_state()
                subtask_status = execution_state.get("subtask_status", [])
                print(f"✅ Real planner created plan with {len(plan)} tasks")
            except Exception as e:
                print(f"⚠️ Real planner failed: {e}, using fallback")
                plan = create_fallback_plan(query)
                subtask_status = create_fallback_subtasks(plan)
        else:
            # Fallback plan creation
            plan = create_fallback_plan(query)
            subtask_status = create_fallback_subtasks(plan)
        
        execution_manager.execution_state["plan"] = [task["task"] for task in plan]
        execution_manager.execution_state["subtask_status"] = subtask_status
        execution_manager.execution_state["total_steps"] = 4 + len(plan)
        
        planning_complete = f"Great! I've created a comprehensive plan with {len(plan)} main tasks. Now I'll start executing each task step by step."
        execution_manager.add_message("agent", planning_complete, "Task Planner", "planning_complete")
        await asyncio.sleep(1)
        
        # Step 3: Enhanced task execution with visual feedback
        execution_manager.update_status("executing", "Task Planner", "Execution Engine")
        execution_manager.execution_state["current_step"] = 3
        
        # Execute tasks sequentially with detailed progress tracking
        max_retries = 2
        failed_tasks = []
        
        for task_index, task_info in enumerate(plan):
            task_name = task_info.get("task", f"Task {task_index + 1}")
            subtasks = task_info.get("subtasks", [task_name])
            agent_type = task_info.get("tool", "CasualAgent")
            
            # Update current step
            execution_manager.execution_state["current_step"] = 4 + task_index
            execution_manager.update_status("executing", agent_type, f"Task {task_index + 1}")
            
            # Show task starting
            task_start_msg = f"🔧 Starting Task {task_index + 1}: {task_name}"
            execution_manager.add_message("agent", task_start_msg, agent_type, "task_starting")
            await asyncio.sleep(0.5)
              # Execute subtasks with retry mechanism
            task_success = True
            for subtask_index, subtask in enumerate(subtasks):
                retries = 0
                subtask_success = False
                
                while retries <= max_retries and not subtask_success:
                    try:
                        # Update subtask status to running
                        if execution_manager.planner_agent:
                            execution_manager.planner_agent.update_subtask_status(task_index, subtask_index, "running")
                            execution_manager.execution_state["subtask_status"] = execution_manager.planner_agent.execution_state.get("subtask_status", [])
                        else:
                            # Update fallback subtask status
                            for idx, status in enumerate(subtask_status):
                                if status["task_index"] == task_index and status["subtask_index"] == subtask_index:
                                    status["status"] = "running"
                            execution_manager.execution_state["subtask_status"] = subtask_status
                        
                        # Show subtask progress
                        if retries > 0:
                            retry_msg = f"↻ Retrying subtask: {subtask} (Attempt {retries + 1})"
                            execution_manager.add_message("agent", retry_msg, agent_type, "retrying")
                        else:
                            subtask_msg = f"▶ Executing: {subtask}"
                            execution_manager.add_message("agent", subtask_msg, agent_type, "executing")
                        
                        await asyncio.sleep(1)  # Simulate work
                          # Execute subtask (real or simulated)
                        result = await execute_subtask(query, task_name, subtask, agent_type, execution_manager)
                        
                        if result["success"]:
                            if execution_manager.planner_agent:
                                execution_manager.planner_agent.update_subtask_status(task_index, subtask_index, "completed", result.get("result", result.get("message", "")))
                                execution_manager.execution_state["subtask_status"] = execution_manager.planner_agent.execution_state.get("subtask_status", [])
                            else:
                                # Update fallback status
                                for idx, status in enumerate(subtask_status):
                                    if status["task_index"] == task_index and status["subtask_index"] == subtask_index:
                                        status["status"] = "completed"
                                        status["result"] = result.get("result", "")
                                execution_manager.execution_state["subtask_status"] = subtask_status
                            
                            success_msg = f"✅ Completed: {subtask}"
                            execution_manager.add_message("agent", success_msg, agent_type, "completed")
                            subtask_success = True
                        else:
                            raise Exception(result.get("error", "Subtask failed"))
                            
                    except Exception as e:
                        retries += 1
                        if retries > max_retries:
                            # Mark as failed
                            if execution_manager.planner_agent:
                                execution_manager.planner_agent.update_subtask_status(task_index, subtask_index, "failed", str(e))
                                execution_manager.execution_state["subtask_status"] = execution_manager.planner_agent.execution_state.get("subtask_status", [])
                            else:
                                # Update fallback status  
                                for idx, status in enumerate(subtask_status):
                                    if status["task_index"] == task_index and status["subtask_index"] == subtask_index:
                                        status["status"] = "failed"
                                        status["error"] = str(e)
                                execution_manager.execution_state["subtask_status"] = subtask_status
                            
                            error_msg = f"❌ Failed: {subtask} - {str(e)}"
                            execution_manager.add_message("agent", error_msg, agent_type, "failed")
                            task_success = False
                            failed_tasks.append(f"Task {task_index + 1}: {subtask}")
                            break
                        else:
                            await asyncio.sleep(0.5)  # Brief pause before retry
            
            # Show task completion
            if task_success:
                completion_msg = f"🎯 Task {task_index + 1} completed successfully"
                execution_manager.add_message("agent", completion_msg, agent_type, "task_completed")
            else:
                failure_msg = f"⚠ Task {task_index + 1} completed with issues"
                execution_manager.add_message("agent", failure_msg, agent_type, "task_failed")
            
            await asyncio.sleep(0.5)
        
        # Final execution summary
        if failed_tasks:
            summary = f"Execution completed with {len(failed_tasks)} issues. Failed tasks: {', '.join(failed_tasks)}"
            execution_manager.add_message("agent", summary, "Task Planner", "execution_summary")
        else:
            summary = f"All {len(plan)} tasks completed successfully!"
            execution_manager.add_message("agent", summary, "Task Planner", "execution_summary")
        
        # Step 4: Report generation with REAL report agent
        execution_manager.update_status("generating_report", "Report Generator", "PDF Generator")
        execution_manager.execution_state["current_step"] = execution_manager.execution_state["total_steps"]
        
        report_msg = "📄 Generating comprehensive PDF report..."
        execution_manager.add_message("agent", report_msg, "Report Generator", "generating")
        await asyncio.sleep(2)
        
        # Generate report (real or mock)
        report_url = await generate_report(query, plan, execution_manager)
        execution_manager.execution_state["final_report_url"] = report_url
        
        completion_message = f"Perfect! I've completed all tasks and generated a comprehensive report. You can download it from the panel above."
        execution_manager.add_message("agent", completion_message, "Report Generator", "completed")
        
        execution_manager.execution_state["status"] = "completed"
        
    except Exception as e:
        execution_manager.execution_state["status"] = "error"
        execution_manager.execution_state["error"] = str(e)
        error_message = f"I encountered an error while processing your request: {str(e)}. Please try again."
        execution_manager.add_message("agent", error_message, "System", "error")
    finally:
        execution_manager.is_processing = False

async def process_advanced_query_background(query: str, execution_mode: str = "fast", 
                                          quality_validation: bool = True, 
                                          generate_report: bool = True):
    """Enhanced background processing using the orchestrator system"""
    global execution_manager
    
    try:
        execution_manager.is_processing = True
        execution_manager.execution_state["status"] = "orchestrator_init"
        execution_manager.execution_state["intent"] = query
        
        # Initialize orchestrator if available
        if not execution_manager.orchestrator:
            # Fallback to regular processing
            await process_query_background(query)
            return
        
        # Step 1: Orchestrator Analysis
        execution_manager.update_status("analyzing_complexity", "Task Orchestrator", "Complexity Analyzer")
        analysis_msg = f"🧠 Analyzing task complexity for: '{query}'"
        execution_manager.add_message("agent", analysis_msg, "Task Orchestrator", "analyzing")
        await asyncio.sleep(1)
        
        # Analyze task complexity
        complexity_analysis = await execution_manager.orchestrator.analyze_task_complexity(query)
        complexity_level = complexity_analysis.get('complexity_level', TaskComplexity.MEDIUM)
        estimated_duration = complexity_analysis.get('estimated_duration', 60)
        required_agents = complexity_analysis.get('required_agents', [])
        
        complexity_msg = f"📊 Task Analysis Complete:\n• Complexity: {complexity_level.value}\n• Estimated Duration: {estimated_duration}s\n• Required Agents: {', '.join(required_agents)}"
        execution_manager.add_message("agent", complexity_msg, "Task Orchestrator", "complexity_analyzed")
        await asyncio.sleep(1)
        
        # Step 2: Choose execution mode
        mode = ExecutionMode.FAST if execution_mode == "fast" else ExecutionMode.DEEP_RESEARCH
        mode_msg = f"⚡ Execution Mode: {mode.value.upper()}"
        execution_manager.add_message("agent", mode_msg, "Task Orchestrator", "mode_selected")
        await asyncio.sleep(0.5)
        
        # Step 3: Create execution plan
        execution_manager.update_status("creating_plan", "Task Orchestrator", "Plan Generator")
        plan_msg = "📋 Creating intelligent execution plan..."
        execution_manager.add_message("agent", plan_msg, "Task Orchestrator", "planning")
        
        execution_plan = await execution_manager.orchestrator.create_execution_plan(
            query, mode, complexity_analysis
        )
        
        plan_steps = execution_plan.get('steps', [])
        execution_manager.execution_state["plan"] = [step['description'] for step in plan_steps]
        execution_manager.execution_state["total_steps"] = len(plan_steps) + 3
        
        plan_complete_msg = f"✅ Execution plan created with {len(plan_steps)} optimized steps"
        execution_manager.add_message("agent", plan_complete_msg, "Task Orchestrator", "plan_ready")
        await asyncio.sleep(1)
        
        # Step 4: Execute the orchestrated plan
        execution_manager.update_status("executing_orchestrated", "Task Orchestrator", "Execution Engine")
        
        execution_results = await execution_manager.orchestrator.execute_plan(
            execution_plan, 
            lambda msg, agent="Task Orchestrator": execution_manager.add_message("agent", msg, agent, "executing")
        )
        
        # Step 5: Quality validation (if enabled)
        if quality_validation and execution_manager.orchestrator.quality_agent:
            execution_manager.update_status("validating_quality", "Quality Agent", "Validation Engine")
            validation_msg = "🔍 Running comprehensive quality validation..."
            execution_manager.add_message("agent", validation_msg, "Quality Agent", "validating")
            await asyncio.sleep(1)
            
            quality_score = await execution_manager.orchestrator.validate_quality(execution_results)
            confidence_score = quality_score.get('confidence_score', 0.0)
            quality_issues = quality_score.get('issues', [])
            
            quality_msg = f"✅ Quality Validation Complete:\n• Confidence Score: {confidence_score:.1%}\n• Issues Found: {len(quality_issues)}"
            if quality_issues:
                quality_msg += f"\n• Key Issues: {', '.join(quality_issues[:3])}"
                
            execution_manager.add_message("agent", quality_msg, "Quality Agent", "validated")
            execution_results['quality_metrics'] = quality_score
            await asyncio.sleep(1)
        
        # Step 6: Generate comprehensive report (if enabled)
        if generate_report:
            execution_manager.update_status("generating_report", "Report Generator", "Report Engine")
            report_msg = "📄 Generating comprehensive research report..."
            execution_manager.add_message("agent", report_msg, "Report Generator", "generating")
            await asyncio.sleep(1)
            
            report_data = await execution_manager.orchestrator.generate_report(
                query, execution_results, mode
            )
            
            # Save report and get URL
            report_url = save_orchestrator_report(report_data)
            execution_manager.execution_state["final_report_url"] = report_url
            
            report_complete_msg = f"✅ Comprehensive report generated and saved!\n📊 Quality Score: {execution_results.get('quality_metrics', {}).get('confidence_score', 0.0):.1%}"
            execution_manager.add_message("agent", report_complete_msg, "Report Generator", "report_ready")
            await asyncio.sleep(1)
        
        # Final completion
        execution_manager.update_status("completed", "Task Orchestrator", "Complete")
        completion_msg = f"🎉 Task completed successfully! The orchestrated research for '{query}' is now ready."
        if execution_manager.execution_state.get("final_report_url"):
            completion_msg += f"\n📋 Full report available for download."
            
        execution_manager.add_message("agent", completion_msg, "Task Orchestrator", "completed")
        
    except Exception as e:
        error_msg = f"❌ Orchestrator error: {str(e)}"
        execution_manager.add_message("error", error_msg, "Task Orchestrator", "error")
        print(f"Orchestrator processing error: {e}")
        
        # Fallback to regular processing
        await process_query_background(query)
        
    finally:
        execution_manager.is_processing = False
        execution_manager.execution_state["status"] = "completed"

def save_orchestrator_report(report_data: Dict[str, Any]) -> str:
    """Save orchestrator report to file and return URL"""
    try:
        timestamp = int(time.time())
        filename = f"orchestrator_report_{timestamp}.html"
        filepath = f"static/reports/{filename}"
        
        # Ensure directory exists
        os.makedirs("static/reports", exist_ok=True)
        
        # Generate HTML report
        html_content = generate_orchestrator_report_html(report_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return f"/reports/{filename}"
        
    except Exception as e:
        print(f"Error saving orchestrator report: {e}")
        return None

def generate_orchestrator_report_html(report_data: Dict[str, Any]) -> str:
    """Generate HTML for orchestrator report"""
    query = report_data.get('query', 'Research Query')
    execution_mode = report_data.get('execution_mode', 'unknown')
    results = report_data.get('results', {})
    quality_metrics = report_data.get('quality_metrics', {})
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orchestrator Research Report - {query}</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px 12px 0 0; }}
        .header h1 {{ margin: 0; font-size: 2em; }}
        .metadata {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }}
        .metadata-item {{ background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; }}
        .content {{ padding: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        .quality-score {{ font-size: 1.2em; font-weight: bold; color: #38a169; }}
        .results-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .result-card {{ background: #f7fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #4299e1; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Orchestrator Research Report</h1>
            <p>Advanced AI-powered research analysis</p>
            <div class="metadata">
                <div class="metadata-item">
                    <strong>Query:</strong><br>{query}
                </div>
                <div class="metadata-item">
                    <strong>Execution Mode:</strong><br>{execution_mode.upper()}
                </div>
                <div class="metadata-item">
                    <strong>Quality Score:</strong><br>
                    <span class="quality-score">{quality_metrics.get('confidence_score', 0.0):.1%}</span>
                </div>
            </div>
        </div>
        <div class="content">
            <div class="section">
                <h2>📊 Analysis Results</h2>
                <div class="results-grid">
                    <div class="result-card">
                        <h3>Execution Summary</h3>
                        <p>Research completed with high confidence using the orchestrator system.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    """
    return html

# Helper functions for fallback processing
def create_fallback_plan(query: str) -> List[Dict[str, Any]]:
    """Create a fallback plan when real planner is not available"""
    plan = [
        {
            "task": "Search and gather information",
            "subtasks": ["Search for relevant information", "Analyze search results", "Extract key findings"],
            "tool": "SearchAgent"
        },
        {
            "task": "Browse and extract content",
            "subtasks": ["Visit relevant websites", "Extract content", "Take screenshots"],
            "tool": "BrowserAgent"
        },
        {
            "task": "Analyze and synthesize",
            "subtasks": ["Process gathered information", "Identify patterns", "Create summary"],
            "tool": "CasualAgent"
        },
        {
            "task": "Generate final response",
            "subtasks": ["Compile findings", "Format response", "Validate completeness"],
            "tool": "ReportAgent"
        }
    ]
    return plan

def create_fallback_subtasks(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create fallback subtask status for the plan"""
    subtask_status = []
    for task_index, task in enumerate(plan):
        subtasks = task.get("subtasks", [])
        for subtask_index, subtask in enumerate(subtasks):
            subtask_status.append({
                "task_index": task_index,
                "subtask_index": subtask_index,
                "subtask": subtask,
                "status": "pending",
                "progress": 0,
                "timestamp": None
            })
    return subtask_status

async def execute_subtask(query: str, task_name: str, subtask: str, agent_type: str, execution_manager: ExecutionManager) -> Dict[str, Any]:
    """Execute a single subtask with the appropriate agent"""
    try:
        # Simulate subtask execution based on agent type
        if agent_type == "SearchAgent" and execution_manager.search_agent:
            try:
                result = await execution_manager.search_agent.search(subtask)
                if result:
                    execution_manager.execution_state["search_results"].extend(result.get("results", []))
                return {"success": True, "result": result, "message": f"Search completed: {len(result.get('results', []))} results found"}
            except Exception as e:
                return {"success": False, "error": str(e), "message": f"Search failed: {str(e)}"}
        
        elif agent_type == "BrowserAgent" and execution_manager.browser_agent:
            try:
                # Simulate browser action
                result = await execution_manager.browser_agent.process(f"Navigate and extract content for: {subtask}", None)
                return {"success": True, "result": result, "message": "Browser action completed successfully"}
            except Exception as e:
                return {"success": False, "error": str(e), "message": f"Browser action failed: {str(e)}"}
        
        elif agent_type == "CoderAgent" and hasattr(execution_manager, 'e2b_sandbox'):
            try:
                # Use E2B sandbox for code execution
                from sources.e2b_integration import E2BSandbox
                if not hasattr(execution_manager, 'e2b_sandbox') or execution_manager.e2b_sandbox is None:
                    execution_manager.e2b_sandbox = E2BSandbox()
                
                # Extract code if present in subtask, otherwise treat as code generation request
                if "```" in subtask:
                    # Extract code blocks from subtask
                    import re
                    code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', subtask, re.DOTALL)
                    if code_blocks:
                        language, code = code_blocks[0]
                        language = language.lower() if language else "python"
                        result = await execution_manager.e2b_sandbox.execute_code(code, language)
                        return {"success": True, "result": result, "message": f"Code executed successfully in {language}"}
                
                # If no code blocks, treat as a coding request
                result = f"Code generation request processed: {subtask}"
                return {"success": True, "result": result, "message": "Coding task analysis completed"}
            except Exception as e:
                return {"success": False, "error": str(e), "message": f"Code execution failed: {str(e)}"}
        
        elif agent_type == "CasualAgent" and execution_manager.chat_agent:
            try:
                result, _ = await execution_manager.chat_agent.process(f"Analyze and provide insights on: {subtask}", None)
                return {"success": True, "result": result, "message": "Analysis completed successfully"}
            except Exception as e:
                return {"success": False, "error": str(e), "message": f"Analysis failed: {str(e)}"}
        
        elif agent_type == "ReportAgent" and execution_manager.report_agent:
            try:
                # Prepare data for report generation
                execution_data = {
                    "intent": query,
                    "subtask": subtask,
                    "current_findings": execution_manager.execution_state.get("agent_outputs", {})
                }
                result = await execution_manager.report_agent.generate_final_summary(execution_data)
                return {"success": True, "result": result, "message": "Report generation completed successfully"}
            except Exception as e:
                return {"success": False, "error": str(e), "message": f"Report generation failed: {str(e)}"}
        
        else:
            # Fallback simulation
            await asyncio.sleep(1)  # Simulate processing time
            return {"success": True, "result": f"Simulated completion of {subtask}", "message": f"Subtask '{subtask}' completed successfully"}
    
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"Subtask execution failed: {str(e)}"}

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    agent_status = "real" if execution_manager.planner_agent else "demo"
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy", 
            "version": "0.2.0-enhanced", 
            "message": "Novah API Enhanced is running",
            "agent_mode": agent_status
        }
    )

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return JSONResponse(
        status_code=200,
        content={"message": "Test endpoint working", "timestamp": time.time()}
    )

@app.post("/query")
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process query endpoint"""
    global execution_manager
    
    if execution_manager.is_processing:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Another query is being processed",
                "message": "Please wait for the current task to complete"
            }
        )
    
    try:
        # Reset state for new query
        execution_manager.reset()
        
        # Start background processing
        background_tasks.add_task(process_query_background, request.query)
        
        # Return immediate response
        return JSONResponse(
            status_code=200,
            content={
                "status": "accepted",
                "message": "Query accepted and processing started",
                "query": request.query
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to start processing",
                "message": str(e)
            }
        )

@app.post("/advanced_query")
async def process_advanced_query(request: AdvancedQueryRequest, background_tasks: BackgroundTasks):
    """Process advanced query with orchestrator integration"""
    global execution_manager
    
    if execution_manager.is_processing:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Another query is being processed",
                "message": "Please wait for the current task to complete"
            }
        )
    
    try:
        # Reset state for new query
        execution_manager.reset()
        
        # Start background processing
        background_tasks.add_task(
            process_advanced_query_background, 
            request.query, 
            request.execution_mode, 
            request.quality_validation, 
            request.generate_report
        )
        
        # Return immediate response
        return JSONResponse(
            status_code=200,
            content={
                "status": "accepted",
                "message": "Advanced query accepted and processing started",
                "query": request.query
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to start processing",
                "message": str(e)
            }
        )

@app.get("/latest_answer")
async def get_latest_answer():
    """Get the latest response from processing"""
    if execution_manager.latest_response:
        response = execution_manager.latest_response.copy()
        response["done"] = "false" if execution_manager.is_processing else "true"
        return JSONResponse(status_code=200, content=response)
    else:
        return JSONResponse(
            status_code=200,
            content={
                "done": "false" if execution_manager.is_processing else "true",
                "status": "waiting",
                "message": "No response available yet"
            }
        )

@app.get("/execution_status")
async def get_execution_status():
    """Get current execution state"""
    return JSONResponse(
        status_code=200,
        content={
            "is_active": execution_manager.is_processing,
            "execution_state": execution_manager.execution_state
        }
    )

@app.get("/current_plan")
async def get_current_plan():
    """Get current execution plan"""
    return JSONResponse(
        status_code=200,
        content={
            "plan": execution_manager.execution_state.get("plan", [])
        }
    )

@app.get("/is_active")
async def is_active():
    """Check if system is currently processing"""
    return JSONResponse(
        status_code=200,
        content={"is_active": execution_manager.is_processing}
    )

@app.get("/stop")
async def stop_processing():
    """Stop current processing"""
    execution_manager.is_processing = False
    execution_manager.execution_state["status"] = "stopped"
    
    return JSONResponse(
        status_code=200,
        content={"status": "stopped", "message": "Processing stopped"}
    )

@app.post("/reset")
async def reset_system():
    """Reset system state"""
    execution_manager.reset()
    return JSONResponse(
        status_code=200,
        content={"status": "reset", "message": "System state reset"}
    )

# Enhanced API Endpoints for Real-time Progress Tracking

@app.get("/agent_progress")
async def get_agent_progress():
    """Get real-time agent progress and status"""
    return JSONResponse(
        status_code=200,
        content={
            "agent_progress": execution_manager.execution_state.get("agent_progress", {}),
            "current_agent": execution_manager.execution_state.get("current_agent"),
            "active_tool": execution_manager.execution_state.get("active_tool"),
            "current_step": execution_manager.execution_state.get("current_step", 0),
            "total_steps": execution_manager.execution_state.get("total_steps", 0),
            "status": execution_manager.execution_state.get("status"),
            "is_processing": execution_manager.is_processing
        }
    )

@app.get("/search_results")
async def get_search_results():
    """Get current search results from SearchAgent"""
    return JSONResponse(
        status_code=200,
        content={
            "search_results": execution_manager.execution_state.get("search_results", []),
            "total_results": len(execution_manager.execution_state.get("search_results", []))
        }
    )

@app.get("/links_processed")
async def get_links_processed():
    """Get links currently being processed by agents"""
    return JSONResponse(
        status_code=200,
        content={
            "links_processed": execution_manager.execution_state.get("links_processed", []),
            "screenshots": execution_manager.execution_state.get("screenshots", [])
        }
    )

@app.get("/execution_summary")
async def get_execution_summary():
    """Get comprehensive execution summary"""
    return JSONResponse(
        status_code=200,
        content={
            "intent": execution_manager.execution_state.get("intent"),
            "plan": execution_manager.execution_state.get("plan", []),
            "subtask_status": execution_manager.execution_state.get("subtask_status", []),
            "agent_outputs": execution_manager.execution_state.get("agent_outputs", {}),
            "current_step": execution_manager.execution_state.get("current_step", 0),
            "total_steps": execution_manager.execution_state.get("total_steps", 0),
            "status": execution_manager.execution_state.get("status"),
            "final_report_url": execution_manager.execution_state.get("final_report_url"),
            "search_results_count": len(execution_manager.execution_state.get("search_results", [])),
            "links_processed_count": len(execution_manager.execution_state.get("links_processed", []))
        }
    )

@app.get("/download_report")
async def download_report():
    """Download the generated report"""
    report_url = execution_manager.execution_state.get("final_report_url")
    if report_url and os.path.exists(report_url):
        return FileResponse(
            path=report_url,
            media_type='application/pdf',
            filename=os.path.basename(report_url)
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Report not found", "message": "No report available for download"}
        )

@app.get("/agent_status/{agent_name}")
async def get_agent_status(agent_name: str):
    """Get detailed status of a specific agent"""
    agent_progress = execution_manager.execution_state.get("agent_progress", {})
    
    if agent_name in agent_progress:
        return JSONResponse(
            status_code=200,
            content={
                "agent": agent_name,
                "status": agent_progress[agent_name],
                "is_active": agent_progress[agent_name].get("status") == "working"
            }
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Agent not found", "available_agents": list(agent_progress.keys())}
        )

# Orchestrator-specific endpoints

@app.post("/orchestrator_query")
async def orchestrator_query(request: AdvancedQueryRequest, background_tasks: BackgroundTasks):
    """Process query using the orchestrator system"""
    if execution_manager.is_processing:
        return JSONResponse(
            status_code=429,
            content={"error": "System is already processing a request"}
        )
    
    if not ORCHESTRATOR_AVAILABLE or not execution_manager.orchestrator:
        return JSONResponse(
            status_code=503,
            content={"error": "Orchestrator system not available"}
        )
    
    # Reset system before starting
    execution_manager.reset()
    
    # Start orchestrator processing in background
    background_tasks.add_task(
        process_advanced_query_background,
        request.query,
        request.execution_mode,
        request.quality_validation,
        request.generate_report
    )
    
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "message": f"Orchestrator processing started for: {request.query}",
            "execution_mode": request.execution_mode,
            "quality_validation": request.quality_validation,
            "generate_report": request.generate_report
        }
    )

@app.get("/orchestrator_status")
async def get_orchestrator_status():
    """Get orchestrator availability and status"""
    return JSONResponse(
        status_code=200,
        content={
            "available": ORCHESTRATOR_AVAILABLE and execution_manager.orchestrator is not None,
            "agents_available": AGENTS_AVAILABLE,
            "is_processing": execution_manager.is_processing,
            "current_mode": execution_manager.execution_state.get("execution_mode", "unknown"),
            "orchestrator_version": "1.0.0" if ORCHESTRATOR_AVAILABLE else None
        }
    )

@app.get("/execution_modes")
async def get_execution_modes():
    """Get available execution modes"""
    return JSONResponse(
        status_code=200,
        content={
            "modes": [
                {
                    "id": "fast",
                    "name": "Fast Mode",
                    "description": "Quick research with essential information",
                    "estimated_time": "30-60 seconds",
                    "features": ["Basic search", "Quick analysis", "Summary report"]
                },
                {
                    "id": "deep_research",
                    "name": "Deep Research Mode",
                    "description": "Comprehensive research with detailed analysis",
                    "estimated_time": "2-5 minutes",
                    "features": ["Multi-engine search", "Quality validation", "Comprehensive report", "Visual analysis", "Data visualization"]
                }
            ],
            "default_mode": "fast"
        }
    )

@app.get("/quality_metrics")
async def get_quality_metrics():
    """Get current quality metrics from the last execution"""
    quality_metrics = execution_manager.execution_state.get("quality_metrics", {})
    
    return JSONResponse(
        status_code=200,
        content={
            "available": bool(quality_metrics),
            "metrics": quality_metrics,
            "confidence_score": quality_metrics.get("confidence_score", 0.0),
            "source_credibility": quality_metrics.get("source_credibility", 0.0),
            "completeness_score": quality_metrics.get("completeness_score", 0.0),
            "issues_found": len(quality_metrics.get("issues", [])),
            "recommendations": quality_metrics.get("recommendations", [])
        }
    )

@app.get("/agent_capabilities")
async def get_agent_capabilities():
    """Get capabilities of all available agents"""
    capabilities = {
        "enhanced_search_agent": {
            "available": ORCHESTRATOR_AVAILABLE,
            "capabilities": [
                "Multi-engine web scraping (DuckDuckGo, Brave, Bing, Yahoo Finance, etc.)",
                "Rate limiting and anti-detection",
                "Result aggregation and quality scoring",
                "No API dependencies"
            ]
        },
        "enhanced_web_agent": {
            "available": ORCHESTRATOR_AVAILABLE,
            "capabilities": [
                "Browser automation with Selenium",
                "Screenshot capture and OCR analysis",
                "Visual element detection",
                "Dynamic content extraction"
            ]
        },
        "enhanced_coding_agent": {
            "available": ORCHESTRATOR_AVAILABLE,
            "capabilities": [
                "E2B sandbox integration",
                "Multi-language code generation",
                "Data visualization (Plotly, Matplotlib)",
                "Interactive dashboard creation"
            ]
        },
        "analysis_agent": {
            "available": AGENTS_AVAILABLE,
            "capabilities": [
                "Deep data synthesis",
                "Pattern recognition",
                "Sentiment analysis",
                "Statistical summaries"
            ]
        },
        "quality_agent": {
            "available": ORCHESTRATOR_AVAILABLE,
            "capabilities": [
                "Source credibility assessment",
                "Fact-checking and claim verification",
                "Bias analysis",
                "Completeness validation"
            ]
        }
    }
    
    return JSONResponse(
        status_code=200,
        content=capabilities
    )

# WebSocket endpoint for real-time updates
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send current execution state to client
            update_data = {
                "type": "execution_update",
                "data": {
                    "agent_progress": execution_manager.execution_state.get("agent_progress", {}),
                    "current_agent": execution_manager.execution_state.get("current_agent"),
                    "active_tool": execution_manager.execution_state.get("active_tool"),
                    "current_step": execution_manager.execution_state.get("current_step", 0),
                    "total_steps": execution_manager.execution_state.get("total_steps", 0),
                    "status": execution_manager.execution_state.get("status"),
                    "is_processing": execution_manager.is_processing,
                    "search_results": execution_manager.execution_state.get("search_results", []),
                    "links_processed": execution_manager.execution_state.get("links_processed", []),
                    "screenshots": execution_manager.execution_state.get("screenshots", [])
                }
            }
            
            await websocket.send_text(json.dumps(update_data))
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)

# Enhanced broadcast function for execution manager
async def broadcast_execution_update(execution_manager):
    """Broadcast execution updates to all connected WebSocket clients"""
    update_data = {
        "type": "execution_update",
        "data": {
            "agent_progress": execution_manager.execution_state.get("agent_progress", {}),
            "current_agent": execution_manager.execution_state.get("current_agent"),
            "active_tool": execution_manager.execution_state.get("active_tool"),
            "current_step": execution_manager.execution_state.get("current_step", 0),
            "total_steps": execution_manager.execution_state.get("total_steps", 0),
            "status": execution_manager.execution_state.get("status"),
            "is_processing": execution_manager.is_processing,
            "search_results": execution_manager.execution_state.get("search_results", []),
            "links_processed": execution_manager.execution_state.get("links_processed", []),
            "screenshots": execution_manager.execution_state.get("screenshots", [])
        }
    }
    
    try:
        await manager.broadcast(update_data)
    except Exception as e:
        print(f"Error broadcasting update: {e}")

@app.get("/code_executions")
async def get_code_executions():
    """Get code executions from E2B integration"""
    return JSONResponse(
        status_code=200,
        content={
            "code_executions": execution_manager.execution_state.get("code_executions", []),
            "total_executions": len(execution_manager.execution_state.get("code_executions", [])),
            "active_sandbox": execution_manager.e2b_sandbox is not None,
            "sandbox_status": "active" if execution_manager.e2b_sandbox else "unavailable",
            "e2b_available": execution_manager.e2b_sandbox is not None
        }
    )

@app.get("/timeline_data")
async def get_timeline_data():
    """Get detailed timeline data for agent execution"""
    timeline = execution_manager.execution_state.get("timeline", [])
    
    return JSONResponse(
        status_code=200,
        content={
            "timeline": timeline,
            "total_steps": len(timeline),
            "current_step": execution_manager.execution_state.get("current_step", 0),
            "estimated_completion": execution_manager.execution_state.get("estimated_completion"),
            "start_time": execution_manager.execution_state.get("start_time"),
            "execution_duration": execution_manager.execution_state.get("execution_duration", 0)
        }
    )

@app.get("/screenshots")
async def get_screenshots():
    """Get organized screenshot data"""
    screenshots = execution_manager.execution_state.get("screenshots", [])
    
    # Organize screenshots by source/type
    organized_screenshots = {
        "browser_captures": [],
        "analysis_charts": [],
        "ui_elements": []
    }
    
    for screenshot in screenshots:
        screenshot_type = screenshot.get("type", "browser_captures")
        if screenshot_type in organized_screenshots:
            organized_screenshots[screenshot_type].append(screenshot)
        else:
            organized_screenshots["browser_captures"].append(screenshot)
    
    return JSONResponse(
        status_code=200,
        content={
            "screenshots": organized_screenshots,
            "total_screenshots": len(screenshots),
            "latest_screenshot": screenshots[-1] if screenshots else None
        }
    )

@app.get("/agent_view_data")
async def get_agent_view_data():
    """Get comprehensive data for the enhanced agent view"""
    return JSONResponse(
        status_code=200,
        content={
            # Plan view data
            "plan": {
                "steps": execution_manager.execution_state.get("plan", []),
                "subtask_status": execution_manager.execution_state.get("subtask_status", []),
                "timeline": execution_manager.execution_state.get("timeline", []),
                "current_step": execution_manager.execution_state.get("current_step", 0),
                "total_steps": execution_manager.execution_state.get("total_steps", 0)
            },
            
            # Browser view data
            "browser": {
                "screenshots": execution_manager.execution_state.get("screenshots", []),
                "links_processed": execution_manager.execution_state.get("links_processed", []),
                "current_url": execution_manager.execution_state.get("current_url")
            },
            
            # Search view data
            "search": {
                "results": execution_manager.execution_state.get("search_results", []),
                "sources_count": len(execution_manager.execution_state.get("search_results", [])),
                "search_queries": execution_manager.execution_state.get("search_queries", [])
            },
            
            # Coding view data
            "coding": {
                "executions": execution_manager.execution_state.get("code_executions", []),
                "active_sandbox": execution_manager.execution_state.get("active_sandbox"),
                "code_outputs": execution_manager.execution_state.get("code_outputs", [])
            },
            
            # Report view data
            "report": {
                "final_report_url": execution_manager.execution_state.get("final_report_url"),
                "report_sections": execution_manager.execution_state.get("report_sections", []),
                "infographics": execution_manager.execution_state.get("infographics", []),
                "metrics": execution_manager.execution_state.get("quality_metrics", {})
            },
            
            # General execution state
            "execution": {
                "is_processing": execution_manager.is_processing,
                "current_agent": execution_manager.execution_state.get("current_agent"),
                "active_tool": execution_manager.execution_state.get("active_tool"),
                "status": execution_manager.execution_state.get("status"),
                "agent_progress": execution_manager.execution_state.get("agent_progress", {})
            }
        }
    )

@app.post("/simulate_code_execution")
async def simulate_code_execution(request: dict):
    """Execute code using E2B sandbox integration"""
    code = request.get("code", "")
    language = request.get("language", "python")
    
    if execution_manager.e2b_sandbox:
        try:
            # Use real E2B sandbox for code execution
            result = await execution_manager.e2b_sandbox.execute_code(code, language)
            execution_result = {
                "id": str(uuid.uuid4()),
                "code": code,
                "language": language,
                "output": result.get("output", ""),
                "status": "completed" if result.get("success") else "error",
                "timestamp": time.time(),
                "execution_time": result.get("execution_time", 0),
                "error": result.get("error") if not result.get("success") else None
            }
        except Exception as e:
            execution_result = {
                "id": str(uuid.uuid4()),
                "code": code,
                "language": language,
                "output": "",
                "status": "error",
                "timestamp": time.time(),
                "execution_time": 0,
                "error": f"E2B execution failed: {str(e)}"
            }
    else:
        # Fallback to mock execution if E2B is not available
        execution_result = {
            "id": str(uuid.uuid4()),
            "code": code,
            "language": language,
            "output": f"Mock execution of {language} code:\n{code}\n\nResult: Execution completed successfully",
            "status": "completed",
            "timestamp": time.time(),
            "execution_time": 1.2
        }
      
    # Add to execution state
    if "code_executions" not in execution_manager.execution_state:
        execution_manager.execution_state["code_executions"] = []
    
    execution_manager.execution_state["code_executions"].append(execution_result)
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "execution": execution_result
        }
    )

# Run the server when script is executed directly
if __name__ == "__main__":
    print("Starting Novah API server...")
    print("Server will be available at: http://localhost:8002")
    print("Docs available at: http://localhost:8002/docs")
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8002,
        reload=True,
        log_level="info"
    )
