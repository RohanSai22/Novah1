#!/usr/bin/env python3
"""
Novah API - Final Corrected
Stateful, orchestrator-driven backend for advanced AI research tasks
"""
import uvicorn
import asyncio
import time
import os
import sys
import configparser
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure the sources directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources.llm_provider import Provider
from sources.agents import Agent, CoderAgent, CasualAgent, FileAgent, PlannerAgent, BrowserAgent, McpAgent
# Import the new Enhanced agents
from sources.agents.enhanced_search_agent import EnhancedSearchAgent
from sources.agents.enhanced_web_agent import EnhancedWebAgent
from sources.agents.enhanced_coding_agent import EnhancedCodingAgent
from sources.agents.enhanced_analysis_agent import EnhancedAnalysisAgent
from sources.agents.enhanced_report_agent import EnhancedReportAgent
from sources.agents.quality_agent import QualityAgent

from sources.orchestrator.task_orchestrator import TaskOrchestrator, ExecutionMode
from sources.browser import create_driver, Browser
from sources.utility import pretty_print

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class OrchestratorQueryRequest(BaseModel):
    query: str
    execution_mode: str = "fast"

app = FastAPI(title="Novah API", version="1.0.0")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static File Serving ---
os.makedirs("reports", exist_ok=True)
os.makedirs(".screenshots", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/screenshots", StaticFiles(directory=".screenshots"), name="screenshots")

# --- Singleton Execution Manager ---
class ExecutionManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ExecutionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return
        self.initialized = True
        self.is_processing = False
        self.orchestrator = None
        self.agents = {}
        self.browser_driver = None
        self.reset()
        self.initialize_agents()

    def initialize_agents(self):
        pretty_print("Initializing AI agents and services...", "status")
        try:
            config_path = 'config.ini'
            if not os.path.exists(config_path):
                # Fallback to config copy if original is missing
                if os.path.exists('config copy.ini'):
                    import shutil
                    shutil.copy('config copy.ini', config_path)
                else:
                    raise FileNotFoundError("config.ini not found. Please create it from 'config copy.ini' or the example.")
            
            config = configparser.ConfigParser()
            config.read(config_path)
            
            provider = Provider(
                provider_name=config.get("MAIN", "provider_name"),
                model=config.get("MAIN", "provider_model"),
                server_address=config.get("MAIN", "provider_server_address"),
                is_local=config.getboolean('MAIN', 'is_local')
            )
            
            self.browser_driver = create_driver(headless=config.getboolean("BROWSER", "headless_browser"))
            browser_instance = Browser(self.browser_driver)

            agent_definitions = {
                "EnhancedSearchAgent": {"prompt_path": "prompts/base/search_agent.txt"},
                "EnhancedWebAgent": {"prompt_path": "prompts/base/browser_agent.txt", "browser": browser_instance},
                "EnhancedCodingAgent": {"prompt_path": "prompts/base/coder_agent.txt"},
                "EnhancedAnalysisAgent": {"prompt_path": "prompts/base/analysis_agent.txt"},
                "EnhancedReportAgent": {"prompt_path": "prompts/base/report_agent.txt"},
                "QualityAgent": {"prompt_path": "prompts/base/quality_agent.txt"},
                "PlannerAgent": {"prompt_path": "prompts/base/planner_agent.txt", "browser": browser_instance},
                "CasualAgent": {"prompt_path": "prompts/base/casual_agent.txt"},
                "FileAgent": {"prompt_path": "prompts/base/file_agent.txt"}
            }
            
            # This makes all agent classes available to globals()
            from sources import agents

            for name, params in agent_definitions.items():
                # Use getattr to dynamically get the class from the agents module
                agent_class = getattr(agents, name)
                self.agents[name] = agent_class(
                    name=name,
                    provider=provider,
                    **params
                )
            
            self.orchestrator = TaskOrchestrator(self.agents, provider)
            pretty_print("✅ All agents and services initialized successfully.", "success")
            
        except Exception as e:
            pretty_print(f"CRITICAL ERROR during initialization: {e}", "failure")
            import traceback
            traceback.print_exc()
            if self.browser_driver:
                self.browser_driver.quit()
            sys.exit(1)

    def reset(self):
        self.is_processing = False
        self.execution_state = {
            "plan": {"steps": [], "subtask_status": [], "timeline": [], "current_step": 0, "total_steps": 0},
            "browser": {"screenshots": [], "links_processed": [], "current_url": ""},
            "search": {"results": [], "sources_count": 0, "search_queries": []},
            "coding": {"executions": [], "active_sandbox": None, "code_outputs": []},
            "report": {"final_report_url": None, "report_sections": [], "infographics": [], "metrics": None},
            "execution": {"is_processing": False, "current_agent": None, "active_tool": None, "status": "idle", "agent_progress": {}}
        }
    
    def update_state(self, updates: Dict):
        for key, value in updates.items():
            if key in self.execution_state:
                if isinstance(self.execution_state[key], dict) and isinstance(value, dict):
                    self.execution_state[key].update(value)
                elif isinstance(self.execution_state[key], list) and isinstance(value, list):
                    # This prevents duplicate entries in lists like screenshots
                    current_list = self.execution_state[key]
                    for item in value:
                        if isinstance(item, dict) and 'id' in item:
                             if not any(d.get('id') == item['id'] for d in current_list):
                                current_list.append(item)
                        elif item not in current_list:
                             current_list.append(item)
                else:
                    self.execution_state[key] = value

execution_manager = ExecutionManager()

async def run_orchestrated_task(query: str, mode: str):
    execution_manager.is_processing = True
    execution_manager.update_state({"execution": {"status": "initializing", "is_processing": True, "current_agent": "TaskOrchestrator", "intent": query}})
    
    try:
        if not execution_manager.orchestrator:
            raise Exception("Orchestrator not initialized.")
        await execution_manager.orchestrator.execute_plan(query, ExecutionMode(mode), execution_manager)
    except Exception as e:
        print(f"Orchestration failed: {e}")
        import traceback
        traceback.print_exc()
        execution_manager.update_state({"execution": {"status": "error", "error_message": str(e)}})
    finally:
        final_status = "failed" if execution_manager.execution_state['execution'].get('error_message') else "completed"
        execution_manager.is_processing = False
        execution_manager.update_state({"execution": {"is_processing": False, "status": final_status}})

@app.post("/orchestrator_query")
async def orchestrator_query(request: OrchestratorQueryRequest, background_tasks: BackgroundTasks):
    if execution_manager.is_processing:
        raise HTTPException(status_code=429, detail="A task is already in progress.")
    execution_manager.reset()
    background_tasks.add_task(run_orchestrated_task, request.query, request.execution_mode)
    return JSONResponse(status_code=202, content={"message": "Orchestrated task started."})

@app.post("/query")
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    if execution_manager.is_processing:
        raise HTTPException(status_code=429, detail="A task is already in progress.")
    execution_manager.reset()
    # Simple queries now also run through the orchestrator in "fast" mode.
    background_tasks.add_task(run_orchestrated_task, request.query, "fast")
    return JSONResponse(status_code=202, content={"message": "Fast-track task started."})

@app.get("/agent_view_data")
async def get_agent_view_data():
    return JSONResponse(status_code=200, content=execution_manager.execution_state)

@app.on_event("shutdown")
def shutdown_event():
    if execution_manager.browser_driver:
        execution_manager.browser_driver.quit()
        print("Browser driver closed.")

if __name__ == "__main__":
    pretty_print("Starting Novah API server...", "status")
    # Use port 8002 to match the new frontend's expectations
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)
