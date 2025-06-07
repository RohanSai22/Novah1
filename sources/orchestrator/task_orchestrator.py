"""
Advanced Task Orchestrator - The Central Brain of the Agent System
This orchestrator manages dynamic task planning, agent routing, and intelligent execution
"""

import asyncio
import time
import json
import os
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass, asdict

from sources.utility import pretty_print
from sources.logger import Logger

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class ExecutionMode(Enum):
    FAST = "fast"
    DEEP_RESEARCH = "deep_research"

@dataclass
class Task:
    id: str
    description: str
    agent_name: str
    priority: int
    dependencies: List[str]
    context: Dict[str, Any]

class ExecutionPlan:
    def __init__(self, query: str, mode: ExecutionMode, tasks: List[Task]):
        self.id = f"plan_{int(time.time())}"
        self.query = query
        self.mode = mode
        self.tasks = tasks

class TaskOrchestrator:
    def __init__(self, agents_registry: Dict, llm_provider=None):
        self.agents_registry = agents_registry
        self.llm_provider = llm_provider
        self.logger = Logger("task_orchestrator.log")

    def _get_agent_key(self, agent_name: str) -> str:
        from sources.agents import EnhancedSearchAgent, EnhancedWebAgent, EnhancedCodingAgent, FileAgent, CasualAgent, EnhancedReportAgent, QualityAgent, EnhancedAnalysisAgent, PlannerAgent
        agent_map = {
            "web": EnhancedWebAgent, "search": EnhancedSearchAgent, "coder": EnhancedCodingAgent,
            "file": FileAgent, "casual": CasualAgent, "report": EnhancedReportAgent,
            "quality": QualityAgent, "analysis": EnhancedAnalysisAgent, "planner": PlannerAgent
        }
        normalized_name = agent_name.lower().strip()
        
        for key, agent_class in agent_map.items():
            if key in normalized_name:
                return agent_class.__name__
        
        self.logger.warning(f"Could not find a specific agent for '{agent_name}'. Defaulting to CasualAgent.")
        return CasualAgent.__name__

    async def create_execution_plan(self, query: str, mode: ExecutionMode) -> ExecutionPlan:
        planner = self.agents_registry.get("PlannerAgent")
        if not planner:
            raise Exception("PlannerAgent not found in registry.")

        raw_plan = await planner.make_plan(query)
        tasks = []
        for i, (task_name, task_details) in enumerate(raw_plan):
            agent_key = self._get_agent_key(task_details['agent'])
            tasks.append(Task(
                id=task_details.get('id', str(i + 1)),
                description=task_details.get('task', task_name),
                agent_name=agent_key,
                priority=i,
                dependencies=task_details.get('need', []),
                context={"query": task_details.get('task', task_name)}
            ))
        return ExecutionPlan(query=query, mode=mode, tasks=tasks)
    
    async def execute_plan(self, query: str, mode: ExecutionMode, execution_manager):
        plan = await self.create_execution_plan(query, mode)
        
        plan_steps_for_state = [{"task": t.description, "tool": t.agent_name.replace('Agent',''), "subtasks": [t.context.get('query', t.description)], "status":"pending"} for t in plan.tasks]
        execution_manager.update_state({
            "plan": {"steps": plan_steps_for_state, "total_steps": len(plan.tasks)},
            "execution": {"status": "planning_complete"}
        })

        results = {}
        executed_tasks = set()

        while len(executed_tasks) < len(plan.tasks):
            ready_tasks = [
                task for task in plan.tasks
                if task.id not in executed_tasks and all(dep in executed_tasks for dep in task.dependencies)
            ]
            if not ready_tasks:
                if len(executed_tasks) < len(plan.tasks):
                     raise Exception("Execution plan stuck. Possible dependency cycle or task failure.")
                else: break

            for task in sorted(ready_tasks, key=lambda t: t.priority):
                agent = self.agents_registry.get(task.agent_name)
                
                if not agent:
                    results[task.id] = {"error": f"Agent {task.agent_name} not found."}
                    executed_tasks.add(task.id)
                    continue

                execution_manager.update_state({
                    "execution": { "current_agent": agent.agent_name, "status": f"Executing: {task.description}" }
                })
                
                context_for_agent = task.context.copy()
                if task.dependencies:
                    for dep_id in task.dependencies:
                        if dep_id in results: context_for_agent.update(results.get(dep_id, {}))
                
                prompt_for_agent = json.dumps(context_for_agent)
                result = await agent.execute(prompt_for_agent, execution_manager)

                results[task.id] = result
                executed_tasks.add(task.id)
                
                current_outputs = execution_manager.execution_state['execution'].get('agent_progress', {})
                current_outputs[agent.agent_name] = {"status": "completed", "output": str(result.get('summary', str(result)))[:500]}
                execution_manager.update_state({"execution": {"agent_progress": current_outputs}})

        if mode == ExecutionMode.DEEP_RESEARCH:
            report_agent = self.agents_registry.get("EnhancedReportAgent")
            if report_agent:
                execution_manager.update_state({"execution": {"status": "Generating Report", "current_agent": "EnhancedReportAgent"}})
                report_context = {"intent": query, "plan": [t.description for t in plan.tasks], "agent_outputs": results}
                report_result = await report_agent.execute(json.dumps(report_context), execution_manager)
                if report_result and report_result.get("report_path"):
                    web_path = os.path.join('/', report_result["report_path"]).replace('\\', '/')
                    execution_manager.update_state({"report": {"final_report_url": web_path}})
