import asyncio
from typing import Dict, List, Any

from sources.router import AgentRouter
from sources.agents import PlannerAgent, ReportAgent, Agent


class AgentPipeline:
    def __init__(self, router: AgentRouter, agents: List[Agent], report_agent: ReportAgent):
        self.router = router
        self.agents = {a.__class__.__name__: a for a in agents}
        self.report_agent = report_agent
        self.plan_cache: Dict[str, Any] = {}
        self.current_plan: List[Dict[str, Any]] = []
        self.subtask_status: List[Dict[str, str]] = []
        self.final_report_url: str | None = None
        self.last_answer: str = ""
        self.blocks: List[Any] = []

    async def summarize_intent(self, prompt: str) -> str:
        return self.router.summarize_intent(prompt)

    async def make_plan(self, prompt: str) -> List[Dict[str, Any]]:
        if prompt in self.plan_cache:
            self.current_plan = self.plan_cache[prompt]
            return self.current_plan
        planner: PlannerAgent = self.agents.get('PlannerAgent')  # type: ignore
        raw_plan = await planner.make_plan(prompt)
        plan = []
        for task_name, task in raw_plan:
            plan.append({
                "task": task_name,
                "tool": task['agent'],
                "subtasks": [task['task']]
            })
        self.plan_cache[prompt] = plan
        self.current_plan = plan
        self.subtask_status = [
            {"task": p['task'], "subtask": st, "status": "pending"}
            for p in plan for st in p['subtasks']
        ]
        return plan

    async def execute_plan(self, plan: List[Dict[str, Any]], max_retries: int = 2):
        idx = 0
        for p in plan:
            agent_name = p['tool'] + 'Agent'
            agent = self.agents.get(agent_name)
            if not agent:
                continue
            for st in p['subtasks']:
                retries = 0
                success = False
                while retries <= max_retries and not success:
                    answer, _ = await agent.process(st, None)
                    success = agent.success
                    self.blocks.extend(agent.get_blocks_result())
                    self.last_answer = answer
                    status = 'done' if success else 'failed'
                    self.subtask_status[idx]['status'] = status
                    if not success:
                        retries += 1
                idx += 1
        report_prompt = "\n".join([s['subtask'] + ": " + s['status'] for s in self.subtask_status])
        await self.report_agent.process(report_prompt, None)
        self.final_report_url = 'reports/final_report.pdf'

    async def run(self, prompt: str):
        intent = await self.summarize_intent(prompt)
        plan = await self.make_plan(intent)
        await self.execute_plan(plan)

    def latest_data(self):
        return {
            "answer": self.last_answer,
            "plan": self.current_plan,
            "subtask_status": self.subtask_status,
            "final_report_url": self.final_report_url,
            "blocks": [b.jsonify() if hasattr(b, 'jsonify') else b for b in self.blocks],
        }
