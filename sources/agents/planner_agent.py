import json
import re
from typing import List, Tuple, Dict, Optional
from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.logger import Logger
from sources.memory import Memory

class PlannerAgent(Agent):
    def __init__(self, name: str, prompt_path: str, provider, verbose: bool = False, browser=None):
        super().__init__(name, prompt_path, provider, verbose, browser)
        self.role = "planification"
        self.type = "planner_agent"
        self.memory = Memory(self.load_prompt(prompt_path), recover_last_session=False, memory_compression=False, model_provider=provider.get_model_name())
        self.logger = Logger("planner_agent.log")

    async def execute(self, prompt: str, execution_manager: Optional['ExecutionManager'] = None):
        """Main entry point for the planner agent."""
        return await self.make_plan(prompt)

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Extracts the first valid JSON object from a string."""
        # Find the first occurrence of '{' and the last '}'
        start_index = text.find('{')
        end_index = text.rfind('}')
        
        if start_index == -1 or end_index == -1:
            return None
            
        json_str = text[start_index : end_index + 1]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback for malformed JSON, like missing closing brackets
            self.logger.warning(f"Failed to parse JSON, attempting to fix: {json_str[:200]}...")
            try:
                # A common issue is trailing commas, which we can try to fix
                fixed_json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                return json.loads(fixed_json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Could not parse JSON even after attempting to fix: {e}")
                return None

    def parse_agent_tasks(self, text: str) -> List[Tuple[str, Dict]]:
        """Parses agent tasks from the LLM text, now with robust JSON extraction."""
        json_data = self._extract_json_from_text(text)
        if not json_data or 'plan' not in json_data or not isinstance(json_data['plan'], list):
            self.logger.error("Failed to find a valid 'plan' list in the LLM response.")
            return []

        tasks = []
        for i, task_data in enumerate(json_data['plan']):
            if not isinstance(task_data, dict) or 'agent' not in task_data or 'task' not in task_data:
                self.logger.warning(f"Skipping malformed task item: {task_data}")
                continue
            
            # The "task name" is just the task description itself
            task_name = task_data['task']
            tasks.append((task_name, task_data))
        
        return tasks

    def show_plan(self, agents_tasks: List[Tuple[str, Dict]]):
        """Displays the plan made by the agent."""
        if not agents_tasks:
            pretty_print("Failed to generate a valid plan. The LLM response might be malformed.", "failure")
            return
        
        pretty_print("\n▂▘ P L A N ▝▂", color="status")
        for task_name, task_details in agents_tasks:
            pretty_print(f"Agent: {task_details['agent']} ➞ Task: {task_name}", color="info")
        pretty_print("▔▗ E N D ▖▔", color="status")

    async def make_plan(self, prompt: str) -> List[Tuple[str, Dict]]:
        """Asks the LLM to make a plan and robustly parses it."""
        self.memory.clear() # Start with a fresh memory for each plan
        self.memory.push('user', prompt)
        
        for attempt in range(2): # Allow one retry
            animate_thinking("Thinking...", color="status")
            answer, reasoning = await self.llm_request()
            
            agents_tasks = self.parse_agent_tasks(answer)
            
            if agents_tasks:
                self.show_plan(agents_tasks)
                self.logger.info(f"Plan made:\n{answer}")
                return agents_tasks
            else:
                pretty_print(f"Failed to parse plan on attempt {attempt + 1}. Retrying...", "warning")
                self.memory.push('assistant', answer) # Add failed answer to context
                self.memory.push('user', "That was not a valid JSON plan. Please provide the plan again, ensuring it is a valid JSON array inside a 'plan' key, like ```json{\"plan\":[...]} ```")

        pretty_print("Failed to make a valid plan after retries.", "failure")
        return []
