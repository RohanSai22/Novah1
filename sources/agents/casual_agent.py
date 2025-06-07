import asyncio
from typing import Optional, Tuple

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent, ExecutionManager
from sources.memory import Memory

class CasualAgent(Agent):
    def __init__(self, name, prompt_path, provider, verbose=False):
        """
        The casual agent is a special for casual talk to the user without specific tasks.
        """
        super().__init__(name, prompt_path, provider, verbose)
        self.role = "talk"
        self.type = "casual_agent"
        self.memory = Memory(self.load_prompt(prompt_path),
                                recover_last_session=False, # session recovery in handled by the interaction class
                                memory_compression=False,
                                model_provider=provider.get_model_name())
    
    async def process(self, prompt: str, execution_manager: Optional[ExecutionManager] = None, speech_module=None) -> Tuple[str, str]:
        return await self.execute(prompt, execution_manager)
        
    async def execute(self, prompt: str, execution_manager: Optional[ExecutionManager] = None) -> dict:
        self.memory.push('user', prompt)
        animate_thinking("Thinking...", color="status")
        answer, reasoning = await self.llm_request()
        self.last_answer = answer
        self.status_message = "Ready"

        if execution_manager:
            execution_manager.update_state({
                "execution": {
                    "agent_progress": {self.agent_name: {"status": "completed", "output": answer}}
                }
            })

        return {
            "success": True,
            "summary": "Responded to user query.",
            "answer": answer,
            "reasoning": reasoning
        }

if __name__ == "__main__":
    pass