from typing import Tuple, Callable, Optional, List, Dict, Any
from pydantic import BaseModel
from sources.utility import pretty_print

class QueryRequest(BaseModel):
    query: str
    tts_enabled: bool = True

    def __str__(self):
        return f"Query: {self.query}, TTS: {self.tts_enabled}"

    def jsonify(self):
        return {
            "query": self.query,
            "tts_enabled": self.tts_enabled,
        }

class SubtaskStatus(BaseModel):
    id: str
    description: str
    status: str  # pending, running, completed, failed
    agent_assigned: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    output: Optional[str] = None

    def jsonify(self):
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "agent_assigned": self.agent_assigned,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "output": self.output
        }

class ExecutionState(BaseModel):
    intent: Optional[str] = None
    plan: Optional[List[str]] = None
    current_subtask: Optional[str] = None
    subtask_status: Optional[List[SubtaskStatus]] = None
    agent_outputs: Optional[Dict[str, Any]] = None
    final_report_url: Optional[str] = None

    def jsonify(self):
        return {
            "intent": self.intent,
            "plan": self.plan,
            "current_subtask": self.current_subtask,
            "subtask_status": [s.jsonify() if hasattr(s, 'jsonify') else s for s in (self.subtask_status or [])],
            "agent_outputs": self.agent_outputs,
            "final_report_url": self.final_report_url
        }

class QueryResponse(BaseModel):
    done: str
    answer: str
    reasoning: str
    agent_name: str
    success: str
    blocks: dict
    status: str
    uid: str
    execution_state: Optional[ExecutionState] = None

    def __str__(self):
        return f"Done: {self.done}, Answer: {self.answer}, Agent Name: {self.agent_name}, Success: {self.success}, Blocks: {self.blocks}, Status: {self.status}, UID: {self.uid}"

    def jsonify(self):
        return {
            "done": self.done,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "agent_name": self.agent_name,
            "success": self.success,
            "blocks": self.blocks,
            "status": self.status,
            "uid": self.uid,
            "execution_state": self.execution_state.jsonify() if self.execution_state else None
        }

class ExecutionStatusResponse(BaseModel):
    execution_state: Optional[ExecutionState] = None
    uid: Optional[str] = None

    def jsonify(self):
        return {
            "execution_state": self.execution_state.jsonify() if self.execution_state else None,
            "uid": self.uid
        }

class executorResult:
    """
    A class to store the result of a tool execution.
    """
    def __init__(self, block: str, feedback: str, success: bool, tool_type: str):
        """
        Initialize an agent with execution results.

        Args:
            block: The content or code block processed by the agent.
            feedback: Feedback or response information from the execution.
            success: Boolean indicating whether the agent's execution was successful.
            tool_type: The type of tool used by the agent for execution.
        """
        self.block = block
        self.feedback = feedback
        self.success = success
        self.tool_type = tool_type
    
    def __str__(self):
        return f"Tool: {self.tool_type}\nBlock: {self.block}\nFeedback: {self.feedback}\nSuccess: {self.success}"
    
    def jsonify(self):
        return {
            "block": self.block,
            "feedback": self.feedback,
            "success": self.success,
            "tool_type": self.tool_type
        }

    def show(self):
        pretty_print('▂'*64, color="status")
        pretty_print(self.feedback, color="success" if self.success else "failure")
        pretty_print('▂'*64, color="status")