export interface Message {
  type: "user" | "agent" | "error";
  content: string;
  reasoning?: string;
  agentName?: string;
  status?: string;
  uid?: string;
  execution_state?: ExecutionState;
}

export interface ToolOutput {
  tool_type: string;
  block: string;
  feedback: string;
  success: boolean;
}

export interface Thread {
  id: string;
  title?: string;
  messages: Message[];
  blocks: ToolOutput[] | null;
}

export interface PlanItem {
  task: string;
  tool: string;
  subtasks: string[];
  status?: string;
}

export interface SubtaskStatus {
  id: string;
  description: string;
  status: "pending" | "running" | "completed" | "failed";
  agent_assigned?: string;
  start_time?: string;
  end_time?: string;
  output?: string;
}

export interface ExecutionState {
  intent?: string;
  plan?: string[];
  current_task?: number;
  current_subtask?: string;
  subtask_status?: SubtaskStatus[];
  agent_outputs?: Record<string, any>;
  final_report_url?: string;
  status?: string;
  current_agent?: string;
  active_tool?: string;
  current_step?: number;
  total_steps?: number;
}

export interface QueryResponse {
  done: string;
  answer: string;
  reasoning: string;
  agent_name: string;
  success: string;
  blocks: Record<string, ToolOutput>;
  status: string;
  uid: string;
  execution_state?: ExecutionState;
}

export interface ExecutionStatusResponse {
  is_active: boolean;
  execution_state: ExecutionState;
  agent_name: string;
  timestamp: number;
}
