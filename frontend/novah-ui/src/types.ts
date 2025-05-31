export interface Message {
  type: 'user' | 'agent' | 'error';
  content: string;
  reasoning?: string;
  agentName?: string;
  status?: string;
  uid?: string;
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
}

export interface SubtaskStatus {
  subtask: string;
  status: string;
}
