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

export interface QualityMetrics {
  confidence_score: number;
  overall_confidence: number;
  source_credibility: number;
  factual_accuracy: number;
  completeness: number;
  completeness_score: number;
  recency: number;
  recency_score?: number;
  bias_analysis: {
    political_bias: number;
    commercial_bias: number;
    cultural_bias: number;
    temporal_bias: number;
  };
  bias_score?: number;
  verification_status: {
    fact_checked: boolean;
    cross_referenced: boolean;
    expert_validated: boolean;
  };
  issues: string[];
  recommendations: string[];
}

export interface ExecutionMode {
  id: string;
  name: string;
  description: string;
  features: string[];
  estimated_time: string;
  quality_level: "basic" | "standard" | "comprehensive";
}

export interface OrchestratorConfig {
  execution_mode: string;
  quality_validation: boolean;
  generate_report: boolean;
  max_agents: number;
  timeout_minutes: number;
}

export interface AgentCapability {
  agent_name: string;
  capabilities: string[];
  supported_tasks: string[];
  estimated_performance: number;
}

// Enhanced types for the new agent view features
export interface BrowserScreenshot {
  id: string;
  url: string;
  timestamp: string;
  action: string;
  step: number;
}

export interface SearchResult {
  id: string;
  query: string;
  source: string;
  title: string;
  url: string;
  snippet: string;
  timestamp: string;
  relevance_score?: number;
}

export interface CodeExecution {
  id: string;
  code: string;
  language: string;
  output: string;
  status: "running" | "completed" | "error";
  timestamp: number;
  execution_time?: number;
  error?: string;
}

export interface Screenshot {
  id: string;
  url: string;
  type: "browser_captures" | "analysis_charts" | "ui_elements";
  caption?: string;
  timestamp: number;
  source_url?: string;
  analysis?: string;
}

export interface TimelineStep {
  id: string;
  timestamp: number;
  agent: string;
  action: string;
  status: "pending" | "in_progress" | "completed" | "error";
  details?: string;
  duration?: number;
}

export interface AgentViewData {
  plan: {
    steps: PlanItem[];
    subtask_status: SubtaskStatus[];
    timeline: TimelineStep[];
    current_step: number;
    total_steps: number;
  };
  browser: {
    screenshots: Screenshot[];
    links_processed: string[];
    current_url?: string;
  };
  search: {
    results: SearchResult[];
    sources_count: number;
    search_queries: string[];
  };
  coding: {
    executions: CodeExecution[];
    active_sandbox?: string;
    code_outputs: any[];
  };
  report: {
    final_report_url?: string;
    report_sections: any[];
    infographics: any[];
    metrics: QualityMetrics;
  };
  execution: {
    is_processing: boolean;
    current_agent?: string;
    active_tool?: string;
    status: string;
    agent_progress: Record<string, any>;
  };
}

// Enhanced execution state interface
export interface EnhancedExecutionState extends ExecutionState {
  code_executions?: CodeExecution[];
  screenshots?: Screenshot[];
  search_results?: SearchResult[];
  timeline?: TimelineStep[];
  search_queries?: string[];
  code_outputs?: any[];
  report_sections?: any[];
  infographics?: any[];
}
