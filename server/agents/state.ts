export type AgentStatus = 'idle' | 'running' | 'completed' | 'error';

export interface AgentState {
  currentAgent: string | null;
  activeTool: string | null;
  status: AgentStatus;
  plan: string[];
  subtaskStatus: any[];
  finalReportUrl: string | null;
}

const state: AgentState = {
  currentAgent: null,
  activeTool: null,
  status: 'idle',
  plan: [],
  subtaskStatus: [],
  finalReportUrl: null,
};

export function getState(): AgentState {
  return state;
}

export function updateState(partial: Partial<AgentState>) {
  Object.assign(state, partial);
}

export function resetState() {
  state.currentAgent = null;
  state.activeTool = null;
  state.status = 'idle';
  state.plan = [];
  state.subtaskStatus = [];
  state.finalReportUrl = null;
}

export default { getState, updateState, resetState };
