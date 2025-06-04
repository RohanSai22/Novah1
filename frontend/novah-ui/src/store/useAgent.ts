import { create } from 'zustand';
import { ExecutionState, PlanItem, SubtaskStatus, ToolOutput } from '../types';

type AgentStore = {
  isProcessing: boolean;
  executionState: ExecutionState | null;
  plan: PlanItem[];
  subStatus: SubtaskStatus[];
  blocks: ToolOutput[] | null;
  reportUrl: string | null;
  status: string;
  currentAgent: string | null;
  activeTool: string | null;
  setState: (partial: Partial<AgentStore>) => void;
  reset: () => void;
};

const initialState = {
  isProcessing: false,
  executionState: null,
  plan: [] as PlanItem[],
  subStatus: [] as SubtaskStatus[],
  blocks: null as ToolOutput[] | null,
  reportUrl: null as string | null,
  status: 'Agents ready',
  currentAgent: null as string | null,
  activeTool: null as string | null,
};

export const useAgent = create<AgentStore>((set) => ({
  ...initialState,
  setState: (partial) => set(partial),
  reset: () => set(initialState),
}));

export default useAgent;
