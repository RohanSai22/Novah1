import { ToolOutput } from '../types';

interface Props {
  blocks: ToolOutput[] | null;
}

export default function AgentWorkspace({ blocks }: Props) {
  if (!blocks) return null;
  return (
    <div className="w-80 border-l border-white/20 p-4 overflow-y-auto">
      {blocks.map((b, i) => (
        <div key={i} className="mb-4">
          <div className="font-bold">{b.tool_type}</div>
          <pre className="bg-black/30 p-2 rounded text-xs whitespace-pre-wrap">{b.block}</pre>
          <div className="text-sm mt-1">{b.feedback}</div>
        </div>
      ))}
    </div>
  );
}
