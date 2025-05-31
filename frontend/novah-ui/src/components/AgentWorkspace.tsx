import { useState } from 'react';
import { motion } from 'framer-motion';
import { ToolOutput } from '../types';

interface Props {
  blocks: ToolOutput[] | null;
}

export default function AgentWorkspace({ blocks }: Props) {
  const [open, setOpen] = useState(true);
  if (!blocks) return null;
  return (
    <motion.div
      animate={{ x: open ? 0 : 320 }}
      className="w-80 border-l border-white/20 p-4 overflow-y-auto bg-black/20 backdrop-blur"
    >
      <button onClick={() => setOpen(o => !o)} className="text-sm underline mb-2">
        {open ? 'Collapse ‹' : 'Expand ›'}
      </button>
      {blocks.map((b, i) => (
        <div key={i} className="mb-4">
          <div className="font-bold">{b.tool_type}</div>
          <pre className="bg-black/30 p-2 rounded text-xs whitespace-pre-wrap">{b.block}</pre>
          <div className="text-sm mt-1">{b.feedback}</div>
        </div>
      ))}
    </motion.div>
  );
}
