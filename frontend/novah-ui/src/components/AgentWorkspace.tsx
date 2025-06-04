import { useState } from "react";
import { motion } from "framer-motion";
import { ToolOutput, ExecutionState } from "../types";

interface Props {
  blocks: ToolOutput[] | null;
  executionState?: ExecutionState | null;
}

export default function AgentWorkspace({ blocks, executionState }: Props) {
  const [open, setOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<"blocks" | "outputs">("blocks");

  // Return empty if no content to show
  if (
    !blocks &&
    (!executionState?.agent_outputs ||
      Object.keys(executionState?.agent_outputs || {}).length === 0)
  ) {
    return null;
  }

  return (
    <motion.div
      animate={{ x: open ? 0 : 320 }}
      className="w-80 border-l border-white/20 p-4 overflow-y-auto bg-black/20 backdrop-blur"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab("blocks")}
            className={`text-sm px-3 py-1 rounded-md ${
              activeTab === "blocks" ? "bg-blue-500" : "bg-gray-700"
            }`}
          >
            Tools
          </button>
          <button
            onClick={() => setActiveTab("outputs")}
            className={`text-sm px-3 py-1 rounded-md ${
              activeTab === "outputs" ? "bg-blue-500" : "bg-gray-700"
            }`}
          >
            Agent Outputs
          </button>
        </div>
        <button
          onClick={() => setOpen((o) => !o)}
          className="text-sm underline"
        >
          {open ? "Collapse ‹" : "Expand ›"}
        </button>
      </div>

      {activeTab === "blocks" &&
        blocks &&
        blocks.map((b, i) => (
          <div
            key={i}
            className="mb-4 bg-black/40 rounded-lg p-3 border border-gray-700/50"
          >
            <div className="font-bold text-blue-300">{b.tool_type}</div>
            <pre className="bg-black/30 p-2 rounded text-xs whitespace-pre-wrap my-2">
              {b.block}
            </pre>
            <div className="text-sm mt-1 text-gray-300">{b.feedback}</div>
          </div>
        ))}

      {activeTab === "outputs" && executionState?.agent_outputs && (
        <div className="space-y-4">
          {Object.entries(executionState.agent_outputs).map(
            ([agentName, output], idx) => (
              <div
                key={idx}
                className="bg-black/40 rounded-lg p-3 border border-gray-700/50"
              >
                <div className="font-bold text-green-300">{agentName}</div>
                <pre className="bg-black/30 p-2 rounded text-xs whitespace-pre-wrap my-2">
                  {output}
                </pre>
                <div className="text-xs text-gray-500">
                  Last updated: {new Date().toLocaleTimeString()}
                </div>
              </div>
            )
          )}
          {Object.keys(executionState.agent_outputs).length === 0 && (
            <div className="text-gray-400 text-center p-4">
              No agent outputs available yet
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
