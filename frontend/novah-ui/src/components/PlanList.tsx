import { motion } from "framer-motion";
import { PlanItem, SubtaskStatus } from "../types";
import GlassCard from "./ui/GlassCard";

interface Props {
  plan: PlanItem[];
  subStatus: SubtaskStatus[];
  reportUrl?: string | null;
}

export default function PlanList({ plan, subStatus, reportUrl }: Props) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return "✅";
      case "running":
        return "🔄";
      case "failed":
        return "❌";
      case "pending":
      default:
        return "⏳";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "text-green-400";
      case "running":
        return "text-blue-400";
      case "failed":
        return "text-red-400";
      case "pending":
      default:
        return "text-gray-400";
    }
  };

  if (plan.length === 0 && subStatus.length === 0) {
    return (
      <div className="p-4">
        <div className="text-center text-gray-400 py-8">
          <div className="text-4xl mb-2">🤖</div>
          <div>No active plan. Send a message to get started!</div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {plan.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <GlassCard title="Execution Plan">
            {plan.map((planItem, planIndex) => (
              <div
                key={`plan-${planIndex}`}
                className="bg-white/5 rounded-lg p-3 mb-3 border border-white/5"
              >
                <div className="font-semibold text-white flex items-center gap-2">
                  <span className="text-blue-300">#{planIndex + 1}</span>
                  {planItem.task}
                  <span className="text-xs bg-gray-600 px-2 py-1 rounded">
                    {planItem.tool}
                  </span>
                </div>
              </div>
            ))}
          </GlassCard>
        </motion.div>
      )}

      {subStatus.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <GlassCard title="Task Progress">
            <div className="space-y-2">
              {subStatus.map((subtask, index) => (
                <motion.div
                  key={`subtask-${subtask.id}-${index}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center gap-3 bg-white/10 rounded-lg p-3"
                >
                  <span className="text-xl">
                    {getStatusIcon(subtask.status)}
                  </span>
                  <div className="flex-1">
                    <div
                      className={`font-medium ${getStatusColor(
                        subtask.status
                      )}`}
                    >
                      {subtask.description}
                    </div>
                    {subtask.agent_assigned && (
                      <div className="text-xs text-gray-400 mt-1">
                        Agent: {subtask.agent_assigned}
                      </div>
                    )}
                    {subtask.output && subtask.status === "completed" && (
                      <div className="text-xs text-gray-300 mt-1 max-w-md truncate">
                        Result: {subtask.output}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-gray-400 capitalize">
                    {subtask.status}
                  </div>
                </motion.div>
              ))}
            </div>
          </GlassCard>
        </motion.div>
      )}

      {reportUrl && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
        >
          <GlassCard title="📄 Final Report">
            <a
              href={`http://localhost:8001/${reportUrl}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition-colors"
            >
              <span>📥</span>
              Download PDF Report
            </a>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}
