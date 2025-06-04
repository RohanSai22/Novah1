import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PlanItem, SubtaskStatus, ExecutionState, ToolOutput } from "../types";

interface Props {
  isProcessing: boolean;
  executionState: ExecutionState | null;
  plan: PlanItem[];
  subStatus: SubtaskStatus[];
  blocks: ToolOutput[] | null;
  reportUrl: string | null;
  currentAgent?: string;
  activeTool?: string;
}

type ViewTab = "plan" | "browser" | "search" | "coding" | "report";

interface AgentStep {
  id: string;
  timestamp: string;
  agent: string;
  action: string;
  status: "running" | "completed" | "failed";
  details?: string;
}

export default function EnhancedAgentView({
  isProcessing,
  executionState,
  plan,
  subStatus,
  blocks,
  reportUrl,
  currentAgent,
  activeTool,
}: Props) {
  const [activeTab, setActiveTab] = useState<ViewTab>("plan");
  const [timeline, setTimeline] = useState<AgentStep[]>([]);
  const [screenshots, setScreenshots] = useState<string[]>([]);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [codeExecutions, setCodeExecutions] = useState<any[]>([]);

  // Auto-switch to active agent's tab
  useEffect(() => {
    if (currentAgent && isProcessing) {
      if (currentAgent.toLowerCase().includes("browser")) {
        setActiveTab("browser");
      } else if (currentAgent.toLowerCase().includes("search")) {
        setActiveTab("search");
      } else if (currentAgent.toLowerCase().includes("code")) {
        setActiveTab("coding");
      } else if (currentAgent.toLowerCase().includes("planner")) {
        setActiveTab("plan");
      }
    }
  }, [currentAgent, isProcessing]);

  // Update timeline based on execution state
  useEffect(() => {
    if (executionState && isProcessing) {
      const newStep: AgentStep = {
        id: `${Date.now()}-${currentAgent}`,
        timestamp: new Date().toLocaleTimeString(),
        agent: currentAgent || "System",
        action: activeTool || "Processing",
        status: "running",
        details: executionState.current_subtask,
      };

      setTimeline((prev) => {
        const existing = prev.find((step) => step.id === newStep.id);
        if (!existing) {
          return [...prev, newStep];
        }
        return prev;
      });
    }
  }, [executionState, currentAgent, activeTool, isProcessing]);

  const tabs = [
    { id: "plan", label: "Plan View", icon: "🎯" },
    { id: "browser", label: "Browser View", icon: "🌐" },
    { id: "search", label: "Search View", icon: "🔍" },
    { id: "coding", label: "Coding View", icon: "💻" },
    { id: "report", label: "Report View", icon: "📊" },
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case "plan":
        return (
          <div className="space-y-4">
            {/* Timeline */}
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 text-white">
                Execution Timeline
              </h3>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {timeline.map((step, index) => (
                  <motion.div
                    key={step.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`flex items-center gap-3 p-3 rounded-lg ${
                      step.status === "running"
                        ? "bg-blue-500/20 border-l-4 border-blue-500"
                        : step.status === "completed"
                        ? "bg-green-500/20 border-l-4 border-green-500"
                        : "bg-red-500/20 border-l-4 border-red-500"
                    }`}
                  >
                    <div className="flex-shrink-0">
                      {step.status === "running" && (
                        <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
                      )}
                      {step.status === "completed" && (
                        <div className="w-3 h-3 bg-green-500 rounded-full" />
                      )}
                      {step.status === "failed" && (
                        <div className="w-3 h-3 bg-red-500 rounded-full" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-white">
                          {step.agent}
                        </span>
                        <span className="text-xs text-gray-400">
                          {step.timestamp}
                        </span>
                      </div>
                      <div className="text-sm text-gray-300">{step.action}</div>
                      {step.details && (
                        <div className="text-xs text-gray-400 mt-1">
                          {step.details}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Plan Items */}
            {plan.length > 0 && (
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-white">
                  Execution Plan
                </h3>
                <div className="space-y-2">
                  {plan.map((item, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-3 p-3 bg-gray-700/30 rounded-lg"
                    >
                      <span className="text-blue-400 font-medium">
                        #{index + 1}
                      </span>
                      <div className="flex-1">
                        <div className="text-white font-medium">
                          {item.task}
                        </div>
                        <div className="text-sm text-gray-400">{item.tool}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Subtask Status */}
            {subStatus.length > 0 && (
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-white">
                  Task Progress
                </h3>
                <div className="space-y-2">
                  {subStatus.map((subtask, index) => (
                    <div
                      key={subtask.id}
                      className={`p-3 rounded-lg border-l-4 ${
                        subtask.status === "completed"
                          ? "bg-green-500/20 border-green-500"
                          : subtask.status === "running"
                          ? "bg-blue-500/20 border-blue-500"
                          : subtask.status === "failed"
                          ? "bg-red-500/20 border-red-500"
                          : "bg-gray-500/20 border-gray-500"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-white font-medium">
                          {subtask.description}
                        </span>
                        <span
                          className={`text-xs px-2 py-1 rounded ${
                            subtask.status === "completed"
                              ? "bg-green-500 text-white"
                              : subtask.status === "running"
                              ? "bg-blue-500 text-white"
                              : subtask.status === "failed"
                              ? "bg-red-500 text-white"
                              : "bg-gray-500 text-white"
                          }`}
                        >
                          {subtask.status}
                        </span>
                      </div>
                      {subtask.agent_assigned && (
                        <div className="text-sm text-gray-400 mt-1">
                          Agent: {subtask.agent_assigned}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case "browser":
        return (
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 text-white">
                Browser Automation
              </h3>
              {screenshots.length > 0 ? (
                <div className="grid grid-cols-1 gap-4">
                  {screenshots.map((screenshot, index) => (
                    <div key={index} className="relative">
                      <img
                        src={screenshot}
                        alt={`Screenshot ${index + 1}`}
                        className="w-full rounded-lg border border-gray-600"
                      />
                      <div className="absolute top-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                        Step {index + 1}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  {isProcessing &&
                  currentAgent?.toLowerCase().includes("browser")
                    ? "Browser agent is working... Screenshots will appear here"
                    : "No browser activity yet"}
                </div>
              )}
            </div>
          </div>
        );

      case "search":
        return (
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 text-white">
                Search Results & Sources
              </h3>
              {blocks && blocks.length > 0 ? (
                <div className="space-y-3">
                  {blocks
                    .filter(
                      (block) =>
                        block.tool_type.includes("search") ||
                        block.tool_type.includes("browser")
                    )
                    .map((block, index) => (
                      <div
                        key={index}
                        className="bg-gray-700/30 rounded-lg p-3 border-l-4 border-blue-500"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-blue-400">
                            {block.tool_type}
                          </span>
                          <span
                            className={`text-xs px-2 py-1 rounded ${
                              block.success
                                ? "bg-green-500 text-white"
                                : "bg-red-500 text-white"
                            }`}
                          >
                            {block.success ? "Success" : "Failed"}
                          </span>
                        </div>
                        <div className="text-sm text-gray-300 mb-2">
                          {block.block}
                        </div>
                        {block.feedback && (
                          <div className="text-xs text-gray-400">
                            {block.feedback}
                          </div>
                        )}
                      </div>
                    ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  {isProcessing
                    ? "Searching for information..."
                    : "No search results yet"}
                </div>
              )}
            </div>
          </div>
        );

      case "coding":
        return (
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 text-white">
                Code Execution (E2B API)
              </h3>
              {blocks && blocks.length > 0 ? (
                <div className="space-y-3">
                  {blocks
                    .filter(
                      (block) =>
                        block.tool_type.includes("code") ||
                        block.tool_type.includes("python")
                    )
                    .map((block, index) => (
                      <div
                        key={index}
                        className="bg-gray-900/50 rounded-lg p-4 border border-gray-600"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-sm font-medium text-green-400">
                            {block.tool_type}
                          </span>
                          <span
                            className={`text-xs px-2 py-1 rounded ${
                              block.success
                                ? "bg-green-500 text-white"
                                : "bg-red-500 text-white"
                            }`}
                          >
                            {block.success ? "Executed" : "Error"}
                          </span>
                        </div>
                        <pre className="bg-black/50 p-3 rounded text-sm text-gray-300 overflow-x-auto">
                          <code>{block.block}</code>
                        </pre>
                        {block.feedback && (
                          <div className="mt-3 p-3 bg-gray-700/30 rounded text-sm">
                            <strong className="text-blue-400">Output:</strong>
                            <pre className="mt-1 text-gray-300 whitespace-pre-wrap">
                              {block.feedback}
                            </pre>
                          </div>
                        )}
                      </div>
                    ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  {isProcessing && currentAgent?.toLowerCase().includes("code")
                    ? "Code execution in progress..."
                    : "No code executions yet"}
                </div>
              )}
            </div>
          </div>
        );

      case "report":
        return (
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 text-white">
                Final Research Report
              </h3>
              {reportUrl ? (
                <div className="space-y-4">
                  <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      <span className="text-green-400 font-medium">
                        Report Generated Successfully
                      </span>
                    </div>
                    <p className="text-gray-300 mb-4">
                      Your comprehensive research report with infographics and
                      detailed analysis has been generated.
                    </p>
                    <div className="flex gap-3">
                      <a
                        href={reportUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                      >
                        <span>📄</span>
                        View Report
                      </a>
                      <a
                        href={reportUrl}
                        download
                        className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                      >
                        <span>⬇️</span>
                        Download PDF
                      </a>
                    </div>
                  </div>
                </div>
              ) : isProcessing ? (
                <div className="text-center py-8">
                  <div className="inline-flex items-center gap-3 text-blue-400">
                    <div className="animate-spin h-6 w-6 border-2 border-blue-400 border-t-transparent rounded-full"></div>
                    <span>Generating comprehensive report...</span>
                  </div>
                  <div className="text-sm text-gray-400 mt-2">
                    This includes data analysis, infographics, and detailed
                    insights
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  Report will be available after deep search completion
                </div>
              )}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Tab Navigation */}
      <div className="border-b border-gray-700/50 px-4 py-3">
        <div className="flex space-x-1 bg-gray-800/30 rounded-lg p-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as ViewTab)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === tab.id
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-gray-400 hover:text-white hover:bg-gray-700/50"
              }`}
            >
              <span>{tab.icon}</span>
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {renderTabContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
