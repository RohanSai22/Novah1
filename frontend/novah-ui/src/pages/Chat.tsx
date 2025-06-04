import { useParams, useLocation } from "react-router-dom";
import { useChat } from "../hooks/useChat";
import ChatArea from "../components/ChatArea";
import PromptInput from "../components/PromptInput";
import Sidebar from "../components/Sidebar";
import AgentWorkspace from "../components/AgentWorkspace";
import AgentProgressMonitor from "../components/AgentProgressMonitor";
import { useState, useEffect } from "react";
import { Thread } from "../types";
import PlanList from "../components/PlanList";
import FinalReportCard from "../components/FinalReportCard";

export default function Chat() {
  const { threadId = "" } = useParams();
  const location = useLocation();
  const [showSidebar, setShowSidebar] = useState(false);
  const [showWorkspace, setShowWorkspace] = useState(false);

  const {
    messages,
    sendQuery,
    blocks,
    status,
    plan,
    subStatus,
    reportUrl,
    executionState,
    isProcessing,
    isRequestInProgress,
    resetBackend,
    stop,
  } = useChat(threadId);
  const [threads] = useState<Thread[]>([{ id: threadId, messages, blocks }]);

  // Auto-show workspace when planning starts
  useEffect(() => {
    if (plan.length > 0 || subStatus.length > 0) {
      setShowWorkspace(true);
    }
  }, [plan.length, subStatus.length]);

  // Handle initial query from home page navigation
  useEffect(() => {
    const initialQuery = location.state?.initialQuery;
    if (
      initialQuery &&
      messages.length === 0 &&
      !isProcessing &&
      !isRequestInProgress
    ) {
      console.log("Processing initial query from home page:", initialQuery);
      // Use a timeout to ensure the component is fully mounted
      const timeoutId = setTimeout(() => {
        sendQuery(initialQuery);
        // Clear the location state to prevent re-execution
        window.history.replaceState({}, document.title);
      }, 100);
      return () => clearTimeout(timeoutId);
    }
  }, []); // Remove dependencies to prevent re-runs

  const handleSubmit = (prompt: string) => {
    sendQuery(prompt);
  };

  const currentAgent = executionState?.current_agent || "System";
  const currentTool = executionState?.active_tool || null;
  const currentStep = executionState?.current_step || 0;
  const totalSteps = executionState?.total_steps || 0;

  return (
    <div className="flex h-screen bg-black text-white overflow-hidden">
      <Sidebar
        threads={threads}
        open={showSidebar}
        onClose={() => setShowSidebar(false)}
      />

      {/* Main Chat Area - Takes full width when workspace is closed, 50% when open */}
      <div
        className={`flex-1 flex flex-col transition-all duration-300 ${
          showWorkspace ? "w-1/2" : "w-full"
        }`}
      >
        {/* Header with status and controls */}
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowSidebar(true)}
              className="p-2 hover:bg-white/10 rounded transition-colors"
              title="Open sidebar"
            >
              ☰
            </button>
            <div className="flex flex-col">
              <div className="text-sm font-medium">
                {isProcessing ? (
                  <span className="text-blue-400">
                    🤖 {currentAgent} {currentTool && `• ${currentTool}`}
                  </span>
                ) : (
                  <span className="text-green-400">Ready</span>
                )}
              </div>
              {totalSteps > 0 && (
                <div className="text-xs text-gray-400">
                  Step {currentStep} of {totalSteps}
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {showWorkspace && (
              <button
                onClick={() => setShowWorkspace(false)}
                className="px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white text-sm rounded transition-colors"
                title="Hide workspace"
              >
                Hide Workspace
              </button>
            )}
            {!showWorkspace && (plan.length > 0 || subStatus.length > 0) && (
              <button
                onClick={() => setShowWorkspace(true)}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
                title="Show workspace"
              >
                Show Workspace
              </button>
            )}
            {isProcessing && (
              <button
                onClick={stop}
                className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                title="Stop processing"
              >
                Stop
              </button>
            )}
            {(status.includes("Error") ||
              status.includes("busy") ||
              status.includes("Server busy")) && (
              <button
                onClick={resetBackend}
                className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                title="Reset backend if stuck"
              >
                Reset
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {isProcessing && totalSteps > 0 && (
          <div className="w-full bg-gray-700 h-1">
            <div
              className="bg-blue-500 h-1 transition-all duration-300"
              style={{ width: `${(currentStep / totalSteps) * 100}%` }}
            />
          </div>
        )}

        {/* Plan List - only show when workspace is closed */}
        {!showWorkspace && (
          <PlanList plan={plan} subStatus={subStatus} reportUrl={reportUrl} />
        )}

        {/* Chat Messages */}
        <div className="flex-1 overflow-hidden">
          <ChatArea messages={messages} />
        </div>

        {/* Final Report Card - only show when not in workspace mode */}
        {!showWorkspace && reportUrl && <FinalReportCard url={reportUrl} />}

        {/* Input Area */}
        <div className="p-4 border-t border-white/10">
          <PromptInput
            onSubmit={handleSubmit}
            disabled={
              isProcessing ||
              isRequestInProgress ||
              status.includes("busy") ||
              status.includes("Retrying")
            }
            placeholder={
              isProcessing
                ? "Processing your request..."
                : isRequestInProgress
                ? "Sending request..."
                : status.includes("busy")
                ? "System busy - please wait..."
                : "Ask anything... Try: 'Research the latest AI trends' or 'Help me build a React app'"
            }
          />
        </div>
      </div>

      {/* Planning Workspace - 50% width when open */}
      {showWorkspace && (
        <div className="w-1/2 border-l border-white/20 bg-black/50 backdrop-blur flex flex-col">
          {/* Workspace Header */}
          <div className="p-4 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-lg">🛠️</span>
              <h2 className="text-lg font-semibold">Agent Workspace</h2>
            </div>
            <button
              onClick={() => setShowWorkspace(false)}
              className="text-gray-400 hover:text-white transition-colors"
              title="Close workspace"
            >
              ✕
            </button>
          </div>

          {/* Workspace Content */}
          <div className="flex-1 overflow-y-auto">
            {/* Agent Progress Monitor */}
            <AgentProgressMonitor
              isProcessing={isProcessing}
              currentAgent={executionState?.current_agent}
              activeTool={executionState?.active_tool}
              currentStep={executionState?.current_step || 0}
              totalSteps={executionState?.total_steps || 0}
              status={executionState?.status || status}
            />

            {/* Plan List */}
            <PlanList plan={plan} subStatus={subStatus} reportUrl={reportUrl} />

            {/* Agent Tools Output */}
            {blocks && blocks.length > 0 && (
              <div className="p-4 border-t border-white/10">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Tool Outputs
                </h3>
                <AgentWorkspace
                  blocks={blocks}
                  executionState={executionState}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
