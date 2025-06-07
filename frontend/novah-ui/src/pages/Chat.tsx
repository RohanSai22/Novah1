import { useParams, useLocation } from "react-router-dom";
import { useChat } from "../hooks/useChat";
import ChatArea from "../components/ChatArea";
import PromptInput from "../components/PromptInput";
import Sidebar from "../components/Sidebar";
import ExecutionModeSelector from "../components/ExecutionModeSelector";
import QualityMetricsCard from "../components/QualityMetricsCard";
import WorkspaceSlider from "../components/WorkspaceSlider";
import EnhancedAgentView from "../components/EnhancedAgentView";
import FinalReportCard from "../components/FinalReportCard";
import { useState, useEffect } from "react";
import { Thread } from "../types";

export default function Chat() {
  const { threadId = "" } = useParams();
  const location = useLocation();
  const [showSidebar, setShowSidebar] = useState(false);
  const [showWorkspace, setShowWorkspace] = useState(false);
  const [workspaceWidth, setWorkspaceWidth] = useState(50);
  const [showAdvancedMode, setShowAdvancedMode] = useState(false);
  const [selectedExecutionMode, setSelectedExecutionMode] = useState<
    "fast" | "deep_research"
  >("fast");
  const [qualityValidationEnabled, setQualityValidationEnabled] =
    useState(true);
  const [generateReportEnabled, setGenerateReportEnabled] = useState(true);

  const {
    messages,
    sendQuery,
    status,
    plan,
    subStatus,
    reportUrl,
    executionState,
    isProcessing,
    isRequestInProgress,
    resetBackend,
    stop,
    agentViewData,
    qualityMetrics,
  } = useChat(threadId);

  const [threads] = useState<Thread[]>([
    {
      id: threadId,
      messages,
      blocks: agentViewData?.coding?.executions || null,
    },
  ]);

  useEffect(() => {
    if (plan.length > 0 || subStatus.length > 0 || isProcessing) {
      setShowWorkspace(true);
    }
  }, [plan.length, subStatus.length, isProcessing]);

  useEffect(() => {
    const initialQuery = location.state?.initialQuery;
    const deepSearch = location.state?.deepSearch;
    if (
      initialQuery &&
      messages.length === 0 &&
      !isProcessing &&
      !isRequestInProgress
    ) {
      const mode = deepSearch ? "deep_research" : "fast";
      const timeoutId = setTimeout(() => {
        sendQuery(initialQuery, { deepSearch: mode === "deep_research" });
        window.history.replaceState({}, document.title);
      }, 100);
      return () => clearTimeout(timeoutId);
    }
  }, [
    location.state,
    messages.length,
    isProcessing,
    isRequestInProgress,
    sendQuery,
  ]);

  const handleSubmit = (prompt: string, options?: any) => {
    const mode = options?.deepSearch ? "deep_research" : selectedExecutionMode;
    sendQuery(prompt, { deepSearch: mode === "deep_research" });
  };

  return (
    <div className="flex h-screen bg-black text-white overflow-hidden">
      <Sidebar
        threads={threads}
        open={showSidebar}
        onClose={() => setShowSidebar(false)}
      />
      <div
        className={`flex-1 flex flex-col transition-all duration-300`}
        style={{ width: showWorkspace ? `${100 - workspaceWidth}%` : "100%" }}
      >
        <div className="flex items-center justify-between p-4 border-b border-white/10 bg-black/50">
          <button
            onClick={() => setShowSidebar(true)}
            className="p-2 hover:bg-white/10 rounded transition-colors"
            title="Open sidebar"
          >
            ☰
          </button>
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <h1 className="text-xl font-light text-white">Novah</h1>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  isProcessing ? "bg-red-500 animate-pulse" : "bg-green-500"
                }`}
              ></div>
              <span className="text-sm text-gray-300">
                {isProcessing ? status : "Ready"}
              </span>
            </div>
            {(plan.length > 0 || subStatus.length > 0 || isProcessing) && (
              <button
                onClick={() => setShowWorkspace(!showWorkspace)}
                className={`p-2 rounded transition-colors ${
                  showWorkspace
                    ? "bg-blue-600 hover:bg-blue-700 text-white"
                    : "bg-gray-600 hover:bg-gray-700 text-white"
                }`}
                title={showWorkspace ? "Hide Agent View" : "Show Agent View"}
              >
                {showWorkspace ? "›" : "‹"}
              </button>
            )}
          </div>
        </div>
        <div className="border-b border-white/10">
          <div
            onClick={() => setShowAdvancedMode(!showAdvancedMode)}
            className="w-full flex items-center justify-between p-3 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <span className="text-sm font-medium text-gray-300">
              Advanced Options
            </span>
            <div className="flex items-center gap-2">
              {isProcessing && (
                <div
                  onClick={(e) => {
                    e.stopPropagation();
                    stop();
                  }}
                  className="px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors cursor-pointer"
                  title="Stop processing"
                >
                  Stop
                </div>
              )}
              {(status.includes("Error") || status.includes("Failed")) && (
                <div
                  onClick={(e) => {
                    e.stopPropagation();
                    resetBackend();
                  }}
                  className="px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors cursor-pointer"
                  title="Reset backend if stuck"
                >
                  Reset
                </div>
              )}
              <span className="text-gray-400 text-sm">
                {showAdvancedMode ? "▲" : "▼"}
              </span>
            </div>
          </div>
          {showAdvancedMode && (
            <div className="p-4 bg-gray-900/30">
              <ExecutionModeSelector
                selectedMode={selectedExecutionMode}
                onModeChange={(mode) =>
                  setSelectedExecutionMode(mode as "fast" | "deep_research")
                }
                qualityValidation={qualityValidationEnabled}
                onQualityValidationChange={setQualityValidationEnabled}
                generateReport={generateReportEnabled}
                onGenerateReportChange={setGenerateReportEnabled}
                disabled={isProcessing}
              />
            </div>
          )}
        </div>
        {qualityMetrics && (
          <div className="border-b border-white/10 p-4">
            <QualityMetricsCard metrics={qualityMetrics} isVisible={true} />
          </div>
        )}
        <div className="flex-1 overflow-y-auto">
          <ChatArea messages={messages} />
        </div>
        {!showWorkspace && reportUrl && <FinalReportCard url={reportUrl} />}
        <div className="p-4 border-t border-white/10">
          <PromptInput
            onSubmit={handleSubmit}
            disabled={isProcessing || isRequestInProgress}
            placeholder={
              isProcessing
                ? "Processing your request..."
                : isRequestInProgress
                ? "Sending request..."
                : "Ask anything... Try: 'Research the latest AI trends'"
            }
          />
        </div>
      </div>
      <WorkspaceSlider
        isOpen={showWorkspace}
        onToggle={() => setShowWorkspace(!showWorkspace)}
        width={workspaceWidth}
        onWidthChange={setWorkspaceWidth}
        title="Agent View"
      >
        <EnhancedAgentView
          isProcessing={isProcessing}
          executionState={executionState}
          plan={plan}
          subStatus={subStatus}
          blocks={agentViewData?.coding?.executions || []}
          reportUrl={reportUrl}
          currentAgent={executionState?.current_agent}
          activeTool={executionState?.active_tool}
        />
      </WorkspaceSlider>
    </div>
  );
}
