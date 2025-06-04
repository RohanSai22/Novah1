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
  const [workspaceWidth, setWorkspaceWidth] = useState(50); // Workspace width percentage
  const [showAdvancedMode, setShowAdvancedMode] = useState(false);
  const [currentQuery, setCurrentQuery] = useState(""); // Track current input query
  const [selectedExecutionMode, setSelectedExecutionMode] = useState("fast");
  const [qualityValidationEnabled, setQualityValidationEnabled] =
    useState(true);
  const [generateReportEnabled, setGenerateReportEnabled] = useState(true);

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
    orchestratorAvailable,
    qualityMetrics,
    executionModes,
    currentExecutionMode,
    sendOrchestratorQuery,
    fetchQualityMetrics,
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
    const deepSearch = location.state?.deepSearch;
    if (
      initialQuery &&
      messages.length === 0 &&
      !isProcessing &&
      !isRequestInProgress
    ) {
      console.log(
        "Processing initial query from home page:",
        initialQuery,
        "Deep search:",
        deepSearch
      );

      // Use a timeout to ensure the component is fully mounted
      const timeoutId = setTimeout(() => {
        if (deepSearch && orchestratorAvailable) {
          // Use deep search mode
          sendOrchestratorQuery(initialQuery, "deep_research", true, true);
        } else {
          // Use normal search
          sendQuery(initialQuery);
        }
        // Clear the location state to prevent re-execution
        window.history.replaceState({}, document.title);
      }, 100);
      return () => clearTimeout(timeoutId);
    }
  }, []); // Remove dependencies to prevent re-runs

  const handleSubmit = (prompt: string, options?: any) => {
    setCurrentQuery(prompt); // Update current query for deep search

    // Check if deep search was requested from options
    const useDeepSearch = options?.deepSearch;

    if ((showAdvancedMode || useDeepSearch) && orchestratorAvailable) {
      // Use orchestrator for advanced queries or deep search
      sendOrchestratorQuery(
        prompt,
        useDeepSearch ? "deep_research" : selectedExecutionMode,
        qualityValidationEnabled,
        generateReportEnabled
      );
    } else {
      // Use regular agent system
      sendQuery(prompt);
    }
  };

  const handleDeepSearch = (query: string, options: any) => {
    if (orchestratorAvailable) {
      sendOrchestratorQuery(
        query,
        options.executionMode || "deep_research",
        options.qualityValidation !== false,
        options.generateReport !== false
      );
    } else {
      // Fallback to regular search
      sendQuery(query);
    }
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

      {/* Main Chat Area - Dynamic width based on workspace */}
      <div
        className={`flex-1 flex flex-col transition-all duration-300`}
        style={{ width: showWorkspace ? `${100 - workspaceWidth}%` : "100%" }}
      >
        {/* Clean Header with essentials only */}
        <div className="flex items-center justify-between p-4 border-b border-white/10 bg-black/50">
          {/* Left: Menu */}
          <button
            onClick={() => setShowSidebar(true)}
            className="p-2 hover:bg-white/10 rounded transition-colors"
            title="Open sidebar"
          >
            ☰
          </button>

          {/* Center: Novah name */}
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <h1 className="text-xl font-light text-white">Novah</h1>
          </div>

          {/* Right: Status indicator */}
          <div className="flex items-center gap-3">
            {/* Agent status indicator */}
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  isProcessing ? "bg-red-500 animate-pulse" : "bg-green-500"
                }`}
              ></div>
              <span className="text-sm text-gray-300">
                {isProcessing ? "Working" : "Ready"}
              </span>
            </div>

            {/* Agent workspace toggle */}
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

        {/* Advanced Options - Collapsible */}
        {orchestratorAvailable && (
          <div className="border-b border-white/10">
            <button
              onClick={() => setShowAdvancedMode(!showAdvancedMode)}
              className="w-full flex items-center justify-between p-3 hover:bg-white/5 transition-colors"
            >
              <span className="text-sm font-medium text-gray-300">
                Advanced Options
              </span>
              <div className="flex items-center gap-2">
                {isProcessing && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      stop();
                    }}
                    className="px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors"
                    title="Stop processing"
                  >
                    Stop
                  </button>
                )}
                {(status.includes("Error") ||
                  status.includes("busy") ||
                  status.includes("Server busy")) && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      resetBackend();
                    }}
                    className="px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors"
                    title="Reset backend if stuck"
                  >
                    Reset
                  </button>
                )}
                <span className="text-gray-400 text-sm">
                  {showAdvancedMode ? "▲" : "▼"}
                </span>
              </div>
            </button>

            {showAdvancedMode && (
              <div className="p-4 bg-gray-900/30">
                <ExecutionModeSelector
                  selectedMode={selectedExecutionMode}
                  onModeChange={setSelectedExecutionMode}
                  qualityValidation={qualityValidationEnabled}
                  onQualityValidationChange={setQualityValidationEnabled}
                  generateReport={generateReportEnabled}
                  onGenerateReportChange={setGenerateReportEnabled}
                  disabled={isProcessing}
                />
              </div>
            )}
          </div>
        )}

        {/* Quality Metrics - Show when available */}
        {qualityMetrics && orchestratorAvailable && (
          <div className="border-b border-white/10 p-4">
            <QualityMetricsCard metrics={qualityMetrics} isVisible={true} />
          </div>
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

      {/* Enhanced Agent Workspace - Dynamic width */}
      <WorkspaceSlider
        isOpen={showWorkspace}
        onToggle={() => setShowWorkspace(!showWorkspace)}
        width={workspaceWidth}
        onWidthChange={setWorkspaceWidth}
        title="Agent View"
      >
        {/* Enhanced Agent View with Tabs */}
        <EnhancedAgentView
          isProcessing={isProcessing}
          executionState={executionState}
          plan={plan}
          subStatus={subStatus}
          blocks={blocks}
          reportUrl={reportUrl}
          currentAgent={executionState?.current_agent}
          activeTool={executionState?.active_tool}
        />
      </WorkspaceSlider>
    </div>
  );
}
