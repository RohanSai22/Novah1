import { useEffect, useState } from "react";
import useAgent from "../store/useAgent";
import axios from "axios";
import {
  Message,
  ToolOutput,
  PlanItem,
  SubtaskStatus,
  ExecutionState,
  QueryResponse,
  QualityMetrics,
  ExecutionMode,
  // Add new types for enhanced features
  CodeExecution,
  Screenshot,
  SearchResult,
  TimelineStep,
  AgentViewData,
} from "../types";

// Create an Axios instance with a base URL
const apiClient = axios.create({
  baseURL: "http://localhost:8002", // Updated to use port 8002 for minimal API
  withCredentials: false, // Don't send cookies with cross-origin requests
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

export function useChat(threadId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRequestInProgress, setIsRequestInProgress] = useState(false);
  const [blocks, setBlocks] = useState<ToolOutput[] | null>(null);

  const agentStore = useAgent();
  const {
    plan,
    subStatus,
    reportUrl,
    status,
    executionState,
    isProcessing,
    currentAgent,
    activeTool,
    setState,
    reset,
  } = agentStore;

  // Add orchestrator-specific state and functions
  const [orchestratorAvailable, setOrchestratorAvailable] = useState(false);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(
    null
  );
  const [executionModes, setExecutionModes] = useState<ExecutionMode[]>([]);
  const [currentExecutionMode, setCurrentExecutionMode] = useState("fast");

  // Enhanced state for new features
  const [codeExecutions, setCodeExecutions] = useState<CodeExecution[]>([]);
  const [screenshots, setScreenshots] = useState<Screenshot[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [timeline, setTimeline] = useState<TimelineStep[]>([]);
  const [agentViewData, setAgentViewData] = useState<AgentViewData | null>(
    null
  );

  useEffect(() => {
    // Only poll if processing and no consecutive errors
    if (!isProcessing) return;

    const interval = setInterval(() => {
      fetchLatest();
      fetchExecutionStatus();
      fetchAgentViewData(); // Comprehensive data fetch
      if (orchestratorAvailable) {
        fetchQualityMetrics();
      }
    }, 3000); // Polling every 3 seconds during processing
    return () => clearInterval(interval);
  }, [isProcessing, orchestratorAvailable]);

  // Check orchestrator availability on mount
  useEffect(() => {
    checkOrchestratorStatus();
    fetchExecutionModes();
  }, []);

  const fetchLatest = async () => {
    if (!isProcessing) return; // Don't fetch when not processing

    try {
      const { data }: { data: QueryResponse } = await apiClient.get(
        "/latest_answer"
      );
      if (data.answer) {
        setMessages((prev) => {
          // Avoid duplicate messages
          const lastMessage = prev[prev.length - 1];
          if (lastMessage?.content !== data.answer) {
            return [
              ...prev,
              {
                type: "agent",
                content: data.answer,
                reasoning: data.reasoning,
                agentName: data.agent_name,
                status: data.status,
                uid: data.uid,
              },
            ];
          }
          return prev;
        });
      }

      // Update execution state from latest answer
      if (data.execution_state) {
        setState({
          executionState: data.execution_state,
          plan:
            data.execution_state.plan?.map((step) => ({
              task: step,
              tool: "PlannerAgent",
              subtasks: [],
              status: "pending",
            })) || [],
          subStatus: data.execution_state.subtask_status || [],
          reportUrl: data.execution_state.final_report_url || null,
        });
      }

      setBlocks(data.blocks ? Object.values(data.blocks) : null);
      setState({
        status: data.status || "Ready",
        isProcessing: data.done === "false",
      });
    } catch (err) {
      // Silently fail during polling to avoid console spam
      if (isProcessing) {
        console.log(
          "Polling error (will retry):",
          err instanceof Error ? err.message : err
        );
      }
    }
  };

  const fetchExecutionStatus = async () => {
    if (!isProcessing) return; // Don't fetch when not processing

    try {
      const { data }: { data: any } = await apiClient.get("/execution_status");

      if (data.execution_state) {
        const newPlan = data.execution_state.plan
          ? data.execution_state.plan.map((step: string) => ({
              task: step,
              tool: "PlannerAgent",
              subtasks: [],
              status: "pending",
            }))
          : [];
        setState({ executionState: data.execution_state, plan: newPlan });

        setState({
          subStatus: data.execution_state.subtask_status || [],
          reportUrl: data.execution_state.final_report_url || null,
          isProcessing: data.is_active || false,
        });
      }
    } catch (err) {
      // Execution status endpoint may not always have data, so don't log errors during polling
    }
  };

  const checkBackendStatus = async (): Promise<boolean> => {
    try {
      const response = await apiClient.get("/is_active");
      return response.data.is_active || false;
    } catch (err) {
      console.log("Could not check backend status:", err);
      return false; // Assume not busy if we can't check
    }
  };

  const checkOrchestratorStatus = async () => {
    try {
      const { data } = await apiClient.get("/orchestrator_status");
      setOrchestratorAvailable(data.available);
    } catch (error) {
      console.error("Failed to check orchestrator status:", error);
      setOrchestratorAvailable(false);
    }
  };

  const fetchExecutionModes = async () => {
    try {
      const { data } = await apiClient.get("/execution_modes");
      setExecutionModes(data.modes || []);
    } catch (error) {
      console.error("Failed to fetch execution modes:", error);
    }
  };

  const fetchQualityMetrics = async () => {
    try {
      const { data } = await apiClient.get("/quality_metrics");
      if (data.available) {
        setQualityMetrics(data.metrics);
      }
    } catch (error) {
      console.error("Failed to fetch quality metrics:", error);
    }
  };

  // Enhanced fetch functions for new features
  const fetchCodeExecutions = async () => {
    try {
      const { data } = await apiClient.get("/code_executions");
      setCodeExecutions(data.code_executions || []);
    } catch (error) {
      console.error("Failed to fetch code executions:", error);
    }
  };

  const fetchScreenshots = async () => {
    try {
      const { data } = await apiClient.get("/screenshots");
      const screenshotData = data.screenshots || {};
      const allScreenshots: Screenshot[] = [];

      // Flatten all screenshot types
      Object.values(screenshotData).forEach((screenshots: any) => {
        if (Array.isArray(screenshots)) {
          allScreenshots.push(...screenshots);
        }
      });

      setScreenshots(allScreenshots);
    } catch (error) {
      console.error("Failed to fetch screenshots:", error);
    }
  };

  const fetchSearchResults = async () => {
    try {
      const { data } = await apiClient.get("/search_results");
      setSearchResults(data.search_results || []);
    } catch (error) {
      console.error("Failed to fetch search results:", error);
    }
  };

  const fetchTimelineData = async () => {
    try {
      const { data } = await apiClient.get("/timeline_data");
      setTimeline(data.timeline || []);
    } catch (error) {
      console.error("Failed to fetch timeline data:", error);
    }
  };

  const fetchAgentViewData = async () => {
    try {
      const { data } = await apiClient.get("/agent_view_data");
      setAgentViewData(data);

      // Update individual states from comprehensive data
      if (data.plan) {
        setState({
          plan: data.plan.steps || [],
          subStatus: data.plan.subtask_status || [],
        });
        setTimeline(data.plan.timeline || []);
      }
      if (data.browser) {
        setScreenshots(data.browser.screenshots || []);
      }
      if (data.search) {
        setSearchResults(data.search.results || []);
      }
      if (data.coding) {
        setCodeExecutions(data.coding.executions || []);
      }
      if (data.report) {
        setState({ reportUrl: data.report.final_report_url || null });
        setQualityMetrics(data.report.metrics || null);
      }
      if (data.execution) {
        setState({
          executionState: data.execution as ExecutionState,
          isProcessing: data.execution.is_processing,
          status: data.execution.status || "Ready",
        });
      }
    } catch (error) {
      console.error("Failed to fetch agent view data:", error);
    }
  };

  const simulateCodeExecution = async (
    code: string,
    language: string = "python"
  ) => {
    try {
      const { data } = await apiClient.post("/simulate_code_execution", {
        code,
        language,
      });

      // Refresh code executions
      fetchCodeExecutions();

      return data.execution;
    } catch (error) {
      console.error("Failed to simulate code execution:", error);
      return null;
    }
  };

  // Enhanced sendQuery function with orchestrator support
  const sendOrchestratorQuery = async (
    query: string,
    executionMode: string = "fast",
    qualityValidation: boolean = true,
    generateReport: boolean = true
  ) => {
    if (isRequestInProgress || isProcessing) return;

    try {
      setIsRequestInProgress(true);
      setState({ isProcessing: true, status: "Starting orchestrator..." });

      const response = await apiClient.post("/orchestrator_query", {
        query,
        execution_mode: executionMode,
        quality_validation: qualityValidation,
        generate_report: generateReport,
      });

      if (response.status === 202) {
        setCurrentExecutionMode(executionMode);
        setMessages((prev) => [...prev, { type: "user", content: query }]);
        setMessages((prev) => [
          ...prev,
          {
            type: "agent",
            content: `🧠 Starting ${
              executionMode === "fast" ? "Fast Mode" : "Deep Research"
            } orchestrator...`,
            agentName: "Task Orchestrator",
            status: "initializing",
          },
        ]);
      }
    } catch (error: any) {
      console.error("Orchestrator query failed:", error);
      setState({
        status: `Error: ${
          error.response?.data?.error || "Orchestrator unavailable"
        }`,
        isProcessing: false,
      });
    } finally {
      setIsRequestInProgress(false);
    }
  };

  const sendQuery = async (q: string) => {
    // Prevent duplicate requests
    if (isRequestInProgress || isProcessing) {
      console.log("Request already in progress, ignoring duplicate");
      return;
    }

    // Check if this query was already sent recently
    const lastUserMessage = messages[messages.length - 1];
    if (lastUserMessage?.type === "user" && lastUserMessage.content === q) {
      console.log("Duplicate query detected, ignoring");
      return;
    }

    // Check if backend is busy before sending
    const backendBusy = await checkBackendStatus();
    if (backendBusy) {
      console.log("Backend is busy, showing user message");
      setMessages((prev) => [
        ...prev,
        { type: "user", content: q },
        {
          type: "agent",
          content:
            "The system is currently processing another request. Please wait a moment and try again.",
          reasoning: "Backend busy",
          agentName: "System",
          status: "waiting",
          uid: `busy-${Date.now()}`,
        },
      ]);
      return;
    }

    console.log("Sending query:", q);
    setIsRequestInProgress(true);

    // Add user message immediately (only once)
    setMessages((prev) => [...prev, { type: "user", content: q }]);

    // Reset all states for new query
    setState({
      isProcessing: true,
      executionState: null,
      plan: [],
      subStatus: [],
      reportUrl: null,
      status: "Initializing...",
    });
    setBlocks(null);

    // Add a temporary "analyzing" message
    setMessages((prev) => [
      ...prev,
      {
        type: "agent",
        content: "I'm analyzing your request and starting the process...",
        reasoning: "Initial analysis",
        agentName: "Router",
        status: "planning",
        uid: `temp-${Date.now()}`,
      },
    ]);

    try {
      const response = await apiClient.post<any>("/query", {
        query: q,
        tts_enabled: false,
      });

      console.log("Query sent successfully:", response.data);

      // Start polling for updates
      setTimeout(() => {
        fetchLatest();
        fetchExecutionStatus();
      }, 1000);
    } catch (err: any) {
      console.error("Error sending query:", err);
      setMessages((prev) => [
        ...prev.slice(0, -1), // Remove the "analyzing" message
        {
          type: "agent",
          content:
            "Sorry, I encountered an error while processing your request. Please try again.",
          reasoning: `Error: ${err.message || "Unknown error"}`,
          agentName: "System",
          status: "error",
          uid: `error-${Date.now()}`,
        },
      ]);
      setState({
        isProcessing: false,
        status: `Error: ${err.message || "Unknown error"}`,
      });
    } finally {
      setIsRequestInProgress(false);
    }
  };

  const stop = async () => {
    try {
      await apiClient.get("/stop");
      setState({ isProcessing: false });
    } catch (err) {
      console.error("Error stopping execution:", err);
    }
  };

  const resetBackend = async () => {
    try {
      await apiClient.post("/reset");
      setState({
        isProcessing: false,
        status: "Backend reset - ready for new requests",
      });
      setIsRequestInProgress(false);
      console.log("Backend state reset successfully");
    } catch (err) {
      console.error("Error resetting backend:", err);
    }
  };

  return {
    messages,
    status,
    sendQuery,
    stop,
    resetBackend,
    blocks,
    plan,
    subStatus,
    reportUrl,
    executionState,
    isProcessing,
    isRequestInProgress,
    orchestratorAvailable,
    qualityMetrics,
    executionModes,
    currentExecutionMode,
    sendOrchestratorQuery,
    fetchQualityMetrics,
    // Enhanced features
    codeExecutions,
    screenshots,
    searchResults,
    timeline,
    agentViewData,
    fetchCodeExecutions,
    fetchScreenshots,
    fetchSearchResults,
    fetchTimelineData,
    fetchAgentViewData,
    simulateCodeExecution,
  };
}
