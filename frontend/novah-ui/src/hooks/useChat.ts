import { useEffect, useState } from "react";
import axios from "axios";
import {
  Message,
  ToolOutput,
  PlanItem,
  SubtaskStatus,
  ExecutionState,
  QueryResponse,
} from "../types";

// Create an Axios instance with a base URL
const apiClient = axios.create({
  baseURL: "http://localhost:8001", // Updated to use port 8001 for minimal API
  withCredentials: false, // Don't send cookies with cross-origin requests
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

export function useChat(threadId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [blocks, setBlocks] = useState<ToolOutput[] | null>(null);
  const [plan, setPlan] = useState<PlanItem[]>([]);
  const [subStatus, setSubStatus] = useState<SubtaskStatus[]>([]);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [status, setStatus] = useState("Agents ready");
  const [executionState, setExecutionState] = useState<ExecutionState | null>(
    null
  );
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRequestInProgress, setIsRequestInProgress] = useState(false);

  useEffect(() => {
    // Only poll if processing and no consecutive errors
    if (!isProcessing) return;

    const interval = setInterval(() => {
      fetchLatest();
      fetchExecutionStatus();
    }, 3000); // Polling every 3 seconds during processing
    return () => clearInterval(interval);
  }, [isProcessing]);

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
        setExecutionState(data.execution_state);
        setPlan(
          data.execution_state.plan?.map((step, index) => ({
            task: step,
            tool: "PlannerAgent",
            subtasks: [],
            status: "pending",
          })) || []
        );
        setSubStatus(data.execution_state.subtask_status || []);
        setReportUrl(data.execution_state.final_report_url || null);
      }

      setBlocks(data.blocks ? Object.values(data.blocks) : null);
      setStatus(data.status || "Ready");
      setIsProcessing(data.done === "false");
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
        setExecutionState(data.execution_state);

        // Update plan from execution state
        if (data.execution_state.plan) {
          setPlan(
            data.execution_state.plan.map((step: string, index: number) => ({
              task: step,
              tool: "PlannerAgent",
              subtasks: [],
              status: "pending",
            }))
          );
        }

        // Update subtask status
        setSubStatus(data.execution_state.subtask_status || []);
        setReportUrl(data.execution_state.final_report_url || null);
        setIsProcessing(data.is_active || false);
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
    setIsProcessing(true);
    setExecutionState(null);
    setPlan([]);
    setSubStatus([]);
    setReportUrl(null);
    setStatus("Initializing...");
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
      setIsProcessing(false);
      setStatus(`Error: ${err.message || "Unknown error"}`);
    } finally {
      setIsRequestInProgress(false);
    }
  };

  const stop = async () => {
    try {
      await apiClient.get("/stop");
      setIsProcessing(false);
    } catch (err) {
      console.error("Error stopping execution:", err);
    }
  };

  const resetBackend = async () => {
    try {
      await apiClient.post("/reset");
      setIsProcessing(false);
      setIsRequestInProgress(false);
      setStatus("Backend reset - ready for new requests");
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
  };
}
