import { useEffect, useState, useCallback, useRef } from "react";
import axios from "axios";
import { Message, AgentViewData } from "../types";

const apiClient = axios.create({
  baseURL: "http://localhost:8002",
});

export function useChat(threadId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRequestInProgress, setIsRequestInProgress] = useState(false);
  const [agentViewData, setAgentViewData] = useState<AgentViewData | null>(
    null
  );

  // Use browser-safe type for setInterval
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null
  );

  const fetchAgentViewData = useCallback(async () => {
    // No need to check isProcessing here, the interval controller will do it.
    try {
      const { data } = await apiClient.get<AgentViewData>("/agent_view_data");
      setAgentViewData(data);

      const newIsProcessing = data.execution?.is_processing || false;
      if (!newIsProcessing && isProcessing) {
        // Check against previous state
        setIsProcessing(false);
        console.log("Processing finished.");
      } else {
        setIsProcessing(newIsProcessing);
      }

      const lastProgress = Object.values(
        data.execution?.agent_progress || {}
      ).pop();
      if (lastProgress?.output) {
        setMessages((prev) => {
          const lastMsg = prev[prev.length - 1];
          const agentName = data.execution?.current_agent || "Agent";
          if (
            lastMsg &&
            lastMsg.type === "agent" &&
            lastMsg.content === lastProgress.output &&
            lastMsg.agentName === agentName
          ) {
            return prev;
          }
          return [
            ...prev,
            { type: "agent", content: lastProgress.output, agentName },
          ];
        });
      }
    } catch (error) {
      console.error("Failed to fetch agent view data:", error);
    }
  }, [isProcessing]); // Dependency on isProcessing to access its current value

  useEffect(() => {
    if (isProcessing) {
      if (!pollingIntervalRef.current) {
        pollingIntervalRef.current = setInterval(fetchAgentViewData, 2500);
      }
    } else {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    }
    // Cleanup on unmount
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [isProcessing, fetchAgentViewData]);

  const sendQuery = async (
    query: string,
    options?: { deepSearch?: boolean }
  ) => {
    if (isRequestInProgress || isProcessing) return;

    setIsRequestInProgress(true);
    setMessages([{ type: "user", content: query }]); // Reset messages on new query
    setIsProcessing(true);
    setAgentViewData(null);

    const endpoint = options?.deepSearch ? "/orchestrator_query" : "/query";
    const payload = {
      query,
      execution_mode: options?.deepSearch ? "deep_research" : "fast",
    };

    try {
      await apiClient.post(endpoint, payload);
      setTimeout(fetchAgentViewData, 500);
    } catch (error: any) {
      const errorMessage = `Error: ${
        error.response?.data?.detail || "Failed to start process"
      }`;
      setMessages((prev) => [
        ...prev,
        { type: "error", content: errorMessage },
      ]);
      setIsProcessing(false);
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
      setMessages([]);
      setAgentViewData(null);
      setIsProcessing(false);
      setIsRequestInProgress(false);
    } catch (err) {
      console.error("Error resetting backend:", err);
    }
  };

  return {
    messages,
    status:
      agentViewData?.execution?.status ||
      (isProcessing ? "Processing..." : "Ready"),
    isProcessing,
    isRequestInProgress,
    agentViewData,
    plan: agentViewData?.plan?.steps || [],
    subStatus: agentViewData?.plan?.subtask_status || [],
    reportUrl: agentViewData?.report?.final_report_url || null,
    executionState: agentViewData?.execution || null,
    blocks: agentViewData?.coding?.executions || [],
    qualityMetrics: agentViewData?.report?.metrics || null,
    sendQuery,
    stop,
    resetBackend,
  };
}
