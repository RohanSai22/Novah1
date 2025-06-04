import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import GlassCard from "./ui/GlassCard";

interface AgentProgress {
  current_task: string;
  current_subtask: string;
  status: "working" | "completed" | "failed";
  start_time: number;
  end_time?: number;
  links_processed: string[];
  search_results: any[];
  screenshots: string[];
  output?: string;
  error?: string;
}

interface AgentProgressData {
  [agentName: string]: AgentProgress;
}

interface SearchResult {
  title: string;
  snippet: string;
  url: string;
  source: string;
}

interface Props {
  isProcessing: boolean;
  currentAgent?: string;
  activeTool?: string;
  currentStep: number;
  totalSteps: number;
  status: string;
}

export default function AgentProgressMonitor({
  isProcessing,
  currentAgent,
  activeTool,
  currentStep,
  totalSteps,
  status,
}: Props) {
  const [agentProgress, setAgentProgress] = useState<AgentProgressData>({});
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [linksProcessed, setLinksProcessed] = useState<string[]>([]);
  const [screenshots, setScreenshots] = useState<string[]>([]);

  // Poll for agent progress updates
  useEffect(() => {
    if (!isProcessing) return;

    const pollProgress = async () => {
      try {
        // Get agent progress
        const progressResponse = await fetch(
          "http://localhost:8001/agent_progress"
        );
        const progressData = await progressResponse.json();

        if (progressData.agent_progress) {
          setAgentProgress(progressData.agent_progress);
        }

        // Get search results
        const searchResponse = await fetch(
          "http://localhost:8001/search_results"
        );
        const searchData = await searchResponse.json();

        if (searchData.search_results) {
          setSearchResults(searchData.search_results);
        }

        // Get links processed
        const linksResponse = await fetch(
          "http://localhost:8001/links_processed"
        );
        const linksData = await linksResponse.json();

        if (linksData.links_processed) {
          setLinksProcessed(linksData.links_processed);
        }

        if (linksData.screenshots) {
          setScreenshots(linksData.screenshots);
        }
      } catch (error) {
        console.error("Failed to fetch agent progress:", error);
      }
    };

    const interval = setInterval(pollProgress, 2000); // Poll every 2 seconds
    pollProgress(); // Initial call

    return () => clearInterval(interval);
  }, [isProcessing]);

  const getAgentIcon = (agentName: string) => {
    switch (agentName) {
      case "SearchAgent":
        return "🔍";
      case "BrowserAgent":
        return "🌐";
      case "CoderAgent":
        return "💻";
      case "Task Planner":
        return "📋";
      case "Report Generator":
        return "📄";
      default:
        return "🤖";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "working":
        return "text-blue-400 bg-blue-500/20";
      case "completed":
        return "text-green-400 bg-green-500/20";
      case "failed":
        return "text-red-400 bg-red-500/20";
      default:
        return "text-gray-400 bg-gray-500/20";
    }
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const calculateDuration = (startTime: number, endTime?: number) => {
    const end = endTime || Date.now() / 1000;
    const duration = Math.floor(end - startTime);
    return `${duration}s`;
  };

  if (!isProcessing && Object.keys(agentProgress).length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {/* Overall Progress */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <GlassCard title="🎯 Execution Progress">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-white/80">Overall Progress</span>
              <span className="text-blue-400 font-mono">
                {currentStep}/{totalSteps}
              </span>
            </div>

            <div className="w-full bg-white/10 rounded-full h-2">
              <motion.div
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{
                  width:
                    totalSteps > 0
                      ? `${(currentStep / totalSteps) * 100}%`
                      : "0%",
                }}
                transition={{ duration: 0.5 }}
              />
            </div>

            {currentAgent && (
              <div className="flex items-center gap-2 text-sm">
                <span className="text-xl">{getAgentIcon(currentAgent)}</span>
                <span className="text-white">Current Agent:</span>
                <span className="text-blue-400 font-medium">
                  {currentAgent}
                </span>
                {activeTool && (
                  <>
                    <span className="text-white/60">using</span>
                    <span className="text-purple-400">{activeTool}</span>
                  </>
                )}
              </div>
            )}

            <div className="text-xs text-gray-400 capitalize">
              Status:{" "}
              <span className={getStatusColor(status).split(" ")[0]}>
                {status}
              </span>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* Active Agents */}
      <AnimatePresence>
        {Object.entries(agentProgress).map(([agentName, progress]) => (
          <motion.div
            key={agentName}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            layout
          >
            <GlassCard title={`${getAgentIcon(agentName)} ${agentName}`}>
              <div className="space-y-3">
                {/* Status Badge */}
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2 py-1 rounded-full text-xs ${getStatusColor(
                      progress.status
                    )}`}
                  >
                    {progress.status}
                  </span>
                </div>

                {/* Current Task */}
                <div>
                  <div className="text-white/80 text-sm">Current Task:</div>
                  <div className="text-white font-medium">
                    {progress.current_task}
                  </div>
                  <div className="text-gray-400 text-sm">
                    {progress.current_subtask}
                  </div>
                </div>

                {/* Timing */}
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Started: {formatTime(progress.start_time)}</span>
                  <span>
                    Duration:{" "}
                    {calculateDuration(progress.start_time, progress.end_time)}
                  </span>
                </div>

                {/* Progress-specific content */}
                {progress.search_results &&
                  progress.search_results.length > 0 && (
                    <div>
                      <div className="text-white/80 text-sm mb-2">
                        Search Results Found:
                      </div>
                      <div className="space-y-1 max-h-32 overflow-y-auto">
                        {progress.search_results
                          .slice(0, 3)
                          .map((result, idx) => (
                            <motion.div
                              key={idx}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              className="bg-white/5 rounded p-2 text-xs"
                            >
                              <div className="text-white font-medium truncate">
                                {result.title}
                              </div>
                              <div className="text-gray-400 truncate">
                                {result.snippet}
                              </div>
                              <div className="text-blue-400">
                                {result.source}
                              </div>
                            </motion.div>
                          ))}
                      </div>
                    </div>
                  )}

                {progress.links_processed &&
                  progress.links_processed.length > 0 && (
                    <div>
                      <div className="text-white/80 text-sm mb-2">
                        Links Processed ({progress.links_processed.length}):
                      </div>
                      <div className="space-y-1 max-h-24 overflow-y-auto">
                        {progress.links_processed.slice(-3).map((link, idx) => (
                          <div
                            key={idx}
                            className="text-xs text-blue-400 truncate"
                          >
                            {link}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                {progress.output && progress.status === "completed" && (
                  <div>
                    <div className="text-white/80 text-sm mb-1">Output:</div>
                    <div className="text-xs text-gray-300 bg-white/5 rounded p-2 max-h-20 overflow-y-auto">
                      {progress.output}
                    </div>
                  </div>
                )}

                {progress.error && progress.status === "failed" && (
                  <div>
                    <div className="text-red-400 text-sm mb-1">Error:</div>
                    <div className="text-xs text-red-300 bg-red-500/10 rounded p-2">
                      {progress.error}
                    </div>
                  </div>
                )}
              </div>
            </GlassCard>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Search Results Summary */}
      {searchResults.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <GlassCard
            title={`🔍 Recent Search Results (${searchResults.length})`}
          >
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {searchResults.slice(0, 5).map((result, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="bg-white/5 rounded-lg p-3 border border-white/10"
                >
                  <div className="text-white font-medium text-sm mb-1">
                    {result.title}
                  </div>
                  <div className="text-gray-400 text-xs mb-2 line-clamp-2">
                    {result.snippet}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-blue-400 text-xs">
                      {result.source}
                    </span>
                    {result.url && (
                      <a
                        href={result.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-purple-400 text-xs hover:underline"
                      >
                        View →
                      </a>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* Links Being Processed */}
      {linksProcessed.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <GlassCard title="🌐 Links Being Processed">
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {linksProcessed.slice(-5).map((link, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-2 text-sm"
                >
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <div className="text-blue-400 truncate flex-1">{link}</div>
                </motion.div>
              ))}
            </div>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}
