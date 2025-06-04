import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import GlassCard from "./ui/GlassCard";

interface ExecutionMode {
  id: string;
  name: string;
  description: string;
  estimatedTime: string;
  features: string[];
  icon: string;
  color: string;
}

interface Props {
  selectedMode: string;
  onModeChange: (mode: string) => void;
  qualityValidation: boolean;
  onQualityValidationChange: (enabled: boolean) => void;
  generateReport: boolean;
  onGenerateReportChange: (enabled: boolean) => void;
  disabled?: boolean;
}

const executionModes: ExecutionMode[] = [
  {
    id: "fast",
    name: "Fast Mode",
    description: "Quick research with essential information",
    estimatedTime: "30-60 seconds",
    features: ["Basic search", "Quick analysis", "Summary report"],
    icon: "⚡",
    color: "from-blue-500/20 to-cyan-500/20 border-blue-500/30",
  },
  {
    id: "deep_research",
    name: "Deep Research Mode",
    description: "Comprehensive research with detailed analysis",
    estimatedTime: "2-5 minutes",
    features: [
      "Multi-engine search",
      "Quality validation",
      "Comprehensive report",
      "Visual analysis",
      "Data visualization",
    ],
    icon: "🧠",
    color: "from-purple-500/20 to-pink-500/20 border-purple-500/30",
  },
];

export default function ExecutionModeSelector({
  selectedMode,
  onModeChange,
  qualityValidation,
  onQualityValidationChange,
  generateReport,
  onGenerateReportChange,
  disabled = false,
}: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const selectedModeData = executionModes.find(
    (mode) => mode.id === selectedMode
  );

  return (
    <div className="w-full max-w-4xl mx-auto mb-6">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative"
      >
        {/* Compact Mode Selector */}
        <div className="flex items-center gap-4 mb-4">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            disabled={disabled}
            className={`flex items-center gap-3 px-4 py-2 rounded-xl border transition-all duration-200 ${
              disabled
                ? "opacity-50 cursor-not-allowed bg-gray-800 border-gray-700"
                : "hover:scale-105 bg-gray-900/60 border-gray-600/40 hover:bg-gray-900/80"
            }`}
          >
            <span className="text-xl">{selectedModeData?.icon}</span>
            <div className="text-left">
              <div className="text-white font-medium text-sm">
                {selectedModeData?.name}
              </div>
              <div className="text-gray-400 text-xs">
                {selectedModeData?.estimatedTime}
              </div>
            </div>
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
              className="text-gray-400"
            >
              ▼
            </motion.div>
          </button>

          {/* Quick Options */}
          <div className="flex items-center gap-3 text-sm">
            <label
              className={`flex items-center gap-2 ${
                disabled ? "opacity-50" : ""
              }`}
            >
              <input
                type="checkbox"
                checked={qualityValidation}
                onChange={(e) => onQualityValidationChange(e.target.checked)}
                disabled={disabled}
                className="w-4 h-4 text-blue-600 bg-gray-900 border-gray-600 rounded focus:ring-blue-500"
              />
              <span className="text-gray-300">Quality Validation</span>
            </label>
            <label
              className={`flex items-center gap-2 ${
                disabled ? "opacity-50" : ""
              }`}
            >
              <input
                type="checkbox"
                checked={generateReport}
                onChange={(e) => onGenerateReportChange(e.target.checked)}
                disabled={disabled}
                className="w-4 h-4 text-blue-600 bg-gray-900 border-gray-600 rounded focus:ring-blue-500"
              />
              <span className="text-gray-300">Generate Report</span>
            </label>
          </div>
        </div>

        {/* Expanded Mode Details */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <GlassCard title="🎯 Execution Mode Selection">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {executionModes.map((mode) => (
                    <motion.button
                      key={mode.id}
                      onClick={() => !disabled && onModeChange(mode.id)}
                      disabled={disabled}
                      whileHover={!disabled ? { scale: 1.02 } : undefined}
                      whileTap={!disabled ? { scale: 0.98 } : undefined}
                      className={`relative p-4 rounded-xl border text-left transition-all duration-200 ${
                        selectedMode === mode.id
                          ? `bg-gradient-to-br ${mode.color} ring-2 ring-white/20`
                          : "bg-gray-900/40 border-gray-700/50 hover:bg-gray-900/60"
                      } ${
                        disabled
                          ? "opacity-50 cursor-not-allowed"
                          : "cursor-pointer"
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <span className="text-2xl">{mode.icon}</span>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="text-white font-semibold">
                              {mode.name}
                            </h3>
                            {selectedMode === mode.id && (
                              <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                className="w-2 h-2 bg-green-400 rounded-full"
                              />
                            )}
                          </div>
                          <p className="text-gray-400 text-sm mb-3">
                            {mode.description}
                          </p>
                          <div className="text-xs text-gray-500 mb-3">
                            ⏱️ {mode.estimatedTime}
                          </div>
                          <div className="space-y-1">
                            {mode.features.map((feature, index) => (
                              <div
                                key={index}
                                className="flex items-center gap-2 text-xs text-gray-300"
                              >
                                <span className="w-1 h-1 bg-blue-400 rounded-full"></span>
                                {feature}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </motion.button>
                  ))}
                </div>

                {/* Advanced Options */}
                <div className="mt-6 pt-4 border-t border-gray-700/50">
                  <h4 className="text-white font-medium mb-4">
                    🔧 Advanced Options
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div
                      className={`p-3 rounded-lg border ${
                        qualityValidation
                          ? "bg-green-500/10 border-green-500/30"
                          : "bg-gray-900/40 border-gray-700/50"
                      }`}
                    >
                      <label
                        className={`flex items-center gap-3 ${
                          disabled ? "opacity-50" : "cursor-pointer"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={qualityValidation}
                          onChange={(e) =>
                            onQualityValidationChange(e.target.checked)
                          }
                          disabled={disabled}
                          className="w-5 h-5 text-green-600 bg-gray-900 border-gray-600 rounded focus:ring-green-500"
                        />
                        <div>
                          <div className="text-white font-medium">
                            Quality Validation
                          </div>
                          <div className="text-gray-400 text-sm">
                            Comprehensive fact-checking and bias analysis
                          </div>
                        </div>
                      </label>
                    </div>

                    <div
                      className={`p-3 rounded-lg border ${
                        generateReport
                          ? "bg-blue-500/10 border-blue-500/30"
                          : "bg-gray-900/40 border-gray-700/50"
                      }`}
                    >
                      <label
                        className={`flex items-center gap-3 ${
                          disabled ? "opacity-50" : "cursor-pointer"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={generateReport}
                          onChange={(e) =>
                            onGenerateReportChange(e.target.checked)
                          }
                          disabled={disabled}
                          className="w-5 h-5 text-blue-600 bg-gray-900 border-gray-600 rounded focus:ring-blue-500"
                        />
                        <div>
                          <div className="text-white font-medium">
                            Generate Report
                          </div>
                          <div className="text-gray-400 text-sm">
                            Create downloadable comprehensive report
                          </div>
                        </div>
                      </label>
                    </div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
