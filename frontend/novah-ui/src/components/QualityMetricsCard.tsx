import React from "react";
import { motion } from "framer-motion";
import GlassCard from "./ui/GlassCard";

interface QualityMetrics {
  confidence_score: number;
  source_credibility: number;
  completeness_score: number;
  bias_score?: number;
  recency_score?: number;
  issues: string[];
  recommendations: string[];
}

interface Props {
  metrics: QualityMetrics | null;
  isVisible: boolean;
}

export default function QualityMetricsCard({ metrics, isVisible }: Props) {
  if (!metrics || !isVisible) return null;

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return "text-green-400";
    if (score >= 0.6) return "text-yellow-400";
    return "text-red-400";
  };

  const getScoreIcon = (score: number) => {
    if (score >= 0.8) return "✅";
    if (score >= 0.6) return "⚠️";
    return "❌";
  };

  const formatScore = (score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mb-6"
    >
      <GlassCard title="🔍 Quality Assessment">
        <div className="space-y-6">
          {/* Overall Score */}
          <div className="text-center">
            <div className="text-4xl font-bold mb-2">
              <span className={getScoreColor(metrics.confidence_score)}>
                {formatScore(metrics.confidence_score)}
              </span>
            </div>
            <div className="text-gray-400">Overall Confidence Score</div>
          </div>

          {/* Detailed Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl mb-1">
                {getScoreIcon(metrics.source_credibility / 10)}
              </div>
              <div
                className={`text-lg font-semibold ${getScoreColor(
                  metrics.source_credibility / 10
                )}`}
              >
                {metrics.source_credibility.toFixed(1)}/10
              </div>
              <div className="text-xs text-gray-400">Source Credibility</div>
            </div>

            <div className="text-center">
              <div className="text-2xl mb-1">
                {getScoreIcon(metrics.completeness_score)}
              </div>
              <div
                className={`text-lg font-semibold ${getScoreColor(
                  metrics.completeness_score
                )}`}
              >
                {formatScore(metrics.completeness_score)}
              </div>
              <div className="text-xs text-gray-400">Completeness</div>
            </div>

            {metrics.bias_score !== undefined && (
              <div className="text-center">
                <div className="text-2xl mb-1">
                  {getScoreIcon(1 - metrics.bias_score)}
                </div>
                <div
                  className={`text-lg font-semibold ${getScoreColor(
                    1 - metrics.bias_score
                  )}`}
                >
                  {formatScore(1 - metrics.bias_score)}
                </div>
                <div className="text-xs text-gray-400">Objectivity</div>
              </div>
            )}

            {metrics.recency_score !== undefined && (
              <div className="text-center">
                <div className="text-2xl mb-1">
                  {getScoreIcon(metrics.recency_score)}
                </div>
                <div
                  className={`text-lg font-semibold ${getScoreColor(
                    metrics.recency_score
                  )}`}
                >
                  {formatScore(metrics.recency_score)}
                </div>
                <div className="text-xs text-gray-400">Recency</div>
              </div>
            )}
          </div>

          {/* Issues and Recommendations */}
          {(metrics.issues.length > 0 ||
            metrics.recommendations.length > 0) && (
            <div className="grid md:grid-cols-2 gap-4">
              {metrics.issues.length > 0 && (
                <div>
                  <h4 className="text-red-400 font-medium mb-2 flex items-center gap-2">
                    ⚠️ Issues Identified ({metrics.issues.length})
                  </h4>
                  <div className="space-y-2">
                    {metrics.issues.slice(0, 3).map((issue, index) => (
                      <div
                        key={index}
                        className="text-sm text-gray-300 bg-red-500/10 rounded p-2 border border-red-500/20"
                      >
                        • {issue}
                      </div>
                    ))}
                    {metrics.issues.length > 3 && (
                      <div className="text-xs text-gray-500">
                        +{metrics.issues.length - 3} more issues
                      </div>
                    )}
                  </div>
                </div>
              )}

              {metrics.recommendations.length > 0 && (
                <div>
                  <h4 className="text-blue-400 font-medium mb-2 flex items-center gap-2">
                    💡 Recommendations ({metrics.recommendations.length})
                  </h4>
                  <div className="space-y-2">
                    {metrics.recommendations.slice(0, 3).map((rec, index) => (
                      <div
                        key={index}
                        className="text-sm text-gray-300 bg-blue-500/10 rounded p-2 border border-blue-500/20"
                      >
                        • {rec}
                      </div>
                    ))}
                    {metrics.recommendations.length > 3 && (
                      <div className="text-xs text-gray-500">
                        +{metrics.recommendations.length - 3} more
                        recommendations
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Quality Score Legend */}
          <div className="border-t border-gray-700/50 pt-4">
            <div className="text-xs text-gray-400 mb-2">
              Quality Score Range:
            </div>
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                <span className="text-green-400">Excellent (80%+)</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
                <span className="text-yellow-400">Good (60-79%)</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                <span className="text-red-400">Needs Review (&lt;60%)</span>
              </div>
            </div>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
}
