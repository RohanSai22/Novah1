import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight, Maximize2, Minimize2 } from "lucide-react";

interface Props {
  isOpen: boolean;
  onToggle: () => void;
  width: number;
  onWidthChange: (width: number) => void;
  children: React.ReactNode;
  title?: string;
  minWidth?: number;
  maxWidth?: number;
}

const PRESET_WIDTHS = [
  { label: "Narrow", value: 25, icon: "▎" },
  { label: "Normal", value: 40, icon: "▍" },
  { label: "Wide", value: 60, icon: "▌" },
  { label: "Full", value: 80, icon: "█" },
];

export default function WorkspaceSlider({
  isOpen,
  onToggle,
  width,
  onWidthChange,
  children,
  title = "Workspace",
  minWidth = 20,
  maxWidth = 80,
}: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [showPresets, setShowPresets] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;

      const newWidth = Math.min(
        maxWidth,
        Math.max(
          minWidth,
          ((window.innerWidth - e.clientX) / window.innerWidth) * 100
        )
      );
      onWidthChange(newWidth);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setIsResizing(false);
    };

    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, minWidth, maxWidth, onWidthChange]);

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    setIsResizing(true);
  };

  return (
    <>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 h-full bg-black/95 backdrop-blur-xl border-l border-white/10 z-40 flex flex-col"
            style={{ width: `${width}%` }}
          >
            {/* Resize Handle */}
            <div
              className={`absolute left-0 top-0 w-1 h-full cursor-col-resize bg-gradient-to-b from-purple-500/50 to-blue-500/50 hover:bg-gradient-to-b hover:from-purple-400/70 hover:to-blue-400/70 transition-all duration-200 ${
                isResizing ? "bg-gradient-to-b from-purple-400 to-blue-400" : ""
              }`}
              onMouseDown={handleResizeStart}
            >
              <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-3 h-12 bg-gray-700 rounded-r-md flex items-center justify-center">
                <div className="w-0.5 h-6 bg-gray-500 rounded"></div>
              </div>
            </div>

            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/10">
              <div className="flex items-center gap-3">
                <span className="text-xl">🛠️</span>
                <h2 className="text-lg font-semibold text-white">{title}</h2>
                <div className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">
                  {width.toFixed(0)}%
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* Width Presets Button */}
                <div className="relative">
                  <button
                    onClick={() => setShowPresets(!showPresets)}
                    className="p-1.5 hover:bg-white/10 rounded-md transition-colors text-gray-400 hover:text-white"
                    title="Width presets"
                  >
                    <Maximize2 size={16} />
                  </button>

                  {/* Presets Dropdown */}
                  <AnimatePresence>
                    {showPresets && (
                      <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute top-full right-0 mt-1 bg-gray-900/95 backdrop-blur-xl border border-gray-700/50 rounded-lg shadow-xl z-50"
                      >
                        <div className="p-2 min-w-[120px]">
                          {PRESET_WIDTHS.map((preset) => (
                            <button
                              key={preset.value}
                              onClick={() => {
                                onWidthChange(preset.value);
                                setShowPresets(false);
                              }}
                              className={`w-full flex items-center gap-2 px-3 py-2 rounded-md transition-colors text-sm ${
                                Math.abs(width - preset.value) < 2
                                  ? "bg-blue-600/20 text-blue-400"
                                  : "hover:bg-white/10 text-gray-300"
                              }`}
                            >
                              <span>{preset.icon}</span>
                              <span>{preset.label}</span>
                              <span className="text-xs text-gray-500 ml-auto">
                                {preset.value}%
                              </span>
                            </button>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Close Button */}
                <button
                  onClick={onToggle}
                  className="p-1.5 hover:bg-white/10 rounded-md transition-colors text-gray-400 hover:text-white"
                  title="Close workspace"
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden">{children}</div>

            {/* Footer with resize info */}
            {isResizing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute bottom-4 left-4 bg-black/80 text-white text-xs px-2 py-1 rounded"
              >
                Width: {width.toFixed(0)}%
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Button (when closed) */}
      {!isOpen && (
        <motion.button
          initial={{ x: 50 }}
          animate={{ x: 0 }}
          onClick={onToggle}
          className="fixed right-4 top-1/2 transform -translate-y-1/2 bg-gradient-to-b from-purple-600/80 to-blue-600/80 backdrop-blur-xl border border-white/20 p-3 rounded-l-xl shadow-xl z-30 hover:from-purple-600 hover:to-blue-600 transition-all duration-200"
          title="Open workspace"
        >
          <ChevronLeft size={20} className="text-white" />
        </motion.button>
      )}

      {/* Backdrop to close presets */}
      {showPresets && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowPresets(false)}
        />
      )}
    </>
  );
}
