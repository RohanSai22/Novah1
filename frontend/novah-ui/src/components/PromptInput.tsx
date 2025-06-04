import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

interface PromptInputProps {
  onSubmit?: (prompt: string, options?: any) => void;
  disabled?: boolean;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  multiline?: boolean;
  showDeepSearchToggle?: boolean;
}

export default function PromptInput({
  onSubmit,
  disabled = false,
  placeholder = "Ask anything... Try: 'Research the latest AI trends' or 'Help me build a React app'",
  value: externalValue,
  onChange: externalOnChange,
  multiline = false,
  showDeepSearchToggle = false,
}: PromptInputProps) {
  const [internalValue, setInternalValue] = useState("");
  const [isDeepSearch, setIsDeepSearch] = useState(false);

  // Use external value if provided, otherwise use internal state
  const value = externalValue !== undefined ? externalValue : internalValue;
  const setValue = externalOnChange || setInternalValue;
  const [isFocused, setIsFocused] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || disabled) return;

    const id = Date.now().toString();
    const searchOptions = showDeepSearchToggle
      ? { deepSearch: isDeepSearch }
      : {};

    if (onSubmit) {
      // If we're on the chat page, just submit directly
      onSubmit(value, searchOptions);
      setValue(""); // Clear input after submitting
    } else {
      // If we're on the home page, navigate to chat with the query
      navigate(`/chat/${id}`, {
        state: { initialQuery: value.trim(), deepSearch: isDeepSearch },
      });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <div
          className={`relative p-6 rounded-3xl transition-all duration-300 ${
            isFocused
              ? "bg-gray-900/60 border border-gray-600/40 shadow-2xl shadow-gray-500/10"
              : "bg-gray-900/40 border border-gray-700/20 hover:bg-gray-900/50"
          }`}
        >
          {/* Text input area - full width */}
          {multiline ? (
            <textarea
              className={`w-full bg-transparent px-4 py-4 text-white text-lg placeholder-gray-500 focus:outline-none resize-none min-h-[120px] max-h-[300px] leading-relaxed overflow-y-auto ${
                disabled ? "opacity-50 cursor-not-allowed" : ""
              }`}
              placeholder={
                disabled ? "Processing your request..." : placeholder
              }
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              onKeyDown={handleKeyDown}
              disabled={disabled}
              rows={4}
            />
          ) : (
            <input
              className={`w-full bg-transparent px-4 py-4 text-white text-lg placeholder-gray-500 focus:outline-none ${
                disabled ? "opacity-50 cursor-not-allowed" : ""
              }`}
              placeholder={
                disabled ? "Processing your request..." : placeholder
              }
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              onKeyDown={handleKeyDown}
              disabled={disabled}
            />
          )}

          {/* Right-aligned controls */}
          <div className="flex items-center justify-end gap-3 mt-4">
            {/* Deep Search Toggle - Only show on home screen */}
            {showDeepSearchToggle && (
              <button
                type="button"
                onClick={() => setIsDeepSearch(!isDeepSearch)}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
                  isDeepSearch
                    ? "bg-gradient-to-r from-purple-600/20 to-blue-600/20 border border-purple-500/30 text-purple-300"
                    : "bg-gray-700/40 border border-gray-600/20 text-gray-400 hover:bg-gray-600/40"
                }`}
              >
                <span className="text-lg">🧠</span>
                <span>{isDeepSearch ? "Deep Search" : "Normal Search"}</span>
              </button>
            )}

            <button
              type="submit"
              className={`p-4 rounded-2xl font-medium transition-all duration-200 flex items-center justify-center ${
                disabled
                  ? "opacity-50 cursor-not-allowed bg-gray-700"
                  : value.trim()
                  ? "bg-white text-black hover:bg-gray-100 shadow-lg hover:shadow-white/20 scale-100 hover:scale-105"
                  : "bg-gray-700 text-gray-500 cursor-not-allowed"
              }`}
              disabled={disabled || !value.trim()}
            >
              {disabled ? (
                <div className="animate-spin h-5 w-5 border-2 border-gray-500 border-t-transparent rounded-full"></div>
              ) : (
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Shortcuts hint */}
        {multiline && (
          <div className="flex justify-between items-center mt-3 px-4 text-sm text-gray-600">
            <span>Press Enter to send, Shift+Enter for new line</span>
            <div className="flex items-center gap-4">
              {showDeepSearchToggle && isDeepSearch && (
                <span className="text-purple-400">
                  Deep Search → Perfect Report
                </span>
              )}
              <span className="text-gray-700">{value.length} characters</span>
            </div>
          </div>
        )}
      </form>
    </div>
  );
}
