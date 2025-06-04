import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

interface PromptInputProps {
  onSubmit?: (prompt: string) => void;
  disabled?: boolean;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  multiline?: boolean;
}

export default function PromptInput({
  onSubmit,
  disabled = false,
  placeholder = "Ask anything... Try: 'Research the latest AI trends' or 'Help me build a React app'",
  value: externalValue,
  onChange: externalOnChange,
  multiline = false,
}: PromptInputProps) {
  const [internalValue, setInternalValue] = useState("");

  // Use external value if provided, otherwise use internal state
  const value = externalValue !== undefined ? externalValue : internalValue;
  const setValue = externalOnChange || setInternalValue;
  const [isFocused, setIsFocused] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || disabled) return;

    const id = Date.now().toString();

    if (onSubmit) {
      // If we're on the chat page, just submit directly
      onSubmit(value);
      setValue(""); // Clear input after submitting
    } else {
      // If we're on the home page, navigate to chat with the query
      navigate(`/chat/${id}`, {
        state: { initialQuery: value.trim() },
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
          className={`relative flex items-start gap-3 p-6 rounded-3xl transition-all duration-300 ${
            isFocused
              ? "bg-gray-900/60 border border-gray-600/40 shadow-2xl shadow-gray-500/10"
              : "bg-gray-900/40 border border-gray-700/20 hover:bg-gray-900/50"
          }`}
        >
          {multiline ? (
            <textarea
              className={`flex-1 bg-transparent px-4 py-4 text-white text-lg placeholder-gray-500 focus:outline-none resize-none min-h-[120px] max-h-[300px] leading-relaxed ${
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
              className={`flex-1 bg-transparent px-4 py-4 text-white text-lg placeholder-gray-500 focus:outline-none ${
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

          <button
            type="submit"
            className={`p-4 rounded-2xl font-medium transition-all duration-200 flex items-center justify-center self-end ${
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

        {/* Shortcuts hint */}
        {multiline && (
          <div className="flex justify-between items-center mt-3 px-4 text-sm text-gray-600">
            <span>Press Enter to send, Shift+Enter for new line</span>
            <span className="text-gray-700">{value.length} characters</span>
          </div>
        )}
      </form>
    </div>
  );
}
