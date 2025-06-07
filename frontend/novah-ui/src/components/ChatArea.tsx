import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Message } from "../types";
import { motion, AnimatePresence } from "framer-motion";

interface ChatAreaProps {
  messages: Message[];
}

export default function ChatArea({ messages }: ChatAreaProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const [expandedReasoning, setExpandedReasoning] = useState<string | null>(
    null
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toggleReasoning = (uid: string) => {
    setExpandedReasoning(expandedReasoning === uid ? null : uid);
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((m, i) => (
        <div
          key={m.uid || i}
          className={`flex flex-col ${
            m.type === "user" ? "items-end" : "items-start"
          }`}
        >
          <div
            className={`p-3 rounded-xl inline-block max-w-2xl ${
              m.type === "user"
                ? "bg-blue-600 text-white"
                : m.type === "error"
                ? "bg-red-500/20 text-red-300"
                : "bg-gray-800/60 text-gray-200"
            }`}
          >
            <ReactMarkdown>{m.content}</ReactMarkdown>
          </div>
          {m.type === "agent" && m.reasoning && (
            <div className="mt-2 max-w-2xl w-full">
              <button
                onClick={() => toggleReasoning(m.uid!)}
                className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                {expandedReasoning === m.uid
                  ? "Hide Thoughts ▼"
                  : "Show Thoughts ►"}
              </button>
              <AnimatePresence>
                {expandedReasoning === m.uid && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden mt-1 p-2 bg-black/30 border-l-2 border-purple-500 rounded text-xs text-gray-400"
                  >
                    <pre className="whitespace-pre-wrap font-sans">
                      <code>{m.reasoning}</code>
                    </pre>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
