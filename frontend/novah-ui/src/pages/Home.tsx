import PromptInput from "../components/PromptInput";
import SuggestionCards from "../components/SuggestionCards";
import { motion } from "framer-motion";
import { useState } from "react";

export default function Home() {
  const [inputValue, setInputValue] = useState("");

  const handleSuggestionClick = (suggestionText: string) => {
    setInputValue(suggestionText);
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="text-6xl font-light mb-4 text-white tracking-tight">
            Novah
          </h1>
          <p className="text-gray-400 text-xl font-light">
            Your Advanced AI Agent
          </p>
        </motion.div>

        {/* Chat Input Area - Made Much Larger */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mb-10"
        >
          <PromptInput
            placeholder="Message Novah..."
            value={inputValue}
            onChange={setInputValue}
            multiline={true}
          />
        </motion.div>

        {/* Suggestion Cards - Made Smaller */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <SuggestionCards onSuggestionClick={handleSuggestionClick} />
        </motion.div>
      </div>
    </div>
  );
}
