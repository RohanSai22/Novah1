import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

interface SuggestionCard {
  icon: string;
  label: string;
  example: string;
  color: string;
}

const suggestions: SuggestionCard[] = [
  {
    icon: "🔍",
    label: "Research",
    example: "Research the latest AI developments",
    color: "bg-blue-500/10 hover:bg-blue-500/20 border-blue-500/20",
  },
  {
    icon: "💡",
    label: "Explain",
    example: "Explain quantum computing in simple terms",
    color: "bg-purple-500/10 hover:bg-purple-500/20 border-purple-500/20",
  },
  {
    icon: "⚡",
    label: "Create",
    example: "Create a business plan for a tech startup",
    color: "bg-green-500/10 hover:bg-green-500/20 border-green-500/20",
  },
];

interface Props {
  onSuggestionClick?: (example: string) => void;
}

export default function SuggestionCards({ onSuggestionClick }: Props) {
  const navigate = useNavigate();

  const handleCardClick = (suggestion: SuggestionCard) => {
    if (onSuggestionClick) {
      onSuggestionClick(suggestion.example);
    } else {
      // Navigate to chat with pre-filled query
      const threadId = Date.now().toString();
      navigate(`/chat/${threadId}`, {
        state: { initialQuery: suggestion.example },
      });
    }
  };

  return (
    <div className="flex flex-wrap justify-center gap-3 max-w-2xl mx-auto">
      {suggestions.map((suggestion, index) => (
        <motion.button
          key={suggestion.label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          whileHover={{ scale: 1.02, y: -2 }}
          whileTap={{ scale: 0.98 }}
          className={`${suggestion.color} rounded-2xl px-6 py-4 border transition-all duration-200 cursor-pointer group flex items-center gap-3 hover:shadow-lg`}
          onClick={() => handleCardClick(suggestion)}
        >
          <span className="text-2xl">{suggestion.icon}</span>
          <div className="text-left">
            <h3 className="text-white font-medium text-sm mb-1">
              {suggestion.label}
            </h3>
            <p className="text-gray-400 text-xs leading-tight">
              {suggestion.example}
            </p>
          </div>
        </motion.button>
      ))}
    </div>
  );
}
