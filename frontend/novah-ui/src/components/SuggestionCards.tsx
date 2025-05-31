import { motion } from 'framer-motion';

const suggestions = ['Find docs', 'Search the web', 'Write code'];

export default function SuggestionCards() {
  return (
    <div className="flex gap-4 mt-6 justify-center">
      {suggestions.map(text => (
        <motion.div
          key={text}
          whileHover={{ scale: 1.05 }}
          className="bg-white/10 rounded-xl px-4 py-2 backdrop-blur cursor-pointer"
        >
          {text}
        </motion.div>
      ))}
    </div>
  );
}
