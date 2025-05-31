import { motion } from 'framer-motion';

const suggestions = [
  { icon: '📄', label: 'Docs' },
  { icon: '🌐', label: 'Web' },
  { icon: '💻', label: 'Code' },
];

export default function SuggestionCards() {
  return (
    <div className="flex gap-4 mt-6 justify-center">
      {suggestions.map(s => (
        <motion.div
          key={s.label}
          whileHover={{ scale: 1.05 }}
          className="bg-white/10 rounded-xl p-4 backdrop-blur cursor-pointer flex flex-col items-center w-20"
        >
          <span className="text-2xl">{s.icon}</span>
          <span className="mt-1 text-sm">{s.label}</span>
        </motion.div>
      ))}
    </div>
  );
}
