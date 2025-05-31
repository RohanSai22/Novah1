import PromptInput from '../components/PromptInput';
import SuggestionCards from '../components/SuggestionCards';
import { motion } from 'framer-motion';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center text-center p-4">
      <motion.h1 initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="text-4xl font-bold mb-8">
        Novah
      </motion.h1>
      <PromptInput />
      <SuggestionCards />
    </div>
  );
}
