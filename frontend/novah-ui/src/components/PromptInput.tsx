import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface PromptInputProps {
  onSubmit?: (prompt: string, threadId: string) => void;
}

export default function PromptInput({ onSubmit }: PromptInputProps) {
  const [value, setValue] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim()) return;
    const id = Date.now().toString();
    navigate(`/chat/${id}`);
    onSubmit?.(value, id);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl mx-auto flex gap-2">
      <input
        className="flex-1 rounded-xl bg-white/10 backdrop-blur px-4 py-2 focus:outline-none"
        placeholder="Ask something..."
        value={value}
        onChange={e => setValue(e.target.value)}
      />
      <button className="px-4 py-2 rounded-xl bg-accent hover:bg-accent/80">
        Go
      </button>
    </form>
  );
}
