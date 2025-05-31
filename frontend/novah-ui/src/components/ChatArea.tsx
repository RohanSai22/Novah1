import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Message } from '../types';

interface ChatAreaProps {
  messages: Message[];
}

export default function ChatArea({ messages }: ChatAreaProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((m, i) => (
        <div key={i} className={m.type === 'user' ? 'text-right' : ''}>
          <div className="bg-white/10 p-2 rounded-xl inline-block max-w-lg">
            <ReactMarkdown>{m.content}</ReactMarkdown>
          </div>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
