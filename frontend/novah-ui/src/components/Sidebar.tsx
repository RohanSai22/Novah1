import { Link, useParams } from 'react-router-dom';
import { Thread } from '../types';

interface Props {
  threads: Thread[];
}

export default function Sidebar({ threads }: Props) {
  const { threadId } = useParams();
  return (
    <aside className="w-60 border-r border-white/20 p-4 space-y-2">
      <Link to="/" className="block mb-4 text-accent">New Chat</Link>
      {threads.map(t => (
        <Link
          key={t.id}
          to={`/chat/${t.id}`}
          className={`block px-2 py-1 rounded hover:bg-white/10 ${t.id === threadId ? 'bg-white/10' : ''}`}
        >
          {t.title || 'Untitled'}
        </Link>
      ))}
    </aside>
  );
}
