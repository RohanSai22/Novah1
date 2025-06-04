import { Link, useParams } from "react-router-dom";
import { Thread } from "../types";

interface Props {
  threads: Thread[];
  open?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ threads, open = false, onClose }: Props) {
  const { threadId } = useParams();

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 lg:hidden"
        onClick={onClose}
      />

      {/* Sidebar */}
      <aside className="fixed lg:relative left-0 top-0 h-full w-60 border-r border-white/20 p-4 space-y-2 bg-black z-50 transform transition-transform lg:translate-x-0">
        <div className="flex items-center justify-between mb-4">
          <Link to="/" className="text-accent font-medium">
            New Chat
          </Link>
          <button
            onClick={onClose}
            className="lg:hidden text-gray-400 hover:text-white"
          >
            ✕
          </button>
        </div>

        <div className="space-y-1">
          {threads.map((t) => (
            <Link
              key={t.id}
              to={`/chat/${t.id}`}
              className={`block px-2 py-1 rounded hover:bg-white/10 transition-colors ${
                t.id === threadId ? "bg-white/10" : ""
              }`}
              onClick={onClose}
            >
              {t.title || "Untitled Chat"}
            </Link>
          ))}
        </div>
      </aside>
    </>
  );
}
