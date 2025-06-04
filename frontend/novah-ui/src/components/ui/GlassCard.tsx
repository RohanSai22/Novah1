// Glass Card component with dark mode styling
import { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  footer?: ReactNode;
  loading?: boolean;
  fadeIn?: boolean;
}

export default function GlassCard({
  children,
  className = "",
  title,
  footer,
  loading = false,
  fadeIn = true,
}: GlassCardProps) {
  return (
    <div
      className={`
        bg-gradient-to-br from-gray-900/80 to-black/90
        backdrop-blur-md border border-white/10
        rounded-xl shadow-xl overflow-hidden
        transition-all duration-300 ease-in-out
        ${fadeIn ? "animate-fadeIn" : ""}
        ${loading ? "opacity-70" : "opacity-100"}
        ${className}
      `}
    >
      {title && (
        <div className="border-b border-white/10 px-4 py-3">
          <h3 className="text-lg font-medium text-white/90">{title}</h3>
        </div>
      )}

      <div className="p-4">
        {loading ? (
          <div className="flex items-center justify-center py-6">
            <div className="animate-spin h-8 w-8 border-3 border-blue-500 border-t-transparent rounded-full"></div>
          </div>
        ) : (
          children
        )}
      </div>

      {footer && (
        <div className="border-t border-white/10 px-4 py-3">{footer}</div>
      )}
    </div>
  );
}
