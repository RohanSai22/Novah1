# Novah Frontend Migration

The previous React app located in `frontend/agentic-seek-front` was bootstrapped with CRA and written in JavaScript.  A new UI has been created inside `frontend/novah-ui` using Vite, React 18 and TypeScript.

## Moved / Renamed
- `frontend/novah-ui` – brand new workspace with Vite configuration.
- Component logic from `App.js` was split into `Chat.tsx`, `Home.tsx`, and smaller components such as `PromptInput`, `ChatArea`, `AgentWorkspace`, and `Sidebar`.
- Global CSS has been replaced with Tailwind utility classes defined in `src/index.css`.
- Removed `frontend/agentic-seek-front` and updated Docker configs.

## TODO / Gaps
- Only the basic chat flow and API calls were ported. Some detailed behaviour from the old interface (screenshots view, plan view) still needs refinement.
- Framer Motion animations were added to suggestion cards and headings, but more animations may be required to fully match the design spec.
- Threads are currently stored in a simple state array and not persisted.
