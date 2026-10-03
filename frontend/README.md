# Frostline dashboard

React, TypeScript, Vite, Recharts and Lucide. All displayed measurements come from the Python research API. The legacy `mocks/` files are not used.

`LiveMission.tsx` is the default screen. It starts a backend session and subscribes to `/api/live/sessions/{id}/events`; the server owns sensor progression, tool execution, decisions, candidate testing and scores. Controls change the running session. The browser retains only the session ID for reload recovery. `main.tsx` also contains the recorded replays, improvement comparisons, real-data pilot and research notes.

`npm ci` then `npm run dev` starts port 5173 and proxies `/api` and `/stream` to 127.0.0.1:8000. `npm run build` runs TypeScript checks and produces `dist/`, served by FastAPI at `/` after backend startup. Fonts are bundled for offline demos.

See the root README for the complete setup, methodology and limitations.
