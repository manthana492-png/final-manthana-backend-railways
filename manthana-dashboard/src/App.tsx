import { useEffect, useState } from "react";
import { Link, Route, Routes } from "react-router-dom";
import { getStoredToken, setStoredToken } from "./lib/config";
import Worklist from "./pages/Worklist";
import StudyDetail from "./pages/StudyDetail";

export default function App() {
  const [token, setToken] = useState(getStoredToken);

  useEffect(() => {
    setStoredToken(token);
  }, [token]);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-ink-800/80 bg-ink-950/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-3">
          <Link to="/" className="text-lg font-bold tracking-tight text-white">
            Manthana<span className="text-accent-glow">.</span>
          </Link>
          <span className="hidden text-xs text-slate-500 sm:inline">
            Radiologist dashboard · Cornerstone · BioViL
          </span>
          <div className="ml-auto flex min-w-[200px] flex-1 items-center gap-2 sm:max-w-md">
            <label htmlFor="api-token" className="sr-only">
              API token
            </label>
            <input
              id="api-token"
              type="password"
              autoComplete="off"
              placeholder="Dashboard API Bearer token (optional)"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              className="w-full rounded-lg border border-ink-700 bg-ink-900 px-3 py-1.5 font-mono text-xs text-white placeholder:text-slate-600 focus:border-accent focus:outline-none"
            />
          </div>
        </div>
      </header>

      <Routes>
        <Route path="/" element={<Worklist />} />
        <Route path="/study/:id" element={<StudyDetail />} />
      </Routes>
    </div>
  );
}
