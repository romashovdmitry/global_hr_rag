import React, { useEffect, useState } from "react";
import { Chat } from "./components/Chat";
import { CVUpload } from "./components/CVUpload";
import { clearMessages, getCVStatus } from "./api/client";
import StatsPage from "./pages/StatsPage";

const isStats = window.location.pathname === "/stats";

export default function App() {
  const [cvLoaded, setCvLoaded] = useState(false);

  // Restore CV status from backend on every page load.
  useEffect(() => {
    getCVStatus().then(setCvLoaded).catch(() => setCvLoaded(false));
  }, []);

  async function handleClear() {
    await clearMessages();
    setCvLoaded(false);
    window.location.reload();
  }

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">IT Vacancy RAG</h1>
        <nav className="app__nav">
          <a
            href="/"
            className={`app__nav-link${!isStats ? " app__nav-link--active" : ""}`}
          >
            Chat
          </a>
          <a
            href="/stats"
            className={`app__nav-link${isStats ? " app__nav-link--active" : ""}`}
          >
            Stats
          </a>
        </nav>
        {!isStats && (
          <div className="app__actions">
            <CVUpload
              cvLoaded={cvLoaded}
              onUploaded={() => setCvLoaded(true)}
              onCleared={() => setCvLoaded(false)}
            />
            {cvLoaded && (
              <span className="badge badge--green">CV attached</span>
            )}
            <button className="btn btn--ghost" onClick={handleClear}>
              Clear history
            </button>
          </div>
        )}
      </header>

      <main className={`app__main${isStats ? " app__main--scrollable" : ""}`}>
        {isStats ? <StatsPage /> : <Chat />}
      </main>
    </div>
  );
}
