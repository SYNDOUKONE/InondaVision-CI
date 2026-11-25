import { useEffect, useState } from "react";
import { getHealth, getScenarios } from "./api";
import Explorer from "./views/Explorer";
import Detection from "./views/Detection";
import FloodMap from "./views/FloodMap";
import Analytics from "./views/Analytics";

const TABS = [
  ["explorer", "🛰 Satellite Explorer"],
  ["detection", "🤖 AI Detection"],
  ["map", "🌊 Flood Map"],
  ["analytics", "📊 Analytics"],
];

export default function App() {
  const [tab, setTab] = useState("explorer");
  const [scenarios, setScenarios] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detection, setDetection] = useState(null);
  const [explorerLayer, setExplorerLayer] = useState([]);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getScenarios().then(setScenarios).catch(() => {});
    getHealth().then(setHealth).catch(() => setHealth({ status: "unreachable", model_ready: false }));
  }, []);

  return (
    <div className="flex h-screen flex-col bg-slate-900 text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-700 bg-slate-800/80 px-5 py-3">
        <div>
          <h1 className="text-lg font-bold tracking-tight text-sky-400">InondaVision CI</h1>
          <p className="text-xs text-slate-400">
            Détection d'inondations à Abidjan — Sentinel-1 VV/VH
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`rounded-full px-3 py-1 ${health?.model_ready
            ? "bg-emerald-950/60 text-emerald-300"
            : "bg-amber-950/60 text-amber-300"}`}>
            {health?.model_ready ? "modèle chargé" : "baseline (sans modèle)"}
          </span>
        </div>
      </header>

      <nav className="flex gap-1 border-b border-slate-700 bg-slate-800/40 px-4">
        {TABS.map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)}
            className={`rounded-t-lg px-4 py-2 text-sm font-medium transition
              ${tab === id ? "bg-slate-900 text-sky-400" : "text-slate-400 hover:text-slate-200"}`}>
            {label}
          </button>
        ))}
      </nav>

      <main className="min-h-0 flex-1 p-4">
        {tab === "explorer" && (
          <Explorer scenarios={scenarios} selected={selected}
                    onSelect={setSelected} onLoaded={setExplorerLayer} />
        )}
        {tab === "detection" && (
          <Detection selected={selected} detection={detection} onDetected={setDetection} />
        )}
        {tab === "map" && <FloodMap detection={detection} />}
        {tab === "analytics" && <Analytics detection={detection} />}
      </main>
    </div>
  );
}
