import { useState } from "react";
import MapPane from "../components/MapPane";
import StatCard from "../components/StatCard";
import { detectScene, fmtKm2 } from "../api";

export default function Detection({ selected, detection, onDetected }) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [showPermanent, setShowPermanent] = useState(true);

  const run = async () => {
    if (!selected) return;
    setRunning(true);
    setError("");
    try {
      const result = await detectScene(selected);
      onDetected(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const d = detection;
  const layers = d ? [
    { type: "geojson", data: d.flood_geojson, enabled: true,
      style: { color: "#ff5050", fillColor: "#ff5050", fillOpacity: 0.5, weight: 1 } },
    { type: "geojson", data: d.permanent_geojson, enabled: showPermanent,
      style: { color: "#0066cc", fillColor: "#0066cc", fillOpacity: 0.5, weight: 1 } },
  ] : [];

  return (
    <div className="grid h-full grid-cols-[320px_1fr] gap-4">
      <div>
        <h2 className="mb-3 text-lg font-semibold">AI Detection</h2>
        <p className="mb-3 text-sm text-slate-400">
          Scène sélectionnée : <span className="text-sky-300">{selected || "aucune"}</span>
        </p>
        <button onClick={run} disabled={!selected || running}
          className="w-full rounded-lg bg-sky-600 py-2 font-medium text-white hover:bg-sky-500 disabled:bg-slate-700 disabled:text-slate-500">
          {running ? "Analyse en cours…" : "Lancer la détection"}
        </button>
        <label className="mt-3 flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={showPermanent} onChange={(e) => setShowPermanent(e.target.checked)} />
          Eau permanente
        </label>
        {error && <p className="mt-3 text-xs text-red-400">{error}</p>}
        {d && (
          <div className="mt-4 space-y-3">
            <div className="rounded-lg bg-slate-800/60 p-3 text-xs text-slate-400">
              méthode : <span className="text-emerald-400">{d.method}</span> ({d.mask_shape[1]}×{d.mask_shape[2]} px)
            </div>
          </div>
        )}
      </div>
      <div className="min-h-[420px] rounded-lg border border-slate-700">
        {d ? (
          <div className="flex h-full flex-col">
            <div className="grid grid-cols-3 gap-3 p-3">
              <StatCard label="Surface inondée" value={fmtKm2(d.areas.flood_water.area_km2)}
                        sub={`${d.areas.flood_water.area_hectares} ha`} tone="flood" />
              <StatCard label="Eau permanente" value={fmtKm2(d.areas.permanent_water.area_km2)}
                        sub={`${d.areas.permanent_water.area_hectares} ha`} tone="water" />
              <StatCard label="Polygones flood" value={d.flood_geojson.features.length}
                        sub={`terre : ${d.areas.background.area_km2} km²`} />
            </div>
            <div className="flex-1 p-3 pt-0">
              <MapPane layers={layers} />
            </div>
          </div>
        ) : (
          <div className="flex h-full items-center justify-center text-slate-500">
            Sélectionnez une scène dans l'Explorer puis lancez la détection
          </div>
        )}
      </div>
    </div>
  );
}
