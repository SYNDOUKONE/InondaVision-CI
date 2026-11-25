import { useState } from "react";
import MapPane from "../components/MapPane";
import { fmtKm2 } from "../api";

const COLORS = {
  flood_water: { c: "#ff5050", label: "Eau d'inondation" },
  permanent_water: { c: "#0066cc", label: "Eau permanente" },
};

export default function FloodMap({ detection }) {
  const [enabled, setEnabled] = useState({ flood_water: true, permanent_water: true });

  if (!detection) {
    return (
      <div className="flex h-full items-center justify-center text-slate-500">
        Lancez d'abord une détection (onglet AI Detection)
      </div>
    );
  }

  const layers = Object.keys(COLORS).map((cls) => ({
    type: "geojson",
    data: cls === "flood_water" ? detection.flood_geojson : detection.permanent_geojson,
    enabled: enabled[cls],
    style: { color: COLORS[cls].c, fillColor: COLORS[cls].c, fillOpacity: 0.5, weight: 1 },
  }));

  return (
    <div className="grid h-full grid-cols-[260px_1fr] gap-4">
      <div>
        <h2 className="mb-3 text-lg font-semibold">Flood Map</h2>
        {Object.entries(COLORS).map(([cls, c]) => (
          <label key={cls} className="mb-2 flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={enabled[cls]}
                   onChange={(e) => setEnabled({ ...enabled, [cls]: e.target.checked })} />
            <span className="inline-block h-3 w-3 rounded-sm"
                  style={{ backgroundColor: c.c }} />
            {c.label}
          </label>
        ))}
        <div className="mt-4 rounded-lg bg-amber-950/40 border border-amber-600/30 p-3 text-xs text-amber-200">
          Superposition satellitaire = exposition estimée. Ne constitue ni un
          rapport de dégâts ni une évaluation d'accessibilité routière.
        </div>
        <a
          className="mt-4 inline-block text-xs text-sky-400 hover:underline"
          download="inondavision-flood.geojson"
          href={URL.createObjectURL(
            new Blob([JSON.stringify(detection.flood_geojson)], { type: "application/geo+json" })
          )}
        >
          Télécharger les polygones flood (GeoJSON)
        </a>
        <p className="mt-4 text-xs text-slate-500">Système de coordonnées : {detection.crs}</p>
      </div>
      <div className="rounded-lg border border-slate-700">
        <MapPane layers={layers} />
      </div>
    </div>
  );
}
