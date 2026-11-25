import { useEffect, useState } from "react";
import MapPane from "../components/MapPane";
import { getPreviewBlob, getScenarios } from "../api";

export default function Explorer({ scenarios, selected, onSelect, onLoaded }) {
  const [band, setBand] = useState(0);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getScenarios().catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    getPreviewBlob(selected, band)
      .then((p) => {
        setPreview(p);
        onLoaded({ type: "image", url: p.url, bounds: p.bounds, enabled: true, label: "Image radar" });
      })
      .catch((e) => setError(e.message));
  }, [selected, band]);

  return (
    <div className="grid h-full grid-cols-[320px_1fr] gap-4">
      <div className="overflow-y-auto">
        <h2 className="mb-3 text-lg font-semibold">Satellite Explorer</h2>
        <div className="mb-3 flex gap-2 text-sm">
          <button onClick={() => setBand(0)}
                  className={`rounded-md px-3 py-1 ${band === 0 ? "bg-sky-600" : "bg-slate-800 text-slate-300"}`}>
            VV
          </button>
          <button onClick={() => setBand(1)}
                  className={`rounded-md px-3 py-1 ${band === 1 ? "bg-sky-600" : "bg-slate-800 text-slate-300"}`}>
            VH
          </button>
        </div>
        {error && <p className="mb-2 text-xs text-red-400">{error}</p>}
        {scenarios.length === 0 && (
          <p className="text-sm text-slate-400">
            Aucune scène. Lancez <code>python scripts/make_synthetic_dataset.py</code> ou
            déposez des GeoTIFF dans <code>data/processed/&lt;dataset&gt;/s1/</code>.
          </p>
        )}
        <ul className="space-y-2">
          {scenarios.map((s) => (
            <li key={s.scene_id}>
              <button onClick={() => onSelect(s.scene_id)}
                className={`w-full rounded-lg border px-3 py-2 text-left text-sm transition
                  ${selected === s.scene_id
                    ? "border-sky-500 bg-sky-950/60 text-sky-100"
                    : "border-slate-700 bg-slate-800/60 hover:border-slate-500"}`}>
                <div className="font-medium">{s.scene_id}</div>
                <div className="text-xs text-slate-400">
                  {s.event} · {s.post_size_mb} Mo ·{" "}
                  {s.has_pre ? "pré ✓" : "pré ✗"} · {s.has_label ? "labels ✓" : "labels ✗"}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div className="min-h-[420px] rounded-lg border border-slate-700">
        <MapPane layers={preview ? [{ type: "image", url: preview.url, bounds: preview.bounds, enabled: true }] : []} />
      </div>
    </div>
  );
}
