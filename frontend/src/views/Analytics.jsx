import StatCard from "../components/StatCard";
import { fmtKm2 } from "../api";

export default function Analytics({ detection }) {
  if (!detection) {
    return (
      <div className="flex h-full items-center justify-center text-slate-500">
        Aucune analyse disponible — lancez une détection
      </div>
    );
  }

  const post = detection.areas.flood_water;
  const pre = detection.areas_pre.flood_water;
  const delta = post.area_km2 - pre.area_km2;
  const deltaPct = pre.area_km2 > 0 ? (delta / pre.area_km2) * 100 : null;

  const rows = [
    { name: "Terre (background)", cls: detection.areas.background, color: "#94a3b8" },
    { name: "Eau permanente", cls: detection.areas.permanent_water, color: "#0066cc" },
    { name: "Eau d'inondation", cls: post, color: "#ff5050" },
  ];
  const total = rows.reduce((acc, r) => acc + r.cls.area_km2, 0);

  return (
    <div className="grid h-full grid-cols-[1fr_360px] gap-6 overflow-y-auto">
      <div>
        <h2 className="mb-4 text-lg font-semibold">Analytics</h2>
        <div className="mb-4 grid grid-cols-3 gap-3">
          <StatCard label="Flood avant" value={fmtKm2(pre.area_km2)} sub="acquisition pré-événement" />
          <StatCard label="Flood après" value={fmtKm2(post.area_km2)} sub="acquisition post-événement" tone="flood" />
          <StatCard label="Évolution" value={`${delta >= 0 ? "+" : ""}${delta.toFixed(2)} km²`}
                    sub={deltaPct !== null ? `${deltaPct >= 0 ? "+" : ""}${deltaPct.toFixed(1)} % vs pré` : "n/a"}
                    tone={delta > 0 ? "warn" : "good"} />
        </div>

        <h3 className="mb-2 text-sm font-medium text-slate-300">
          Répartition des classes — post-événement
        </h3>
        <div className="space-y-3 rounded-xl border border-slate-700 bg-slate-800/40 p-4">
          {rows.map((r) => {
            const pct = total > 0 ? (r.cls.area_km2 / total) * 100 : 0;
            return (
              <div key={r.name}>
                <div className="mb-1 flex justify-between text-xs text-slate-400">
                  <span>{r.name}</span>
                  <span>{r.cls.area_km2} km² · {pct.toFixed(1)} %</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-700">
                  <div className="h-full rounded-full"
                       style={{ width: `${pct}%`, backgroundColor: r.color }} />
                </div>
              </div>
            );
          })}
        </div>

        <table className="mt-4 w-full text-left text-xs text-slate-400">
          <thead className="text-slate-500 uppercase">
            <tr>
              <th className="py-2">Classe</th>
              <th>Pixels</th>
              <th>km²</th>
              <th>hectares</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className="border-t border-slate-800">
                <td className="py-2">{r.name}</td>
                <td className="text-slate-300">{r.cls.pixels.toLocaleString("fr-FR")}</td>
                <td className="text-slate-300">{r.cls.area_km2}</td>
                <td className="text-slate-300">{r.cls.area_hectares}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="space-y-3 text-xs text-slate-400">
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3">
          <div className="mb-1 font-medium text-slate-300">Méthode</div>
          {detection.method} sur {detection.mask_shape[1]}×{detection.mask_shape[2]} px
          (résolution 10 m/pixel)
        </div>
        <div className="rounded-lg border border-amber-600/30 bg-amber-950/40 p-3 text-amber-200">
          <div className="mb-1 font-medium">Réserve d'interprétation</div>
          Les superficies reposent sur des pixels de 10 m et restent des
          estimations. Un même masque doit être validé par des observations
          terrain (rapports ONPC) avant usage décisionnel.
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3">
          <div className="mb-1 font-medium text-slate-300">Pour aller plus loin</div>
          <ul className="list-inside list-disc space-y-1">
            <li><code>POST /api/v1/change</code> : nouveau / retiré / persistant entre deux dates</li>
            <li><code>POST /api/v1/exposure?osm=true</code> : bâtiments &amp; routes exposés (OpenStreetMap)</li>
            <li><code>evaluation/metrics.json</code> : IoU, Dice des modèles</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
