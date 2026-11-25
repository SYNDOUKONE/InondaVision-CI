export default function StatCard({ label, value, sub, tone = "slate" }) {
  const tones = {
    slate: "border-slate-700 bg-slate-800/60",
    flood: "border-red-500/40 bg-red-950/40",
    water: "border-sky-500/40 bg-sky-950/40",
    good: "border-emerald-500/40 bg-emerald-950/40",
    warn: "border-amber-500/40 bg-amber-950/40",
  };
  return (
    <div className={`rounded-xl border p-4 ${tones[tone]}`}>
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-100">{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-400">{sub}</div>}
    </div>
  );
}
