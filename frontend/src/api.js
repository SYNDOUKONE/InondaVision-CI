export async function getHealth() {
  const r = await fetch("/api/v1/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function getScenarios() {
  const r = await fetch("/api/v1/scenarios");
  if (!r.ok) throw new Error(`scenarios ${r.status}`);
  return r.json();
}

export async function getPreviewBlob(sceneId, band) {
  const r = await fetch(`/api/v1/scenes/${sceneId}/preview.png?band=${band}`);
  if (!r.ok) throw new Error(`preview ${r.status}`);
  const bounds = JSON.parse(r.headers.get("X-Scene-Bounds"));
  const blob = await r.blob();
  return { url: URL.createObjectURL(blob), bounds };
}

export async function detectScene(sceneId, minAreaM2 = 500) {
  const r = await fetch(
    `/api/v1/detect?scene_id=${encodeURIComponent(sceneId)}&min_area_m2=${minAreaM2}`,
    { method: "POST" }
  );
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `detect ${r.status}`);
  return r.json();
}

export function fmtKm2(v) {
  return `${v.toLocaleString("fr-FR", { maximumFractionDigits: 2 })} km²`;
}
