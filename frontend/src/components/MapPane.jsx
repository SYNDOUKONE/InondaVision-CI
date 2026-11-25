import { useEffect, useRef } from "react";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

export default function MapPane({ layers = [], center = [5.36, -3.98], zoom = 12 }) {
  const mapEl = useRef(null);
  const mapRef = useRef(null);
  const layersRef = useRef([]);

  useEffect(() => {
    if (mapRef.current) return;
    mapRef.current = L.map(mapEl.current, { preferCanvas: true }).setView(center, zoom);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap",
      maxZoom: 19,
    }).addTo(mapRef.current);
  }, []);

  useEffect(() => {
    if (!mapRef.current) return;
    layersRef.current.forEach((l) => mapRef.current.removeLayer(l));
    layersRef.current = layers
      .filter((l) => l.enabled)
      .map((l) => {
        const layer = l.type === "image"
          ? L.imageOverlay(l.url, l.bounds, { opacity: 0.75 })
          : L.geoJSON(l.data, {
              style: l.style,
              onEachFeature: (f, p) =>
                p.bindPopup(
                  `<b>${f.properties.class ?? "feature"}</b>` +
                  (f.properties.name ? `<br>${f.properties.name}` : "")
                ),
            });
        return layer.addTo(mapRef.current);
      });

    const visible = layers.filter((l) => l.enabled);
    if (visible.length > 0) {
      try {
        const bounds = L.latLngBounds(
          visible.map((l) =>
            l.type === "image"
              ? [[l.bounds.minlat, l.bounds.minlon], [l.bounds.maxlat, l.bounds.maxlon]]
              : l.data.bounds ?? l.bounds
          )
        );
        if (bounds.isValid()) mapRef.current.fitBounds(bounds, { padding: [24, 24] });
      } catch { /* pas de géométrie exploitable */ }
    }
  }, [layers]);

  return <div ref={mapEl} className="h-full w-full rounded-lg" />;
}
