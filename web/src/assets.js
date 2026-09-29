// Public-domain imagery from Wikimedia Commons (US government works).
const commons = (file, width = 1400) =>
  `https://commons.wikimedia.org/wiki/Special:FilePath/${encodeURIComponent(file)}?width=${width}`;

export const IMAGES = {
  jet: commons("F-22 Raptor edit1.jpg"),
  tank: commons("M1A2 Abrams firing.jpg"),
  ship: commons("USS Arleigh Burke (DDG 51) steams through the Mediterranean Sea.jpg"),
};
// Telemetry Archive hover clips (played muted, only while hovered).
export const ARCHIVE_VIDEOS = {
  aerial: "/archive-aerial.mp4",
  ground: "/archive-ground.mp4",
  maritime: "/archive-maritime.mp4",
};

// Hero background loop: clips play back to back, then the playlist restarts.
export const HERO_VIDEOS = ["/hero-loop.mp4"];
export const HERO_POSTER = "/hero-poster.jpg";
export const ASTRA_LOGO = "/astra-logo.jpeg";

export const BACKEND_URL = (process.env.REACT_APP_BACKEND_URL || "").replace(/\/$/, "");
export const API = `${BACKEND_URL}/api`;
