/** API base URL — empty string uses same origin (Vite dev proxy). */
export const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export async function parseApiResponse(res) {
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text.slice(0, 200) || res.statusText);
  }
  if (!res.ok) {
    const detail = data.detail ?? data.message ?? res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

const COLD_START_MS = 5000;

/**
 * Fetch with optional cold-start callback (HF Spaces wake-up).
 * @param {string} path — e.g. "/predict"
 * @param {RequestInit} init
 * @param {{ onSlow?: () => void }} opts
 */
export async function apiFetch(path, init = {}, { onSlow } = {}) {
  const url = `${API_BASE}${path}`;
  let slowTimer;
  if (onSlow) {
    slowTimer = setTimeout(onSlow, COLD_START_MS);
  }
  try {
    const res = await fetch(url, init);
    return res;
  } finally {
    if (slowTimer) clearTimeout(slowTimer);
  }
}

/** Ping backend on load to wake a sleeping HF Space (no-op if same-origin proxy). */
export function wakeBackend() {
  apiFetch("/health").catch(() => {});
}
