export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8100";

/** Throws the backend's `detail` message rather than a bare status code. */
export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body — keep the status message */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}
