/** Same-origin paths when served behind nginx; Vite dev proxies these. */
export const dashboardApiBase =
  import.meta.env.VITE_DASHBOARD_API_BASE?.replace(/\/$/, "") || "/api/dashboard";
export const biovilApiBase =
  import.meta.env.VITE_BIOVIL_API_BASE?.replace(/\/$/, "") || "/api/biovil";
export const orthancBase = import.meta.env.VITE_ORTHANC_BASE?.replace(/\/$/, "") || "/orthanc";

const TOKEN_KEY = "manthana.authToken";

export function getStoredToken(): string {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function setStoredToken(token: string): void {
  if (token.trim()) localStorage.setItem(TOKEN_KEY, token.trim());
  else localStorage.removeItem(TOKEN_KEY);
}

export function authHeaders(): HeadersInit {
  const t = getStoredToken();
  if (!t) return {};
  return { Authorization: `Bearer ${t}` };
}
