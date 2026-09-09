const apiBaseUrl = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

export function apiUrl(path) {
  return `${apiBaseUrl}${path.startsWith('/') ? path : `/${path}`}`;
}

export async function apiFetch(path, options) {
  return fetch(apiUrl(path), options);
}