/**
 * Clear the browser's Basic-Auth credential cache and React Query cache,
 * then navigate to the backend's /logout page (which forces the browser
 * to drop its Authorization header for subsequent requests).
 *
 * Cookies set under a static prefix (e.g. ``/mlflow/``) carry that path,
 * so a single ``path=/`` deletion misses them. Clear at root, the current
 * app's base path (which usually ends in ``/``), AND the same base path
 * with a trailing slash stripped — cookies are often scoped to
 * ``Path=/mlflow`` rather than ``Path=/mlflow/``, and the browser's
 * path-match for deletion is exact, not a prefix.
 *
 * Resolving 'logout' relative to ``window.location.href`` preserves any
 * static prefix (the backend registers /logout via ``_add_static_prefix``)
 * so deployments served under a sub-path continue to work.
 */
export const performLogout = (queryClient?: { clear: () => void }) => {
  const candidatePaths = new Set<string>(['/']);
  const basePath = new URL('.', window.location.href).pathname;
  if (basePath) {
    candidatePaths.add(basePath);
    const normalizedBasePath = basePath.replace(/\/$/, '');
    if (normalizedBasePath) {
      candidatePaths.add(normalizedBasePath);
    }
  }
  const expiresAttr = 'expires=Thu, 01 Jan 1970 00:00:00 UTC';
  for (const name of ['mlflow_user', 'mlflow-request-header-Authorization']) {
    for (const path of candidatePaths) {
      document.cookie = `${name}=; ${expiresAttr}; path=${path};`;
    }
  }

  queryClient?.clear();

  window.location.href = new URL('logout', window.location.href).toString();
};
