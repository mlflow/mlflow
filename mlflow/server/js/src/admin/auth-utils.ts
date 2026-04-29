/**
 * Drop the browser's HTTP Basic Auth credential cache, expire app cookies,
 * clear the React Query cache, then redirect to the app root.
 *
 * Basic Auth has no server-side session, so the only way to "log out" is to
 * make the browser forget the credentials it has cached for the realm. The
 * canonical trick is an XHR with deliberately wrong user/password — the
 * ``user``/``password`` arguments to ``xhr.open()`` *replace* the cached
 * credentials for the realm, and the resulting 401 confirms they're bogus.
 *
 * Why XHR and not ``fetch()``: a manually-set ``Authorization`` header on
 * ``fetch()`` is treated by browsers as a one-off override that does not
 * update the credential cache. Only the ``xhr.open(url, async, user, pass)``
 * signature actually overwrites the realm cache.
 *
 * Cookie expiry: cookies set under a static prefix (e.g. ``/mlflow/``) carry
 * that path, so a single ``path=/`` deletion misses them. Clear at root, the
 * current app's base path, AND the same base path with a trailing slash
 * stripped — the browser's path-match for deletion is exact, not a prefix.
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

  // Per-load nonce so the bogus creds can't accidentally collide with a real
  // user's credentials (in which case the XHR could succeed and the browser
  // would keep the cached creds, defeating logout).
  const nonce = Date.now().toString(36) + Math.random().toString(36).slice(2);
  const bogus = `mlflow-logged-out-${nonce}`;
  const usersCurrentUrl = new URL('ajax-api/2.0/mlflow/users/current', window.location.href).toString();
  const homeUrl = new URL('.', window.location.href).toString();

  const goHome = () => window.location.assign(homeUrl);
  try {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', usersCurrentUrl, true, bogus, bogus);
    // Wait for the cache write to complete before navigating; otherwise the
    // home-page request could still be sent with the (still-cached) real
    // creds and silently auto-auth.
    xhr.onloadend = goHome;
    xhr.send();
  } catch {
    goHome();
  }
};
