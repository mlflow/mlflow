export const AUTH_HEADER_COOKIE = 'mlflow-request-header-Authorization';
export const MLFLOW_USER_COOKIE = 'mlflow_user';

const AUTH_COOKIE_NAMES = [MLFLOW_USER_COOKIE, AUTH_HEADER_COOKIE];

/**
 * Cookie deletion does an exact path match, so cover root + the app's
 * base path (with and without trailing slash) - otherwise cookies set
 * under a static prefix like ``/mlflow/`` survive a ``path=/`` delete.
 */
export const getAuthCookiePaths = (): string[] => {
  const paths = new Set<string>(['/']);
  const basePath = new URL('.', window.location.href).pathname;
  if (basePath) {
    paths.add(basePath);
    const stripped = basePath.replace(/\/$/, '');
    if (stripped) paths.add(stripped);
  }
  return Array.from(paths);
};

export const clearAuthCookies = () => {
  const expiresAttr = 'expires=Thu, 01 Jan 1970 00:00:00 UTC';
  for (const name of AUTH_COOKIE_NAMES) {
    for (const path of getAuthCookiePaths()) {
      document.cookie = `${name}=; ${expiresAttr}; path=${path};`;
    }
  }
};

/**
 * Basic Auth has no server-side session - logging out means making the
 * browser forget its cached realm creds. ``xhr.open(url, async, user,
 * pass)`` is the only API that overwrites them; ``fetch()`` with an
 * ``Authorization`` header is treated as a one-off override and leaves
 * the cache intact.
 */
export const performLogout = (queryClient?: { clear: () => void }) => {
  clearAuthCookies();
  queryClient?.clear();

  // Per-load nonce - if the bogus creds collide with real ones, the XHR
  // would succeed and the browser would keep the cached creds.
  const nonce = Date.now().toString(36) + Math.random().toString(36).slice(2);
  const bogus = `mlflow-logged-out-${nonce}`;
  const usersCurrentUrl = new URL('ajax-api/2.0/mlflow/users/current', window.location.href).toString();
  const homeUrl = new URL('.', window.location.href).toString();

  const goHome = () => window.location.assign(homeUrl);
  try {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', usersCurrentUrl, true, bogus, bogus);
    // Wait for the cache write before navigating - otherwise the home-page
    // request could fire with stale creds and silently auto-auth.
    xhr.onloadend = goHome;
    xhr.send();
  } catch {
    goHome();
  }
};
