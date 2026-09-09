/**
 * Resolves artifact requests against a stored MLflow artifact-proxy URI.
 *
 * MLflow can run its tracking server with `--no-serve-artifacts` alongside a
 * separately addressed artifact-serving process started with `--artifacts-only`.
 * In that topology an entity's artifact URI points at the artifact-serving
 * process, and the Python client follows it via `HttpArtifactRepository`.
 * These helpers give the UI the same behaviour, mirroring the URL semantics of
 * `mlflow/store/artifact/http_artifact_repo.py`.
 */

import { ErrorWrapper } from './ErrorWrapper';
import { getDefaultHeaders, HTTPMethods } from './FetchUtils';

/**
 * The routes the artifact proxy API is served under. The first mirrors
 * `mlflow_artifacts_http_route_anchor` in `mlflow/server/handlers.py`; the
 * second is the equivalent route the UI serves under its `ajax-api` prefix.
 *
 * These are not AJAX URLs the UI builds against the tracking server; they are
 * anchors used to recognise and rebuild absolute URIs that the server stored on
 * the entity, so they must not be routed through `getAjaxUrl`.
 */
/* eslint-disable @mlflow/no-absolute-ajax-urls -- route anchors, not request URLs */
export const ARTIFACT_PROXY_ROUTE_ANCHORS = [
  '/api/2.0/mlflow-artifacts/artifacts',
  '/ajax-api/2.0/mlflow-artifacts/artifacts',
];
/* eslint-enable @mlflow/no-absolute-ajax-urls */

const findRouteAnchor = (pathname: string) => ARTIFACT_PROXY_ROUTE_ANCHORS.find((anchor) => pathname.includes(anchor));

// Bounds the decode loop below so a pathological input cannot spin.
const MAX_DECODE_PASSES = 10;

export interface ArtifactProxyFileInfo {
  path: string;
  is_dir?: boolean;
  file_size?: string;
}

export interface ArtifactProxyListResponse {
  files?: ArtifactProxyFileInfo[];
}

/**
 * Percent-decodes until the value stops changing, so that layered encodings
 * such as `..%252f` cannot smuggle a traversal past the checks below. Mirrors
 * the decode loop in `_validate_non_local_source_contains_relative_paths`.
 */
const decodeUntilStable = (value: string) => {
  let current = value;
  for (let pass = 0; pass < MAX_DECODE_PASSES; pass++) {
    let decoded: string;
    try {
      decoded = decodeURIComponent(current.replace(/\+/g, ' '));
    } catch {
      // Malformed escape sequence; treat the value as fully decoded.
      return current;
    }
    if (decoded === current) {
      return current;
    }
    current = decoded;
  }
  return current;
};

const stripLeadingSlashes = (value: string) => value.replace(/^\/+/, '');
const stripTrailingSlashes = (value: string) => value.replace(/\/+$/, '');

/**
 * Determines whether the UI may issue artifact requests directly against the
 * given stored URI.
 *
 * Eligibility is deliberately narrow: the URI must be an absolute HTTP(S) URL
 * on the UI's own origin whose path targets the artifact proxy API and contains
 * no traversal. Restricting to the UI origin is what makes it safe to attach the
 * headers the UI already sends, since they can never reach a foreign host.
 */
export const isEligibleArtifactProxyUri = (artifactUri?: string | null): artifactUri is string => {
  if (!artifactUri) {
    return false;
  }

  let url: URL;
  try {
    url = new URL(artifactUri);
  } catch {
    // Not an absolute URL, so there is no origin to compare against.
    return false;
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return false;
  }

  if (url.origin !== window.location.origin) {
    return false;
  }

  // Checked against the raw URI rather than `url.pathname`, because the URL
  // constructor silently resolves `..` segments away.
  const decodedUri = decodeUntilStable(artifactUri);
  if (decodedUri.includes('\x00') || decodedUri.split('/').some((segment) => segment === '..')) {
    return false;
  }

  const normalizedPath = stripTrailingSlashes(decodeUntilStable(url.pathname).replace(/\/+/g, '/'));

  // Containment rather than a prefix match: MLflow may be served under a static
  // prefix, so the anchor is not necessarily at the start of the path.
  return Boolean(findRouteAnchor(normalizedPath));
};

/**
 * Splits an eligible artifact proxy URI into the origin serving the proxy API
 * (including any static prefix) and the artifact root beneath the route anchor.
 */
export const getArtifactProxyRoot = (artifactUri: string): { origin: string; root: string; anchor: string } => {
  const url = new URL(artifactUri);
  const anchor = findRouteAnchor(url.pathname);

  if (!anchor) {
    throw new Error(`Artifact URI does not target the MLflow artifact proxy API: ${artifactUri}`);
  }

  const anchorIndex = url.pathname.indexOf(anchor);
  const staticPrefix = url.pathname.slice(0, anchorIndex);
  const tail = url.pathname.slice(anchorIndex + anchor.length);

  return {
    origin: `${url.origin}${staticPrefix}`,
    root: stripTrailingSlashes(stripLeadingSlashes(tail)),
    anchor,
  };
};

/** Builds the URL that serves the artifact at `path` under the stored root. */
export const getArtifactProxyDownloadUrl = (artifactUri: string, path: string): string => {
  const { origin, root, anchor } = getArtifactProxyRoot(artifactUri);
  const encodedPath = stripLeadingSlashes(path ?? '')
    .split('/')
    .filter(Boolean)
    .map(encodeURIComponent)
    .join('/');
  const suffix = [root, encodedPath].filter((segment) => segment.length > 0).join('/');

  return `${origin}${anchor}${suffix ? `/${suffix}` : ''}`;
};

/** Builds the URL that lists the artifacts at `path` under the stored root. */
export const getArtifactProxyListUrl = (artifactUri: string, path?: string): string => {
  const { origin, root, anchor } = getArtifactProxyRoot(artifactUri);
  const relativePath = stripLeadingSlashes(path ?? '');
  const joinedPath = [root, relativePath].filter((segment) => segment.length > 0).join('/');
  const listUrl = `${origin}${anchor}`;

  // An absent `path` is how the proxy API expresses "list the artifact root".
  return joinedPath ? `${listUrl}?${new URLSearchParams({ path: joinedPath }).toString()}` : listUrl;
};

/**
 * Reshapes a proxy list response into the shape the tracking list API returns.
 *
 * The proxy handler returns bare basenames and omits `root_uri`, whereas the UI
 * builds its artifact tree from full run-relative paths.
 */
export const adaptArtifactProxyListResponse = (
  response: ArtifactProxyListResponse,
  path?: string,
): { files: ArtifactProxyFileInfo[] } => {
  const requestedPath = stripTrailingSlashes(path ?? '');
  const files = (response?.files ?? []).map((file) => ({
    ...file,
    path: requestedPath ? `${requestedPath}/${file.path}` : file.path,
  }));

  // The proxy API echoes the file itself when `path` refers to a single file,
  // while the tracking API returns an empty list. The UI relies on the latter.
  if (files.length === 1 && !files[0].is_dir && requestedPath) {
    const basename = requestedPath.split('/').filter(Boolean).pop() ?? '';
    if (files[0].path === `${requestedPath}/${basename}`) {
      return { files: [] };
    }
  }

  return { files: files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0)) };
};

/**
 * Lists the artifacts under `path` directly from the artifact proxy.
 *
 * The request bypasses `getJson` because that helper routes relative URLs
 * through `getAjaxUrl`, which would corrupt an absolute URL. A failure is
 * raised rather than retried through the tracking server: falling back would
 * put tracking back on the artifact path and mask an authorization or
 * configuration problem on the artifact server.
 */
export const fetchArtifactProxyList = async (
  artifactUri: string,
  path?: string,
): Promise<{ files: ArtifactProxyFileInfo[] }> => {
  const request = new Request(getArtifactProxyListUrl(artifactUri, path), {
    method: HTTPMethods.GET,
    headers: new Headers(getDefaultHeaders(document.cookie) as HeadersInit),
  });

  // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
  const response = await fetch(request);

  if (!response.ok) {
    const errorMessage = (await response.text()) || response.statusText;
    throw new ErrorWrapper(errorMessage, response.status);
  }

  return adaptArtifactProxyListResponse(await response.json(), path);
};
