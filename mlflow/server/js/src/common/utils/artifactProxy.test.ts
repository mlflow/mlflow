import { afterAll, afterEach, beforeAll, describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from './setup-msw';
import { ErrorWrapper } from './ErrorWrapper';
import {
  adaptArtifactProxyListResponse,
  getArtifactProxyDownloadUrl,
  getArtifactProxyListUrl,
  getArtifactProxyRoot,
  fetchArtifactProxyList,
  isEligibleArtifactProxyUri,
} from './artifactProxy';

// jsdom serves pages from http://localhost, so that is the UI origin under test.
const ORIGIN = 'http://localhost';
const ANCHOR = '/api/2.0/mlflow-artifacts/artifacts';

describe('isEligibleArtifactProxyUri', () => {
  test('accepts a same-origin proxy URI with an artifact root', () => {
    expect(isEligibleArtifactProxyUri(`${ORIGIN}${ANCHOR}/my-root`)).toBe(true);
  });

  test('accepts a same-origin proxy URI ending exactly at the route anchor', () => {
    expect(isEligibleArtifactProxyUri(`${ORIGIN}${ANCHOR}`)).toBe(true);
  });

  test('accepts a URI mounted under a static prefix', () => {
    // MLflow can be served under a static prefix, so the anchor is not
    // necessarily at the start of the path.
    expect(isEligibleArtifactProxyUri(`${ORIGIN}/mlflow${ANCHOR}/my-root`)).toBe(true);
  });

  test('accepts a URI anchored at the ajax-api artifact proxy route', () => {
    // The UI serves the artifact proxy API under both anchors.
    expect(isEligibleArtifactProxyUri(`${ORIGIN}/ajax-api/2.0/mlflow-artifacts/artifacts/my-root`)).toBe(true);
  });

  test('rejects a cross-origin URI', () => {
    expect(isEligibleArtifactProxyUri(`http://artifacts.example.com${ANCHOR}/my-root`)).toBe(false);
  });

  test('rejects a same-host URI on a different port', () => {
    expect(isEligibleArtifactProxyUri(`http://localhost:5500${ANCHOR}/my-root`)).toBe(false);
  });

  test('rejects a non-http scheme', () => {
    expect(isEligibleArtifactProxyUri('mlflow-artifacts://localhost/my-root')).toBe(false);
    expect(isEligibleArtifactProxyUri('s3://bucket/my-root')).toBe(false);
    expect(isEligibleArtifactProxyUri('file:///tmp/my-root')).toBe(false);
  });

  test('rejects a same-origin URI that does not target the artifact proxy API', () => {
    expect(isEligibleArtifactProxyUri(`${ORIGIN}/api/2.0/mlflow/runs/get`)).toBe(false);
  });

  test('rejects relative path traversal', () => {
    expect(isEligibleArtifactProxyUri(`${ORIGIN}${ANCHOR}/../../../../`)).toBe(false);
  });

  test('rejects percent-encoded path traversal', () => {
    expect(isEligibleArtifactProxyUri(`${ORIGIN}${ANCHOR}/..%2f..%2f..%2f`)).toBe(false);
  });

  test('rejects doubly-encoded path traversal', () => {
    expect(isEligibleArtifactProxyUri(`${ORIGIN}${ANCHOR}/..%252f..%252f`)).toBe(false);
  });

  test('rejects an embedded NUL byte', () => {
    expect(isEligibleArtifactProxyUri(`${ORIGIN}${ANCHOR}%00`)).toBe(false);
  });

  test('rejects a non-absolute URI', () => {
    expect(isEligibleArtifactProxyUri('/api/2.0/mlflow-artifacts/artifacts/my-root')).toBe(false);
  });

  test('rejects empty and nullish input', () => {
    expect(isEligibleArtifactProxyUri('')).toBe(false);
    expect(isEligibleArtifactProxyUri(undefined)).toBe(false);
  });
});

describe('getArtifactProxyRoot', () => {
  test('splits the URI into its origin and artifact root', () => {
    expect(getArtifactProxyRoot(`${ORIGIN}${ANCHOR}/my-root/nested`)).toEqual({
      origin: ORIGIN,
      root: 'my-root/nested',
      anchor: ANCHOR,
    });
  });

  test('returns an empty root when the URI ends at the route anchor', () => {
    expect(getArtifactProxyRoot(`${ORIGIN}${ANCHOR}`)).toEqual({ origin: ORIGIN, root: '', anchor: ANCHOR });
  });

  test('splits a URI anchored at the ajax-api route', () => {
    expect(getArtifactProxyRoot(`${ORIGIN}/ajax-api/2.0/mlflow-artifacts/artifacts/my-root`)).toEqual({
      origin: ORIGIN,
      root: 'my-root',
      anchor: '/ajax-api/2.0/mlflow-artifacts/artifacts',
    });
  });

  test('preserves a static prefix in the origin', () => {
    expect(getArtifactProxyRoot(`${ORIGIN}/mlflow${ANCHOR}/my-root`)).toEqual({
      origin: `${ORIGIN}/mlflow`,
      root: 'my-root',
      anchor: ANCHOR,
    });
  });
});

describe('getArtifactProxyDownloadUrl', () => {
  test('joins the artifact root and the requested path', () => {
    expect(getArtifactProxyDownloadUrl(`${ORIGIN}${ANCHOR}/my-root`, 'dir/file.txt')).toBe(
      `${ORIGIN}${ANCHOR}/my-root/dir/file.txt`,
    );
  });

  test('omits the root when the URI ends at the route anchor', () => {
    expect(getArtifactProxyDownloadUrl(`${ORIGIN}${ANCHOR}`, 'file.txt')).toBe(`${ORIGIN}${ANCHOR}/file.txt`);
  });

  test('tolerates a leading slash on the requested path', () => {
    expect(getArtifactProxyDownloadUrl(`${ORIGIN}${ANCHOR}/my-root`, '/file.txt')).toBe(
      `${ORIGIN}${ANCHOR}/my-root/file.txt`,
    );
  });

  test('preserves the ajax-api anchor in the download URL', () => {
    expect(getArtifactProxyDownloadUrl(`${ORIGIN}/ajax-api/2.0/mlflow-artifacts/artifacts/my-root`, 'file.txt')).toBe(
      `${ORIGIN}/ajax-api/2.0/mlflow-artifacts/artifacts/my-root/file.txt`,
    );
  });

  test('percent-encodes each path segment', () => {
    expect(getArtifactProxyDownloadUrl(`${ORIGIN}${ANCHOR}/my-root`, 'a b/c#d.txt')).toBe(
      `${ORIGIN}${ANCHOR}/my-root/a%20b/c%23d.txt`,
    );
  });
});

describe('getArtifactProxyListUrl', () => {
  test('sends the joined root and path as the path query parameter', () => {
    expect(getArtifactProxyListUrl(`${ORIGIN}${ANCHOR}/my-root`, 'dir')).toBe(`${ORIGIN}${ANCHOR}?path=my-root%2Fdir`);
  });

  test('sends the bare root when no path is requested', () => {
    expect(getArtifactProxyListUrl(`${ORIGIN}${ANCHOR}/my-root`, undefined)).toBe(`${ORIGIN}${ANCHOR}?path=my-root`);
  });

  test('omits the path parameter entirely when the joined path is empty', () => {
    // An absent `path` is how the artifact proxy API expresses "list the root".
    expect(getArtifactProxyListUrl(`${ORIGIN}${ANCHOR}`, undefined)).toBe(`${ORIGIN}${ANCHOR}`);
  });
});

describe('adaptArtifactProxyListResponse', () => {
  test('re-prefixes returned basenames with the requested path', () => {
    const response = {
      files: [
        { path: 'file1', is_dir: false, file_size: '159' },
        { path: 'dir1', is_dir: true },
      ],
    };

    expect(adaptArtifactProxyListResponse(response, 'sub')).toEqual({
      files: [
        { path: 'sub/dir1', is_dir: true },
        { path: 'sub/file1', is_dir: false, file_size: '159' },
      ],
    });
  });

  test('returns basenames unchanged when no path was requested', () => {
    const response = { files: [{ path: 'file1', is_dir: false, file_size: '159' }] };

    expect(adaptArtifactProxyListResponse(response, undefined)).toEqual({
      files: [{ path: 'file1', is_dir: false, file_size: '159' }],
    });
  });

  test('sorts entries by path', () => {
    const response = {
      files: [
        { path: 'b', is_dir: false },
        { path: 'a', is_dir: false },
      ],
    };

    expect(adaptArtifactProxyListResponse(response, undefined).files.map((f) => f.path)).toEqual(['a', 'b']);
  });

  test('returns no entries when the requested path refers to a single file', () => {
    // The proxy API echoes the file itself; the tracking list API returns an
    // empty list in this case, and the UI depends on that behaviour.
    const response = { files: [{ path: 'file.txt', is_dir: false, file_size: '5' }] };

    expect(adaptArtifactProxyListResponse(response, 'sub/file.txt')).toEqual({ files: [] });
  });

  test('tolerates a response with no files', () => {
    expect(adaptArtifactProxyListResponse({}, 'sub')).toEqual({ files: [] });
  });
});

describe('fetchArtifactProxyList', () => {
  const ARTIFACT_URI = `${ORIGIN}${ANCHOR}/my-root`;

  let capturedRequests: { url: string; authorization: string | null }[] = [];
  const server = setupServer();

  beforeAll(() => server.listen());

  afterEach(() => {
    capturedRequests = [];
    server.resetHandlers();
  });

  afterAll(() => server.close());

  const respondWith = (body: any, status = 200) =>
    server.use(
      rest.get(/\/api\/2\.0\/mlflow-artifacts\/artifacts/, (req, res, ctx) => {
        capturedRequests.push({ url: req.url.toString(), authorization: req.headers.get('Authorization') });
        return res(ctx.status(status), ctx.json(body));
      }),
    );

  test('requests the joined path and re-prefixes the returned basenames', async () => {
    respondWith({ files: [{ path: 'file1', is_dir: false, file_size: '159' }] });

    const result = await fetchArtifactProxyList(ARTIFACT_URI, 'sub');

    expect(capturedRequests).toHaveLength(1);
    expect(capturedRequests[0].url).toBe(`${ORIGIN}${ANCHOR}?path=my-root%2Fsub`);
    expect(result).toEqual({ files: [{ path: 'sub/file1', is_dir: false, file_size: '159' }] });
  });

  test('forwards the headers the UI already sends', async () => {
    document.cookie = 'mlflow-request-header-Authorization=Bearer token';
    respondWith({ files: [] });

    await fetchArtifactProxyList(ARTIFACT_URI, undefined);

    expect(capturedRequests[0].authorization).toBe('Bearer token');
  });

  test('surfaces a failed request rather than falling back to tracking', async () => {
    respondWith({ message: 'permission denied' }, 403);

    await expect(fetchArtifactProxyList(ARTIFACT_URI, 'sub')).rejects.toBeInstanceOf(ErrorWrapper);
  });
});
