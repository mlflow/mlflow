import type { ArtifactRequestOptions } from '../../common/utils/ArtifactUtils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { getAjaxUrl, getDefaultHeaders } from '../../common/utils/FetchUtils';
import {
  getArtifactsPresignedOnlySync,
  getLoggedModelArtifactReadCredentialsSupportedSync,
  getModelArtifactPresignedDownloadsSupportedSync,
  getMultipartDownloadsEnabledSync,
  getMultipartUploadsEnabledSync,
  getPresignedUploadRunIdSupportedSync,
} from '../hooks/useServerInfo';
import { MlflowService } from '../sdk/MlflowService';

const MLFLOW_ARTIFACTS_ROUTE = /\/(?:ajax-api|api)\/2\.0\/mlflow-artifacts\/artifacts\//;

const PRESIGNED_FALLBACK_STATUSES = [400, 404, 501, 503];

const joinArtifactPaths = (rootPath: string, artifactPath: string) =>
  [rootPath.replace(/^\/+|\/+$/g, ''), artifactPath.replace(/^\/+/, '')].filter(Boolean).join('/');

class ArtifactRootResolutionError extends ErrorWrapper {
  constructor() {
    super(
      {
        error_code: 'INVALID_PARAMETER_VALUE',
        message: 'The artifact root URI cannot be decoded safely.',
        invalid_artifact_root: true,
      },
      400,
    );
  }
}

const getDecodedPathname = (url: URL) => {
  try {
    // URI roots can contain a literal percent sign. Preserve stray percent
    // signs while still decoding valid UTF-8 escapes and encoded separators.
    return decodeURIComponent(url.pathname.replace(/%(?![0-9a-f]{2})/gi, '%25'));
  } catch {
    throw new ArtifactRootResolutionError();
  }
};

/**
 * Returns an artifact-service-relative path for proxied artifact URIs.
 * Direct cloud storage URIs intentionally return undefined.
 */
export type ProxiedArtifactRoute = {
  path: string;
  artifactServiceBaseUrl?: string;
};

export const getProxiedArtifactRoute = (
  artifactRootUri?: string,
  artifactPath = '',
): ProxiedArtifactRoute | undefined => {
  if (!artifactRootUri) {
    return undefined;
  }
  try {
    const parsedArtifactRootUri = new URL(artifactRootUri);
    if (parsedArtifactRootUri.protocol === 'mlflow-artifacts:') {
      const path = joinArtifactPaths(getDecodedPathname(parsedArtifactRootUri), artifactPath);
      if (parsedArtifactRootUri.host) {
        // Explicit authorities replace the tracking host, retaining its scheme and
        // deployment prefix, just as the Python artifact repository does.
        const artifactServiceUrl = new URL(getAjaxUrl('./'), document.baseURI);
        // Parse as an HTTP origin first: assigning an invalid URL.host can fail
        // silently and leave the tracking host in place.
        const artifactOrigin = new URL(`${artifactServiceUrl.protocol}//${parsedArtifactRootUri.host}/`);
        artifactServiceUrl.hostname = artifactOrigin.hostname;
        // Setting host without a port leaves the old URL's port unchanged.
        artifactServiceUrl.port = artifactOrigin.port;
        return { path, artifactServiceBaseUrl: artifactServiceUrl.toString() };
      }
      return { path };
    }
    if (parsedArtifactRootUri.protocol === 'http:' || parsedArtifactRootUri.protocol === 'https:') {
      const rootPath = getDecodedPathname(parsedArtifactRootUri);
      const route = MLFLOW_ARTIFACTS_ROUTE.exec(rootPath);
      if (route) {
        const deploymentPrefix = rootPath.slice(0, route.index + 1);
        const artifactServiceBaseUrl = new URL(deploymentPrefix, parsedArtifactRootUri.origin).toString();
        return {
          path: joinArtifactPaths(rootPath.slice(route.index + route[0].length), artifactPath),
          artifactServiceBaseUrl,
        };
      }
    }
  } catch (error) {
    if (
      error instanceof ArtifactRootResolutionError ||
      artifactRootUri.toLowerCase().startsWith('mlflow-artifacts:') ||
      MLFLOW_ARTIFACTS_ROUTE.test(artifactRootUri)
    ) {
      throw error instanceof ArtifactRootResolutionError ? error : new ArtifactRootResolutionError();
    }
    return undefined;
  }
  return undefined;
};

const getErrorStatus = (error: unknown) => (error instanceof ErrorWrapper ? error.getStatus() : undefined);

export const shouldFallBackToProxy = (error: unknown, presignedOnly: boolean) => {
  if (presignedOnly || error instanceof ArtifactRootResolutionError) return false;
  // Browser fetch reports network and CORS failures as TypeError without an HTTP status.
  if (
    error instanceof TypeError ||
    (error instanceof ErrorWrapper && error.getStatus() === 500 && error.text instanceof TypeError)
  ) {
    return true;
  }
  const status = getErrorStatus(error);
  return status !== undefined && PRESIGNED_FALLBACK_STATUSES.includes(status);
};

const normalizeCredentialHeaders = (
  headers?: Record<string, string> | { name: string; value: string }[],
): Record<string, string> =>
  Array.isArray(headers) ? Object.fromEntries(headers.map(({ name, value }) => [name, value])) : (headers ?? {});

const RUN_ARTIFACT_ROOT_CACHE_TTL_MS = 60_000;
const RUN_ARTIFACT_ROOT_CACHE_MAX_ENTRIES = 100;
const runArtifactRootCache = new Map<string, { promise: Promise<string | undefined>; expiresAt: number }>();

const getRunArtifactRootUri = (runUuid: string): Promise<string | undefined> => {
  // Artifact locations are metadata, not credentials. Share them briefly across
  // image grids and uploads without reusing them across deployments or sessions.
  const key = JSON.stringify([
    new URL(getAjaxUrl('./'), document.baseURI).toString(),
    document.cookie,
    Object.entries(getDefaultHeaders(document.cookie)).sort(([left], [right]) => left.localeCompare(right)),
    runUuid,
  ]);
  const cached = runArtifactRootCache.get(key);
  if (cached && cached.expiresAt > Date.now()) {
    // Keep the most recently used entries when the size limit is reached.
    runArtifactRootCache.delete(key);
    runArtifactRootCache.set(key, cached);
    return cached.promise;
  }

  const entry = {
    promise: MlflowService.getRun({ run_id: runUuid }).then(
      (response) => response?.run?.info?.artifactUri as string | undefined,
    ),
    expiresAt: Date.now() + RUN_ARTIFACT_ROOT_CACHE_TTL_MS,
  };
  runArtifactRootCache.delete(key);
  runArtifactRootCache.set(key, entry);
  if (runArtifactRootCache.size > RUN_ARTIFACT_ROOT_CACHE_MAX_ENTRIES) {
    const oldestKey = runArtifactRootCache.keys().next().value;
    if (oldestKey !== undefined) runArtifactRootCache.delete(oldestKey);
  }

  const evict = () => {
    if (runArtifactRootCache.get(key) === entry) runArtifactRootCache.delete(key);
  };
  // Failed or incomplete metadata must be retried on the next request.
  entry.promise.then((root) => !root && evict(), evict);
  return entry.promise;
};

export type PresignedArtifactParams = {
  runUuid: string;
  path: string;
  artifactRootUri?: string;
  isLoggedModelsMode?: boolean;
  loggedModelId?: string;
  multipartDownloadsEnabled?: boolean;
};

export type PresignedArtifactDownload = {
  url: string;
  headers: Record<string, string>;
  /** Same-service byte route retained when a previously resolved download is reused. */
  proxyUrl?: string;
};

export type ArtifactDownloadResolution = {
  presigned?: PresignedArtifactDownload;
  proxyUrl?: string;
};

/** Resolve signing and its compatibility destination together so fallback preserves the artifact store. */
export const resolveArtifactDownload = async (params: PresignedArtifactParams): Promise<ArtifactDownloadResolution> => {
  const artifactsPresignedOnly = getArtifactsPresignedOnlySync();
  const canResolveLoggedModel = params.isLoggedModelsMode && params.loggedModelId;
  if (!params.artifactRootUri && !params.runUuid && !artifactsPresignedOnly && !canResolveLoggedModel) {
    return {};
  }
  // Discovery failures do not identify a safe compatibility destination.
  const artifactRootUri =
    params.artifactRootUri ??
    (canResolveLoggedModel
      ? (await MlflowService.getLoggedModel(canResolveLoggedModel))?.model?.info?.artifact_uri
      : params.runUuid
        ? await getRunArtifactRootUri(params.runUuid)
        : undefined);
  if (!artifactRootUri) {
    throw new ErrorWrapper('The server did not return an artifact root for this artifact.', 501);
  }
  let proxyUrl: string | undefined;
  try {
    const proxiedArtifactRoute = getProxiedArtifactRoute(artifactRootUri, params.path);
    if (proxiedArtifactRoute) {
      // Validate the origin even when signing is disabled, and retain the exact
      // service prefix and artifact root rather than switching to the tracking proxy.
      proxyUrl = MlflowService.getMlflowArtifactsProxyUrl(
        proxiedArtifactRoute.path,
        proxiedArtifactRoute.artifactServiceBaseUrl,
      );
      const multipartDownloadsEnabled = params.multipartDownloadsEnabled ?? getMultipartDownloadsEnabledSync();
      if (!multipartDownloadsEnabled && !artifactsPresignedOnly && !proxiedArtifactRoute.artifactServiceBaseUrl) {
        return { proxyUrl };
      }
      const response = proxiedArtifactRoute.artifactServiceBaseUrl
        ? await MlflowService.getMlflowArtifactsPresignedDownloadUrl(
            proxiedArtifactRoute.path,
            proxiedArtifactRoute.artifactServiceBaseUrl,
          )
        : await MlflowService.getMlflowArtifactsPresignedDownloadUrl(proxiedArtifactRoute.path);
      if (response.url) {
        return { presigned: { url: response.url, headers: response.headers ?? {} }, proxyUrl };
      }
    } else if (params.isLoggedModelsMode && params.loggedModelId) {
      if (artifactsPresignedOnly || getModelArtifactPresignedDownloadsSupportedSync()) {
        const response = await MlflowService.getLoggedModelPresignedDownloadUrl({
          loggedModelId: params.loggedModelId,
          path: params.path,
        });
        if (response.url) return { presigned: { url: response.url, headers: response.headers ?? {} } };
        throw new ErrorWrapper('The server did not return a presigned logged-model artifact URL.', 501);
      }
      if (!getLoggedModelArtifactReadCredentialsSupportedSync()) return {};
      const response = await MlflowService.getCredentialsForLoggedModelArtifactRead({
        loggedModelId: params.loggedModelId,
        path: params.path,
      });
      const credentials = response.credentials ?? [];
      const credential =
        credentials.find(({ credential_info }) => credential_info.path === params.path)?.credential_info ??
        credentials[0]?.credential_info;
      if (credential?.signed_uri) {
        return {
          presigned: {
            url: credential.signed_uri,
            headers: normalizeCredentialHeaders(credential.headers),
          },
        };
      }
    } else if (!params.isLoggedModelsMode && params.runUuid) {
      const response = await MlflowService.createPresignedDownloadUrl({
        run_id: params.runUuid,
        path: params.path,
      });
      if (response.presigned_url) {
        return { presigned: { url: response.presigned_url, headers: response.headers ?? {} } };
      }
    } else if (!artifactsPresignedOnly) {
      return {};
    }

    throw new ErrorWrapper('The server did not return a presigned artifact URL.', 501);
  } catch (error) {
    if (shouldFallBackToProxy(error, artifactsPresignedOnly)) {
      return { proxyUrl };
    }
    throw error;
  }
};

export type GetArtifactDataFn<T = unknown> = (artifactLocation: string, options?: ArtifactRequestOptions) => Promise<T>;

export const fetchArtifactPreferPresigned = async <T>(
  params: PresignedArtifactParams,
  legacyArtifactLocation: string,
  getArtifactData: GetArtifactDataFn<T>,
  presignedDownload?: PresignedArtifactDownload,
) => {
  let resolution: ArtifactDownloadResolution;
  if (presignedDownload) {
    const route = getProxiedArtifactRoute(params.artifactRootUri, params.path);
    resolution = {
      presigned: presignedDownload,
      proxyUrl:
        presignedDownload.proxyUrl ??
        (route && MlflowService.getMlflowArtifactsProxyUrl(route.path, route.artifactServiceBaseUrl)),
    };
  } else {
    resolution = await resolveArtifactDownload(params);
  }
  const { presigned } = resolution;
  const proxyLocation = resolution.proxyUrl ?? legacyArtifactLocation;
  if (!presigned) {
    return getArtifactData(proxyLocation);
  }
  try {
    return await getArtifactData(presigned.url, { headers: presigned.headers });
  } catch (error) {
    if (shouldFallBackToProxy(error, getArtifactsPresignedOnlySync())) {
      return getArtifactData(proxyLocation);
    }
    throw error;
  }
};

export const fetchRunArtifactPreferPresigned = async <T>(
  runUuid: string,
  path: string,
  legacyArtifactLocation: string,
  getArtifactData: GetArtifactDataFn<T>,
  artifactRootUri?: string,
) => {
  const root = artifactRootUri ?? (await getRunArtifactRootUri(runUuid));
  return fetchArtifactPreferPresigned(
    { runUuid, path, artifactRootUri: root },
    legacyArtifactLocation,
    getArtifactData,
  );
};

const throwForResponse = async (response: Response) => {
  if (!response.ok) {
    throw new ErrorWrapper((await response.text()) || response.statusText, response.status);
  }
  return response;
};

const uploadToPresignedUrl = (url: string, headers: Record<string, string>, body: BodyInit) =>
  // Do not attach tracking-server headers to a cloud-storage request.
  // eslint-disable-next-line no-restricted-globals -- presigned URLs must be fetched directly
  fetch(url, { method: 'PUT', headers, body }).then(throwForResponse);

const splitArtifactPath = (path: string) => {
  const normalized = path.replace(/^\/+|\/+$/g, '');
  const separatorIndex = normalized.lastIndexOf('/');
  return separatorIndex === -1
    ? { directory: '', filename: normalized }
    : { directory: normalized.slice(0, separatorIndex), filename: normalized.slice(separatorIndex + 1) };
};

class MissingMultipartUploadETagError extends ErrorWrapper {
  constructor() {
    super(
      'The storage upload response did not include an ETag. Configure storage CORS to expose the ETag response header.',
      500,
    );
  }
}

const uploadProxiedArtifact = async (proxiedArtifactRoute: ProxiedArtifactRoute, body: BodyInit) => {
  const { directory, filename } = splitArtifactPath(proxiedArtifactRoute.path);
  const createData = { path: filename, num_parts: 1 };
  const createResponse = proxiedArtifactRoute.artifactServiceBaseUrl
    ? await MlflowService.createMlflowArtifactsMultipartUpload(
        directory,
        createData,
        proxiedArtifactRoute.artifactServiceBaseUrl,
      )
    : await MlflowService.createMlflowArtifactsMultipartUpload(directory, createData);
  const credential = createResponse.credentials?.[0];
  try {
    if (!credential?.url) {
      throw new ErrorWrapper('The server did not return a multipart upload credential.', 501);
    }
    const uploadResponse = await uploadToPresignedUrl(credential.url, credential.headers ?? {}, body);
    const etag = uploadResponse.headers.get('ETag');
    const uploadUrl = new URL(credential.url);
    // Azure Put Block has no ETag; completion uses the signed URL's block ID.
    const isAzureBlockUpload =
      uploadUrl.searchParams.get('comp') === 'block' && Boolean(uploadUrl.searchParams.get('blockid'));
    if (!etag && !isAzureBlockUpload) {
      throw new MissingMultipartUploadETagError();
    }
    const completeData = {
      path: filename,
      upload_id: createResponse.upload_id,
      parts: [
        {
          part_number: credential.part_number,
          etag: etag ?? '',
          url: credential.url,
        },
      ],
    };
    if (proxiedArtifactRoute.artifactServiceBaseUrl) {
      await MlflowService.completeMlflowArtifactsMultipartUpload(
        directory,
        completeData,
        proxiedArtifactRoute.artifactServiceBaseUrl,
      );
    } else {
      await MlflowService.completeMlflowArtifactsMultipartUpload(directory, completeData);
    }
  } catch (error) {
    try {
      const abortData = { path: filename, upload_id: createResponse.upload_id };
      if (proxiedArtifactRoute.artifactServiceBaseUrl) {
        await MlflowService.abortMlflowArtifactsMultipartUpload(
          directory,
          abortData,
          proxiedArtifactRoute.artifactServiceBaseUrl,
        );
      } else {
        await MlflowService.abortMlflowArtifactsMultipartUpload(directory, abortData);
      }
    } catch {
      // Preserve the upload failure; abort is only best-effort cleanup.
    }
    throw error;
  }
};

/**
 * Attempts a direct upload and returns false only when legacy upload is an
 * allowed compatibility fallback.
 */
export const uploadArtifactWithPresignedUrl = async (
  runUuid: string,
  path: string,
  body: BodyInit,
  artifactRootUri?: string,
): Promise<boolean> => {
  const artifactsPresignedOnly = getArtifactsPresignedOnlySync();
  const multipartUploadsEnabled = getMultipartUploadsEnabledSync();
  const presignedUploadRunIdSupported = getPresignedUploadRunIdSupportedSync();
  // A failed metadata lookup must not redirect an upload to an unknown store.
  const root = artifactRootUri ?? (await getRunArtifactRootUri(runUuid));
  if (!root) {
    throw new ErrorWrapper('The server did not return an artifact root for this run.', 501);
  }
  let proxiedArtifactRoute: ProxiedArtifactRoute | undefined;
  let proxyUploadAttempted = false;
  try {
    proxiedArtifactRoute = getProxiedArtifactRoute(root, path);
    if (proxiedArtifactRoute) {
      // Reject unsupported origins before either direct or compatibility transfer.
      MlflowService.getMlflowArtifactsProxyUrl(proxiedArtifactRoute.path, proxiedArtifactRoute.artifactServiceBaseUrl);
      if (!multipartUploadsEnabled && !artifactsPresignedOnly && !proxiedArtifactRoute.artifactServiceBaseUrl) {
        proxyUploadAttempted = true;
        await MlflowService.uploadMlflowArtifact(
          proxiedArtifactRoute.path,
          body,
          proxiedArtifactRoute.artifactServiceBaseUrl,
        );
        return true;
      }
      await uploadProxiedArtifact(proxiedArtifactRoute, body);
      return true;
    }

    if (presignedUploadRunIdSupported || artifactsPresignedOnly) {
      const response = await MlflowService.createPresignedUploadUrl({ run_id: runUuid, path });
      if (!response.presigned_url) {
        throw new ErrorWrapper('The server did not return a presigned artifact upload URL.', 501);
      }
      await uploadToPresignedUrl(response.presigned_url, response.headers ?? {}, body);
      return true;
    }

    return false;
  } catch (error) {
    if (
      proxiedArtifactRoute &&
      !proxyUploadAttempted &&
      !artifactsPresignedOnly &&
      (error instanceof MissingMultipartUploadETagError || shouldFallBackToProxy(error, false))
    ) {
      // Multipart cleanup has completed before errors reach here. Compatibility
      // uploads must use that same service and full root path, including custom roots.
      await MlflowService.uploadMlflowArtifact(
        proxiedArtifactRoute.path,
        body,
        proxiedArtifactRoute.artifactServiceBaseUrl,
      );
      return true;
    }
    if (!proxiedArtifactRoute && shouldFallBackToProxy(error, artifactsPresignedOnly)) {
      return false;
    }
    throw error;
  }
};
