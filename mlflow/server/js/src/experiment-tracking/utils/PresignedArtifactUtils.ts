import type { ArtifactRequestOptions } from '../../common/utils/ArtifactUtils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import {
  getArtifactsPresignedOnlySync,
  getMultipartDownloadsEnabledSync,
  getMultipartUploadsEnabledSync,
  getPresignedUploadRunIdSupportedSync,
} from '../hooks/useServerInfo';
import { MlflowService } from '../sdk/MlflowService';

const MLFLOW_ARTIFACTS_ROUTE_ANCHORS = [
  'api/2.0/mlflow-artifacts/artifacts/',
  'ajax-api/2.0/mlflow-artifacts/artifacts/',
];

const PRESIGNED_FALLBACK_STATUSES = [400, 404, 501, 503];

const joinArtifactPaths = (rootPath: string, artifactPath: string) =>
  [rootPath.replace(/^\/+|\/+$/g, ''), artifactPath.replace(/^\/+/, '')].filter(Boolean).join('/');

const getDecodedPathname = (url: URL) => decodeURIComponent(url.pathname);

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
      return { path: joinArtifactPaths(getDecodedPathname(parsedArtifactRootUri), artifactPath) };
    }
    if (parsedArtifactRootUri.protocol === 'http:' || parsedArtifactRootUri.protocol === 'https:') {
      const rootPath = getDecodedPathname(parsedArtifactRootUri).replace(/^\/+/, '');
      const routeAnchor = MLFLOW_ARTIFACTS_ROUTE_ANCHORS.find((anchor) => rootPath.includes(anchor));
      if (routeAnchor) {
        const routeAnchorIndex = rootPath.indexOf(routeAnchor);
        const deploymentPrefix = rootPath.slice(0, routeAnchorIndex);
        const artifactServiceBaseUrl = new URL(`/${deploymentPrefix}`, parsedArtifactRootUri.origin).toString();
        return {
          path: joinArtifactPaths(rootPath.slice(routeAnchorIndex + routeAnchor.length), artifactPath),
          artifactServiceBaseUrl,
        };
      }
    }
  } catch {
    return undefined;
  }
  return undefined;
};

export const getProxiedArtifactPath = (artifactRootUri?: string, artifactPath = '') =>
  getProxiedArtifactRoute(artifactRootUri, artifactPath)?.path;

const getErrorStatus = (error: unknown) => (error instanceof ErrorWrapper ? error.getStatus() : undefined);

const canFallBackFromPresignedError = (error: unknown) => {
  const status = getErrorStatus(error);
  return status !== undefined && PRESIGNED_FALLBACK_STATUSES.includes(status);
};

const normalizeCredentialHeaders = (
  headers?: Record<string, string> | { name: string; value: string }[],
): Record<string, string> =>
  Array.isArray(headers) ? Object.fromEntries(headers.map(({ name, value }) => [name, value])) : (headers ?? {});

const getRunArtifactRootUri = async (runUuid: string) => {
  const response = await MlflowService.getRun({ run_id: runUuid });
  return response?.run?.info?.artifactUri as string | undefined;
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
};

/**
 * Resolves the direct download URL for an artifact. Returning undefined means
 * that callers may use the legacy tracking-server route for compatibility.
 */
export const resolvePresignedArtifactDownload = async (
  params: PresignedArtifactParams,
): Promise<PresignedArtifactDownload | undefined> => {
  const artifactsPresignedOnly = getArtifactsPresignedOnlySync();
  if (!params.artifactRootUri && !artifactsPresignedOnly) {
    return undefined;
  }
  const artifactRootUri = params.artifactRootUri ?? (params.runUuid ? await getRunArtifactRootUri(params.runUuid) : '');
  const proxiedArtifactRoute = getProxiedArtifactRoute(artifactRootUri, params.path);

  try {
    if (proxiedArtifactRoute) {
      const multipartDownloadsEnabled = params.multipartDownloadsEnabled ?? getMultipartDownloadsEnabledSync();
      if (!multipartDownloadsEnabled && !artifactsPresignedOnly && !proxiedArtifactRoute.artifactServiceBaseUrl) {
        return undefined;
      }
      const response = proxiedArtifactRoute.artifactServiceBaseUrl
        ? await MlflowService.getMlflowArtifactsPresignedDownloadUrl(
            proxiedArtifactRoute.path,
            proxiedArtifactRoute.artifactServiceBaseUrl,
          )
        : await MlflowService.getMlflowArtifactsPresignedDownloadUrl(proxiedArtifactRoute.path);
      if (response.url) {
        return { url: response.url, headers: response.headers ?? {} };
      }
    } else if (params.isLoggedModelsMode && params.loggedModelId) {
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
          url: credential.signed_uri,
          headers: normalizeCredentialHeaders(credential.headers),
        };
      }
    } else if (!params.isLoggedModelsMode && params.runUuid) {
      const response = await MlflowService.createPresignedDownloadUrl({
        run_id: params.runUuid,
        path: params.path,
      });
      if (response.presigned_url) {
        return { url: response.presigned_url, headers: response.headers ?? {} };
      }
    } else if (!artifactsPresignedOnly) {
      return undefined;
    }

    throw new ErrorWrapper('The server did not return a presigned artifact URL.', 501);
  } catch (error) {
    if (!artifactsPresignedOnly && canFallBackFromPresignedError(error)) {
      return undefined;
    }
    throw error;
  }
};

export type GetArtifactDataFn<T = unknown> = (artifactLocation: string, options?: ArtifactRequestOptions) => Promise<T>;

export const fetchArtifactWithPresignedUrl = async <T>(
  params: PresignedArtifactParams,
  legacyArtifactLocation: string,
  getArtifactData: GetArtifactDataFn<T>,
) => {
  const presigned = await resolvePresignedArtifactDownload(params);
  return presigned
    ? getArtifactData(presigned.url, { headers: presigned.headers })
    : getArtifactData(legacyArtifactLocation);
};

export const fetchRunArtifactWithPresignedUrl = async <T>(
  runUuid: string,
  path: string,
  legacyArtifactLocation: string,
  getArtifactData: GetArtifactDataFn<T>,
) => {
  const artifactRootUri = await getRunArtifactRootUri(runUuid);
  return fetchArtifactWithPresignedUrl({ runUuid, path, artifactRootUri }, legacyArtifactLocation, getArtifactData);
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
  if (!credential?.url) {
    throw new ErrorWrapper('The server did not return a multipart upload credential.', 501);
  }

  try {
    const uploadResponse = await uploadToPresignedUrl(credential.url, credential.headers ?? {}, body);
    const completeData = {
      path: filename,
      upload_id: createResponse.upload_id,
      parts: [
        {
          part_number: credential.part_number,
          etag: uploadResponse.headers.get('ETag') ?? '',
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
): Promise<boolean> => {
  const artifactsPresignedOnly = getArtifactsPresignedOnlySync();
  const multipartUploadsEnabled = getMultipartUploadsEnabledSync();
  const presignedUploadRunIdSupported = getPresignedUploadRunIdSupportedSync();
  if (!artifactsPresignedOnly && !multipartUploadsEnabled && !presignedUploadRunIdSupported) {
    return false;
  }
  const artifactRootUri = await getRunArtifactRootUri(runUuid);
  const proxiedArtifactRoute = getProxiedArtifactRoute(artifactRootUri, path);

  try {
    if (proxiedArtifactRoute) {
      if (!multipartUploadsEnabled && !artifactsPresignedOnly && !proxiedArtifactRoute.artifactServiceBaseUrl) {
        return false;
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
    if (!artifactsPresignedOnly && canFallBackFromPresignedError(error)) {
      return false;
    }
    throw error;
  }
};
