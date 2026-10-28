import { ErrorWrapper } from './ErrorWrapper';
import { getAjaxUrl, getDefaultHeaders, HTTPMethods } from './FetchUtils';

export type ArtifactRequestOptions = {
  /** Replace tracking-server headers with only those required by object storage. */
  headers?: HeadersInit;
};

export const MAX_ARTIFACT_PREVIEW_BYTES = 50 * 1024 * 1024;

export class ArtifactPreviewTooLargeError extends Error {
  constructor() {
    super('File is too large to preview. Maximum file size for preview: 50 MiB.');
    this.name = 'ArtifactPreviewTooLargeError';
  }
}

async function fetchArtifactResponse(artifactLocation: string, options?: ArtifactRequestOptions): Promise<Response> {
  const request = new Request(artifactLocation, {
    method: HTTPMethods.GET,
    redirect: 'follow',
    headers: new Headers(
      options?.headers !== undefined ? options.headers : (getDefaultHeaders(document.cookie) as HeadersInit),
    ),
  });
  // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
  const response = await fetch(request);
  if (!response.ok) {
    const errorMessage = (await response.text()) || response.statusText;
    throw new ErrorWrapper(errorMessage, response.status);
  }
  return response;
}

/** Enforce the preview limit on bytes actually received unless a caller supplies another limit. */
async function readArtifactPreview(
  artifactLocation: string,
  options: ArtifactRequestOptions | undefined,
  onChunk: (chunk: Uint8Array) => void,
  maxBytes: number = MAX_ARTIFACT_PREVIEW_BYTES,
): Promise<Response> {
  const response = await fetchArtifactResponse(artifactLocation, options);
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Can't get artifact data from the server");
  }

  let bytesRead = 0;
  try {
    for (let result = await reader.read(); !result.done; result = await reader.read()) {
      const { value } = result;
      bytesRead += value.byteLength;
      if (bytesRead > maxBytes) {
        throw new ArtifactPreviewTooLargeError();
      }
      onChunk(value);
    }
    return response;
  } catch (error) {
    // Cancellation can itself fail after a transport error. Preserve the original failure.
    try {
      await reader.cancel();
    } catch {
      // The original read/size failure is the useful error for the caller.
    }
    throw error;
  } finally {
    reader.releaseLock();
  }
}

/** Fetch a blob for a preview, cancelling responses that exceed the preview limit. */
export async function getArtifactBlob(artifactLocation: string, options?: ArtifactRequestOptions): Promise<Blob> {
  const chunks: Uint8Array[] = [];
  const response = await readArtifactPreview(artifactLocation, options, (chunk) => chunks.push(chunk));
  return new Blob(chunks as BlobPart[], { type: response.headers.get('Content-Type') ?? '' });
}

/** Downloads may exceed the preview limit, including signed downloads that require headers. */
export async function getArtifactDownloadBlob(
  artifactLocation: string,
  options?: ArtifactRequestOptions,
): Promise<Blob> {
  const response = await fetchArtifactResponse(artifactLocation, options);
  return response.blob();
}

/** Decode text incrementally, preserving UTF-8 characters split across response chunks. */
export async function getArtifactChunkedText(
  artifactLocation: string,
  options?: ArtifactRequestOptions,
): Promise<string> {
  let text = '';
  const decoder = new TextDecoder();
  await readArtifactPreview(
    artifactLocation,
    options,
    (chunk) => {
      text += decoder.decode(chunk, { stream: true });
    },
    Number.POSITIVE_INFINITY,
  );
  return text + decoder.decode();
}

/** Fetch artifact text for a preview. */
export async function getArtifactContent(artifactLocation: string, options?: ArtifactRequestOptions): Promise<string> {
  const blob = await getArtifactBlob(artifactLocation, options);
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    // FileReader preserves BOM-based encoding detection after the byte limit is enforced.
    reader.readAsText(blob);
  });
}

/** Fetch artifact bytes for a preview without converting them to text. */
export async function getArtifactBytesContent(
  artifactLocation: string,
  options?: ArtifactRequestOptions,
): Promise<ArrayBuffer> {
  const chunks: Uint8Array[] = [];
  let bytesRead = 0;
  await readArtifactPreview(artifactLocation, options, (chunk) => {
    chunks.push(chunk);
    bytesRead += chunk.byteLength;
  });
  const bytes = new Uint8Array(bytesRead);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes.buffer;
}

export const getLoggedModelArtifactLocationUrl = (path: string, loggedModelId: string) => {
  return getAjaxUrl(
    `ajax-api/2.0/mlflow/logged-models/${loggedModelId}/artifacts/files?artifact_file_path=${encodeURIComponent(path)}`,
  );
};

export const getArtifactLocationUrl = (path: string, runUuid: string) => {
  const artifactEndpointPath = getAjaxUrl('get-artifact');
  return `${artifactEndpointPath}?path=${encodeURIComponent(path)}&run_uuid=${encodeURIComponent(runUuid)}`;
};
