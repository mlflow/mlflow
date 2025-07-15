/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { ErrorWrapper } from './ErrorWrapper';
import { getDefaultHeaders, HTTPMethods } from './FetchUtils';

/**
 * Async function to fetch and return the specified artifact blob from response.
 * Throw exception if the request fails.
 */
export async function getArtifactBlob(artifactLocation: any) {
  const getArtifactRequest = new Request(artifactLocation, {
    method: HTTPMethods.GET,
    redirect: 'follow',
    // TODO: fix types
    headers: new Headers(getDefaultHeaders(document.cookie) as any),
  });
  // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
  const response = await fetch(getArtifactRequest);

  if (!response.ok) {
    const errorMessage = (await response.text()) || response.statusText;
    throw new ErrorWrapper(errorMessage, response.status);
  }
  return response.blob();
}

class TextArtifactTooLargeError extends Error {}

/**
 * Async function to fetch and return the specified text artifact.
 * Avoids unnecessary conversion to blob, parses chunked responses directly to text.
 */
export const getArtifactChunkedText = async (artifactLocation: string) =>
  new Promise<string>(async (resolve, reject) => {
    const getArtifactRequest = new Request(artifactLocation, {
      method: HTTPMethods.GET,
      redirect: 'follow',
      headers: new Headers(getDefaultHeaders(document.cookie) as HeadersInit),
    });
    // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
    const response = await fetch(getArtifactRequest);

    if (!response.ok) {
      const errorMessage = (await response.text()) || response.statusText;
      reject(new ErrorWrapper(errorMessage, response.status));
      return;
    }
    const reader = response.body?.getReader();

    if (reader) {
      let resultData = '';
      const decoder = new TextDecoder();
      const appendChunk = async (result: ReadableStreamReadResult<Uint8Array>) => {
        const decodedChunk = decoder.decode(result.value || new Uint8Array(), {
          stream: !result.done,
        });
        resultData += decodedChunk;
        if (result.done) {
          resolve(resultData);
        } else {
          reader.read().then(appendChunk).catch(reject);
        }
      };

      reader.read().then(appendChunk).catch(reject);
    } else {
      reject(new Error("Can't get artifact data from the server"));
    }
  });

/**
 * Fetches the specified artifact, returning a Promise that resolves with
 * the raw content converted to text of the artifact if the fetch is
 * successful, and rejects otherwise
 */
export function getArtifactContent<R = unknown>(artifactLocation: string, isBinary = false): Promise<R> {
  return new Promise<R>(async (resolve, reject) => {
    try {
      const blob = await getArtifactBlob(artifactLocation);

      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        // Resolve promise with artifact contents
        // @ts-expect-error TS(2531): Object is possibly 'null'.
        resolve(event.target.result);
      };
      fileReader.onerror = (error) => {
        reject(error);
      };
      if (isBinary) {
        fileReader.readAsArrayBuffer(blob);
      } else {
        fileReader.readAsText(blob);
      }
    } catch (error) {
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      console.error(error);
      reject(error);
    }
  });
}

/**
 * Fetches the specified artifact, returning a Promise that resolves with
 * the raw content in bytes of the artifact if the fetch is successful, and rejects otherwise
 */
export function getArtifactBytesContent(artifactLocation: any) {
  return getArtifactContent(artifactLocation, true);
}

export const getLoggedModelArtifactLocationUrl = (path: string, loggedModelId: string) => {
  return `ajax-api/2.0/mlflow/logged-models/${loggedModelId}/artifacts/files?artifact_file_path=${encodeURIComponent(
    path,
  )}`;
};

export const getArtifactLocationUrl = (path: string, runUuid: string) => {
  const artifactEndpointPath = 'get-artifact';
  return `${artifactEndpointPath}?path=${encodeURIComponent(path)}&run_uuid=${encodeURIComponent(runUuid)}`;
};
