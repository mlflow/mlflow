import { getDefaultHeaders, HTTPMethods } from './FetchUtils';

/**
 * Fetches the specified artifact, returning a Promise that resolves with
 * the raw content converted to text of the artifact if the fetch is
 * successful, and rejects otherwise
 */
export function getArtifactContent(artifactLocation) {
  return new Promise((resolve, reject) => {
    const getArtifactRequest = new Request(artifactLocation, {
      method: HTTPMethods.GET,
      redirect: 'follow',
      headers: new Headers(getDefaultHeaders(document.cookie)),
    });
    fetch(getArtifactRequest)
      .then((response) => {
        return response.blob();
      })
      .then((blob) => {
        const fileReader = new FileReader();
        fileReader.onload = (event) => {
          // Resolve promise with artifact contents
          resolve(event.target.result);
        };
        fileReader.readAsText(blob);
      })
      .catch((error) => reject(error));
  });
}

/**
 * Fetches the specified artifact, returning a Promise that resolves with
 * the raw content in bytes of the artifact if the fetch is successful, and rejects otherwise
 */
export function getArtifactBytesContent(artifactLocation) {
  return new Promise((resolve, reject) => {
    const getArtifactRequest = new Request(artifactLocation, {
      method: HTTPMethods.GET,
      redirect: 'follow',
      headers: new Headers(getDefaultHeaders(document.cookie)),
    });
    fetch(getArtifactRequest)
      .then((response) => {
        return response.blob();
      })
      .then((blob) => {
        const fileReader = new FileReader();
        fileReader.onload = (event) => {
          // Resolve promise with artifact contents
          resolve(event.target.result);
        };
        fileReader.readAsArrayBuffer(blob);
      })
      .catch((error) => reject(error));
  });
}
