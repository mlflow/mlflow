import { fileURLToPath } from 'node:url';

/**
 * Extract the scheme from an artifact URI.
 */
export function getArtifactUriScheme(uri: string): string {
  const match = /^([a-zA-Z][a-zA-Z0-9+.-]*):/.exec(uri);
  if (!match) {
    return '';
  }
  const scheme = match[1].toLowerCase();
  // A single-letter scheme is a Windows drive designator (e.g. `C:`), not a URI
  // scheme; treat it as a bare local path.
  if (scheme.length === 1) {
    return '';
  }
  return scheme;
}

/**
 * Whether an artifact URI refers to a location on the local filesystem.
 */
export function isLocalArtifactUri(uri: string): boolean {
  const scheme = getArtifactUriScheme(uri);
  return scheme === '' || scheme === 'file';
}

/**
 * Convert a local artifact URI to an absolute filesystem path.
 */
export function artifactUriToLocalPath(uri: string): string {
  if (getArtifactUriScheme(uri) === 'file') {
    return fileURLToPath(uri);
  }
  return uri;
}
