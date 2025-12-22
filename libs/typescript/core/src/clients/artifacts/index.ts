import { ArtifactsClient } from './base';
import { DatabricksArtifactsClient } from './databricks';
import { MlflowArtifactsClient } from './mlflow';
import { AuthProvider } from '../../auth';

/**
 * Options for creating an ArtifactsClient.
 */
export interface ArtifactsClientOptions {
  /**
   * The tracking URI (e.g., "databricks", "http://localhost:5000")
   */
  trackingUri: string;

  /**
   * The resolved host URL for API requests
   */
  host: string;

  /**
   * Authentication provider
   */
  authProvider: AuthProvider;
}

/**
 * Get the appropriate artifacts client based on the tracking URI.
 *
 * @param trackingUri - The tracking URI (e.g., "databricks", "http://localhost:5000")
 * @param host - The resolved host URL for API requests
 * @param authProvider - The authentication provider to get tokens for authenticated requests
 * @returns The appropriate artifacts client.
 */
export function getArtifactsClient({
  trackingUri,
  host,
  authProvider
}: {
  trackingUri: string;
  host: string;
  authProvider: AuthProvider;
}): ArtifactsClient {
  if (trackingUri.startsWith('databricks')) {
    return new DatabricksArtifactsClient({ host, authProvider });
  } else {
    return new MlflowArtifactsClient({ host, authProvider });
  }
}

export { ArtifactsClient };
