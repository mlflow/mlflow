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
   * Authentication provider (new mode)
   */
  authProvider?: AuthProvider;

  /**
   * Databricks personal access token (legacy mode)
   * @deprecated Use authProvider instead
   */
  databricksToken?: string;
}

/**
 * Get the appropriate artifacts client based on the tracking URI.
 *
 * @param options - Options for creating the artifacts client
 * @returns The appropriate artifacts client.
 */
export function getArtifactsClient(options: ArtifactsClientOptions): ArtifactsClient {
  const { trackingUri, host, authProvider, databricksToken } = options;

  if (trackingUri.startsWith('databricks')) {
    return new DatabricksArtifactsClient({ host, authProvider, databricksToken });
  } else {
    return new MlflowArtifactsClient({ host, authProvider });
  }
}

export { ArtifactsClient };
