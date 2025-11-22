import { ArtifactsClient } from './base';
import { DatabricksArtifactsClient } from './databricks';
import { MlflowArtifactsClient } from './mlflow';
import type { DatabricksAuthProvider } from '../../core/databricks_auth';

/**
 * Get the appropriate artifacts client based on the tracking URI.
 *
 * @param trackingUri - The tracking URI to use to determine the artifacts client.
 * @returns The appropriate artifacts client.
 */
export function getArtifactsClient({
  trackingUri,
  host,
  databricksToken,
  databricksAuthProvider
}: {
  trackingUri: string;
  host: string;
  databricksToken?: string;
  databricksAuthProvider?: DatabricksAuthProvider;
}): ArtifactsClient {
  if (trackingUri.startsWith('databricks')) {
    return new DatabricksArtifactsClient({ host, databricksToken, databricksAuthProvider });
  } else {
    return new MlflowArtifactsClient({ host });
  }
}

export { ArtifactsClient };
