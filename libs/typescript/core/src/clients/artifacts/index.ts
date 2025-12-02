import { AuthProvider } from '../../auth/types';
import { ArtifactsClient } from './base';
import { DatabricksArtifactsClient } from './databricks';
import { MlflowArtifactsClient } from './mlflow';

/**
 * Get the appropriate artifacts client based on the tracking URI.
 *
 * @param trackingUri - The tracking URI to use to determine the artifacts client.
 * @returns The appropriate artifacts client.
 */
export function getArtifactsClient({
  trackingUri,
  host,
  authProvider,
  databricksToken
}: {
  trackingUri: string;
  host: string;
  authProvider?: AuthProvider;
  /** @deprecated Use authProvider instead */
  databricksToken?: string;
}): ArtifactsClient {
  if (trackingUri.startsWith('databricks')) {
    return new DatabricksArtifactsClient({ host, authProvider, databricksToken });
  } else {
    return new MlflowArtifactsClient({ host });
  }
}

export { ArtifactsClient };
