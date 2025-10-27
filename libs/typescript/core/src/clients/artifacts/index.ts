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
  databricksToken
}: {
  trackingUri: string;
  host: string;
  databricksToken?: string;
}): ArtifactsClient {
  if (trackingUri.startsWith('databricks')) {
    return new DatabricksArtifactsClient({ host, databricksToken });
  } else {
    return new MlflowArtifactsClient({ host });
  }
}

export { ArtifactsClient };
