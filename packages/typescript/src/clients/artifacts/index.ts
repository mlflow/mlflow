import { ArtifactsClient } from './base';
import { MlflowArtifactsClient } from './mlflow';

/**
 * Get the appropriate artifacts client based on the tracking URI.
 *
 * @param trackingUri - The tracking URI to use to determine the artifacts client.
 * @returns The appropriate artifacts client.
 */
export function getArtifactsClient(host: string): ArtifactsClient {
  return new MlflowArtifactsClient({ host });
}

export { ArtifactsClient };
