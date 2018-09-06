package org.mlflow.artifacts;

import java.net.URI;

import org.mlflow.tracking.creds.MlflowHostCredsProvider;

public class ArtifactRepositoryFactory {
  private final MlflowHostCredsProvider hostCredsProvider;

  public ArtifactRepositoryFactory(MlflowHostCredsProvider hostCredsProvider) {
    this.hostCredsProvider = hostCredsProvider;
  }

  public ArtifactRepository getArtifactRepository(URI baseArtifactUri, String runId) {
    return new CliBasedArtifactRepository(baseArtifactUri.toString(), runId, hostCredsProvider);
  }
}
