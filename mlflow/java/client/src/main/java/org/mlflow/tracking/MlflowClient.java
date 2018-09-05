package org.mlflow.tracking;

import org.apache.http.client.utils.URIBuilder;

import org.mlflow.api.proto.Service.*;
import org.mlflow.artifacts.ArtifactRepository;
import org.mlflow.artifacts.ArtifactRepositoryFactory;
import org.mlflow.tracking.creds.*;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Client to an MLflow Tracking Sever.
 */
public class MlflowClient {
  private static final long DEFAULT_EXPERIMENT_ID = 0;

  private final MlflowProtobufMapper mapper = new MlflowProtobufMapper();
  private final ArtifactRepositoryFactory artifactRepositoryFactory;
  private final MlflowHttpCaller httpCaller;
  private final MlflowHostCredsProvider hostCredsProvider;

  /** Returns a default client based on the MLFLOW_TRACKING_URI environment variable. */
  public MlflowClient() {
    this(getDefaultTrackingUri());
  }

  /** Instantiates a new client using the provided tracking uri. */
  public MlflowClient(String trackingUri) {
    this(getHostCredsProviderFromTrackingUri(trackingUri));
  }

  /**
   * Creates a new MlflowClient; users should prefer constructing ApiClients via
   * {@link #MlflowClient()} or {@link #MlflowClient(String)} if possible.
   */
  public MlflowClient(MlflowHostCredsProvider hostCredsProvider) {
    this.hostCredsProvider = hostCredsProvider;
    this.httpCaller = new MlflowHttpCaller(hostCredsProvider);
    this. artifactRepositoryFactory = new ArtifactRepositoryFactory(hostCredsProvider);
  }

  /** @return run associated with the id. */
  public Run getRun(String runUuid) {
    URIBuilder builder = newURIBuilder("runs/get").setParameter("run_uuid", runUuid);
    return mapper.toGetRunResponse(httpCaller.get(builder.toString())).getRun();
  }

  /**
   * Creates a new run under the default experiment with no application name.
   * @return RunInfo created by the server
   */
  public RunInfo createRun() {
    return createRun(DEFAULT_EXPERIMENT_ID);
  }

  /**
   * Creates a new run under the given experiment with no application name.
   * @return RunInfo created by the server
   */
  public RunInfo createRun(long experimentId) {
    return createRun(experimentId, "Java Application");
  }

  /**
   * Creates a new run under the given experiment with the given application name.
   * @return RunInfo created by the server
   */
  public RunInfo createRun(long experimentId, String appName) {
    CreateRun.Builder request = CreateRun.newBuilder();
    request.setExperimentId(experimentId);
    request.setSourceName(appName);
    request.setSourceType(SourceType.LOCAL);
    request.setStartTime(System.currentTimeMillis());
    String username = System.getProperty("user.name");
    if (username != null) {
      request.setUserId(System.getProperty("user.name"));
    }
    return createRun(request.build());
  }

  /**
   * Creates a new run.
   * @return RunInfo created by the server
   */
  public RunInfo createRun(CreateRun request) {
    String ijson = mapper.toJson(request);
    String ojson = sendPost("runs/create", ijson);
    return mapper.toCreateRunResponse(ojson).getRun().getInfo();
  }

  /** @return  a list of all RunInfos associated with the given experiment. */
  public List<RunInfo> listRunInfos(long experimentId) {
    SearchRuns request = SearchRuns.newBuilder().addExperimentIds(experimentId).build();
    String ijson = mapper.toJson(request);
    String ojson = sendPost("runs/search", ijson);
    return mapper.toSearchRunsResponse(ojson).getRunsList().stream().map(Run::getInfo)
      .collect(Collectors.toList());
  }

  /** @return  a list of all Experiments. */
  public List<Experiment> listExperiments() {
    return mapper.toListExperimentsResponse(httpCaller.get("experiments/list"))
      .getExperimentsList();
  }

  /** @return  an experiment with the given id. */
  public GetExperiment.Response getExperiment(long experimentId) {
    URIBuilder builder = newURIBuilder("experiments/get")
      .setParameter("experiment_id", "" + experimentId);
    return mapper.toGetExperimentResponse(httpCaller.get(builder.toString()));
  }

  /** @return  the experiment associated with the given name or Optional.empty if none exists. */
  public Optional<Experiment> getExperimentByName(String experimentName) {
    return listExperiments().stream().filter(e -> e.getName()
      .equals(experimentName)).findFirst();
  }

  /**
   * Creates a new experiment using the default artifact location provided by the server.
   * @param experimentName Name of the experiment. This must be unique across all experiments.
   * @return experiment id of the newly created experiment.
   */
  public long createExperiment(String experimentName) {
    String ijson = mapper.makeCreateExperimentRequest(experimentName);
    String ojson = httpCaller.post("experiments/create", ijson);
    return mapper.toCreateExperimentResponse(ojson).getExperimentId();
  }

  /**
   * Logs a parameter against the given run, as a key-value pair.
   * This cannot be called against the same parameter key more than once.
   */
  public void logParam(String runUuid, String key, String value) {
    sendPost("runs/log-parameter", mapper.makeLogParam(runUuid, key, value));
  }

  /**
   * Logs a new metric against the given run, as a key-value pair.
   * New values for the same metric may be recorded over time, and are marked with a timestamp.
   * */
  public void logMetric(String runUuid, String key, float value) {
    sendPost("runs/log-metric", mapper.makeLogMetric(runUuid, key, value,
      System.currentTimeMillis()));
  }

  /** Sets the status of a run to be FINISHED at the current time. */
  public void setTerminated(String runUuid) {
    setTerminated(runUuid, RunStatus.FINISHED);
  }

  /** Sets the status of a run to be completed at the current time. */
  public void setTerminated(String runUuid, RunStatus status) {
    setTerminated(runUuid, status, System.currentTimeMillis());
  }

  /** Sets the status of a run to be completed at the given endTime. */
  public void setTerminated(String runUuid, RunStatus status, long endTime) {
    sendPost("runs/update", mapper.makeUpdateRun(runUuid, status, endTime));
  }

  /**
   * Send a GET to the following path, including query parameters.
   * This is mostly an internal API, but allows making lower-level or unsupported requests.
   * @return JSON response from the server
   */
  public String sendGet(String path) {
    return httpCaller.get(path);
  }

  /**
   * Send a POST to the following path, with a String-encoded JSON body.
   * This is mostly an internal API, but allows making lower-level or unsupported requests.
   * @return JSON response from the server
   */
  public String sendPost(String path, String json) {
    return httpCaller.post(path, json);
  }

  /**
   * @return HostCredsProvider backing this MlflowClient. Visible for testing.
   */
  MlflowHostCredsProvider getInternalHostCredsProvider() {
    return hostCredsProvider;
  }

  private URIBuilder newURIBuilder(String base) {
    try {
      return new URIBuilder(base);
    } catch (URISyntaxException e) {
      throw new MlflowClientException("Failed to construct URI for " + base, e);
    }
  }

  /**
   * Returns the tracking URI from MLFLOW_TRACKING_URI or throws if not available.
   * This is used as the body of the no-argument constructor, as constructors must first call
   * this().
   */
  private static String getDefaultTrackingUri() {
    String defaultTrackingUri = System.getenv("MLFLOW_TRACKING_URI");
    if (defaultTrackingUri == null) {
      throw new IllegalStateException("Default client requires MLFLOW_TRACKING_URI is set." +
        " Use fromTrackingUri() instead.");
    }
    return defaultTrackingUri;
  }

  /**
   * Returns the MlflowHostCredsProvider associated with the given tracking URI.
   * This is used as the body of the String-argument constructor, as constructors must first call
   * this().
   */
  private static MlflowHostCredsProvider getHostCredsProviderFromTrackingUri(String trackingUri) {
    URI uri = URI.create(trackingUri);
    MlflowHostCredsProvider provider;

    if ("http".equals(uri.getScheme()) || "https".equals(uri.getScheme())) {
      provider = new BasicMlflowHostCreds(trackingUri);
    } else if (trackingUri.equals("databricks")) {
      MlflowHostCredsProvider profileProvider = new DatabricksConfigHostCredsProvider();
      MlflowHostCredsProvider dynamicProvider =
        DatabricksDynamicHostCredsProvider.createIfAvailable();
      if (dynamicProvider != null) {
        provider = new HostCredsProviderChain(dynamicProvider, profileProvider);
      } else {
        provider = profileProvider;
      }
    } else if ("databricks".equals(uri.getScheme())) {
      provider = new DatabricksConfigHostCredsProvider(uri.getHost());
    } else if (uri.getScheme() == null || "file".equals(uri.getScheme())) {
      throw new IllegalArgumentException("Java Client currently does not support" +
        " local tracking URIs. Please point to a Tracking Server.");
    } else {
      throw new IllegalArgumentException("Invalid tracking server uri: " + trackingUri);
    }
    return provider;
  }

  /**
   * Returns an {@link ArtifactRepository} capable of uploading and downloading MLflow artifacts
   * under the given base artifact URI.
   *
   * @param baseArtifactUri Artifact URI of an MLflow run (e.g., s3://bucket/0/12345/artifacts).
   *                        This is part of {@link RunInfo#getArtifactUri()}.
   * @return ArtifactRepository, capable of uploading and downloading MLflow artifacts.
   */
  public ArtifactRepository getArtifactRepository(URI baseArtifactUri) {
    return artifactRepositoryFactory.getArtifactRepository(baseArtifactUri);
  }

  /**
   * Returns an {@link ArtifactRepository} capable of uploading and downloading MLflow artifacts
   * for the given MLflow Run.
   *
   * @param runId Run ID of an existing MLflow run.
   * @return ArtifactRepository, capable of uploading and downloading MLflow artifacts.
   */
  public ArtifactRepository getArtifactRepositoryForRun(String runId) {
    URI baseArtifactUri = URI.create(getRun(runId).getInfo().getArtifactUri());
    return getArtifactRepository(baseArtifactUri);
  }
}
