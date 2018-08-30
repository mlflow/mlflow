package org.mlflow.tracking;

import org.apache.http.client.utils.URIBuilder;

import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.creds.*;

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
   * Creates a new MlflowClient. Users should prefer constructing ApiClients via
   * {@link #MlflowClient()} ()} or {@link #MlflowClient(String)} if possible.
   */
  public MlflowClient(MlflowHostCredsProvider hostCredsProvider) {
    this.hostCredsProvider = hostCredsProvider;
    this.httpCaller = new MlflowHttpCaller(hostCredsProvider);
  }

  /** Returns the run associated with the id. */
  public Run getRun(String runUuid) {
    URIBuilder builder = newURIBuilder("runs/get").setParameter("run_uuid", runUuid);
    return mapper.toGetRunResponse(httpCaller.get(builder.toString())).getRun();
  }

  /** Creates a new run under the default experiment with no application name. */
  public RunInfo createRun() {
    return createRun(DEFAULT_EXPERIMENT_ID);
  }

  /** Creates a new run under the given experiment with no application name. */
  public RunInfo createRun(long experimentId) {
    return createRun(experimentId, "Java Application");
  }

  /** Creates a new run under the given experiment with the given application name. */
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

  /** Creates a new run. */
  public RunInfo createRun(CreateRun request) {
    String ijson = mapper.toJson(request);
    String ojson = doPost("runs/create", ijson);
    return mapper.toCreateRunResponse(ojson).getRun().getInfo();
  }

  /** Returns a list of all RunInfos associated with the given experiment. */
  public List<RunInfo> listRunInfos(long experimentId) {
    SearchRuns request = SearchRuns.newBuilder().addExperimentIds(experimentId).build();
    String ijson = mapper.toJson(request);
    String ojson = doPost("runs/search", ijson);
    return mapper.toSearchRunsResponse(ojson).getRunsList().stream().map(Run::getInfo)
      .collect(Collectors.toList());
  }

  /** Returns a list of all Experiments. */
  public List<Experiment> listExperiments() {
    return mapper.toListExperimentsResponse(httpCaller.get("experiments/list"))
      .getExperimentsList();
  }

  /** Returns an experiment with the given id. */
  public GetExperiment.Response getExperiment(long experimentId) {
    URIBuilder builder = newURIBuilder("experiments/get")
      .setParameter("experiment_id", "" + experimentId);
    return mapper.toGetExperimentResponse(httpCaller.get(builder.toString()));
  }

  /** Returns the experiment associated with the given name or Optional.empty if none exists. */
  public Optional<Experiment> getExperimentByName(String experimentName) {
    return listExperiments().stream().filter(e -> e.getName()
      .equals(experimentName)).findFirst();
  }

  /** Creates a new experiment using the default artifact location provided by the server. */
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
    doPost("runs/log-parameter", mapper.makeLogParam(runUuid, key, value));
  }

  /**
   * Logs a new metric against the given run, as a key-value pair.
   * New values for the same metric may be recorded over time, and are marked with a timestamp.
   * */
  public void logMetric(String runUuid, String key, float value) {
    doPost("runs/log-metric", mapper.makeLogMetric(runUuid, key, value,
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
    doPost("runs/update", mapper.makeUpdateRun(runUuid, status, endTime));
  }

  /** Returns a list of all artifacts under the given artifact path within the run. */
  public ListArtifacts.Response listArtifacts(String runUuid, String path) {
    URIBuilder builder = newURIBuilder("artifacts/list")
      .setParameter("run_uuid", runUuid)
      .setParameter("path", path);
    return mapper.toListArtifactsResponse(httpCaller.get(builder.toString()));
  }

  /**
   * Send a GET to the following path, including query parameters.
   * This is mostly an internal API, but allows making lower-level or unsupported requests.
   */
  public String doGet(String path) {
    return httpCaller.get(path);
  }

  /**
   * Send a POST to the following path, with a String-encoded JSON body.
   * This is mostly an internal API, but allows making lower-level or unsupported requests.
   */
  public String doPost(String path, String json) {
    return httpCaller.post(path, json);
  }

  /**
   * Returns the HostCredsProvider backing this MlflowClient.
   * Intended for internal usage, and may be removed in future versions.
   */
  public MlflowHostCredsProvider getInternalHostCredsProvider() {
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
}
