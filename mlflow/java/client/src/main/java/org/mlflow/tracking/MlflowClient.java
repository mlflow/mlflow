package org.mlflow.tracking;

import org.apache.http.client.utils.URIBuilder;

import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.creds.BasicMlflowHostCreds;
import org.mlflow.tracking.creds.MlflowHostCredsProvider;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Client to an MLflow Tracking Sever.
 */
public class MlflowClient {
  private final MlflowProtobufMapper mapper = new MlflowProtobufMapper();
  private final MlflowHttpCaller httpCaller;

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
    httpCaller = new MlflowHttpCaller(hostCredsProvider);
  }

  public Run getRun(String runUuid) {
    URIBuilder builder = newURIBuilder("runs/get").setParameter("run_uuid", runUuid);
    return mapper.toGetRunResponse(httpCaller.get(builder.toString())).getRun();
  }

  public RunInfo createRun(long experimentId, String sourceName) {
    CreateRun.Builder request = CreateRun.newBuilder();
    request.setExperimentId(experimentId);
    request.setSourceName(sourceName);
    request.setSourceType(SourceType.PROJECT);
    request.setStartTime(System.currentTimeMillis());
    String username = System.getProperty("user.name");
    if (username != null) {
      request.setUserId(System.getProperty("user.name"));
    }
    return createRun(request.build());
  }

  public RunInfo createRun(CreateRun request) {
    String ijson = mapper.toJson(request);
    String ojson = doPost("runs/create", ijson);
    return mapper.toCreateRunResponse(ojson).getRun().getInfo();
  }

  public List<RunInfo> listRunInfos(long experimentId) {
    SearchRuns request = SearchRuns.newBuilder().addExperimentIds(experimentId).build();
    String ijson = mapper.toJson(request);
    String ojson = doPost("runs/search", ijson);
    return mapper.toSearchRunsResponse(ojson).getRunsList().stream().map(Run::getInfo)
      .collect(Collectors.toList());
  }

  public List<Experiment> listExperiments() {
    return mapper.toListExperimentsResponse(httpCaller.get("experiments/list"))
      .getExperimentsList();
  }

  public GetExperiment.Response getExperiment(long experimentId) {
    URIBuilder builder = newURIBuilder("experiments/get")
      .setParameter("experiment_id", "" + experimentId);
    return mapper.toGetExperimentResponse(httpCaller.get(builder.toString()));
  }

  public long createExperiment(String experimentName) {
    String ijson = mapper.makeCreateExperimentRequest(experimentName);
    String ojson = httpCaller.post("experiments/create", ijson);
    return mapper.toCreateExperimentResponse(ojson).getExperimentId();
  }

  public void logParam(String runUuid, String key, String value) {
    doPost("runs/log-parameter", mapper.makeLogParam(runUuid, key, value));
  }

  public void logMetric(String runUuid, String key, float value) {
    doPost("runs/log-metric", mapper.makeLogMetric(runUuid, key, value));
  }

  public Metric getMetric(String runUuid, String metricKey) {
    URIBuilder builder = newURIBuilder("metrics/get")
      .setParameter("run_uuid", runUuid)
      .setParameter("metric_key", metricKey);
    return mapper.toGetMetricResponse(httpCaller.get(builder.toString())).getMetric();
  }

  public List<Metric> getMetricHistory(String runUuid, String metricKey) {
    URIBuilder builder = newURIBuilder("metrics/get-history")
      .setParameter("run_uuid", runUuid)
      .setParameter("metric_key", metricKey);
    return mapper.toGetMetricHistoryResponse(httpCaller.get(builder.toString())).getMetricsList();
  }

  public ListArtifacts.Response listArtifacts(String runUuid, String path) {
    URIBuilder builder = newURIBuilder("artifacts/list")
      .setParameter("run_uuid", runUuid)
      .setParameter("path", path);
    return mapper.toListArtifactsResponse(httpCaller.get(builder.toString()));
  }

  public byte[] getArtifact(String runUuid, String path) {
    throw new UnsupportedOperationException();
  }

  public Optional<Experiment> getExperimentByName(String experimentName) {
    return listExperiments().stream().filter(e -> e.getName()
      .equals(experimentName)).findFirst();
  }

  public void setTerminated(String runUuid, RunStatus status, long endTime) {
    doPost("runs/update", mapper.makeUpdateRun(runUuid, status, endTime));
  }

  public void setTerminated(String runUuid, RunStatus status) {
    setTerminated(runUuid, status, System.currentTimeMillis());
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
    } else if (uri.getScheme() == null || "file".equals(uri.getScheme())) {
      throw new IllegalArgumentException("Java Client currently does not support" +
        " local tracking URIs. Please point to a Tracking Server.");
    } else {
      throw new IllegalArgumentException("Invalid tracking server uri: " + trackingUri);
    }
    return provider;
  }
}
