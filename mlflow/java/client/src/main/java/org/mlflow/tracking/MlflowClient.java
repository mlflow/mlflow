package org.mlflow.tracking;

import com.google.common.collect.Lists;
import org.apache.http.client.utils.URIBuilder;
import org.mlflow.artifacts.ArtifactRepository;
import org.mlflow.artifacts.ArtifactRepositoryFactory;
import org.mlflow.artifacts.CliBasedArtifactRepository;
import org.mlflow.api.proto.ModelRegistry.*;
import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.creds.*;

import java.io.Closeable;
import java.io.File;
import java.io.Serializable;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Client to an MLflow Tracking Sever.
 */
public class MlflowClient implements Serializable, Closeable {
  protected static final String DEFAULT_EXPERIMENT_ID = "0";
  private static final String DEFAULT_MODELS_ARTIFACT_REPOSITORY_SCHEME = "models";

  private final MlflowProtobufMapper mapper = new MlflowProtobufMapper();
  private final ArtifactRepositoryFactory artifactRepositoryFactory;
  private final MlflowHttpCaller httpCaller;
  private final MlflowHostCredsProvider hostCredsProvider;

  /** Return a default client based on the MLFLOW_TRACKING_URI environment variable. */
  public MlflowClient() {
    this(getDefaultTrackingUri());
  }

  /** Instantiate a new client using the provided tracking uri. */
  public MlflowClient(String trackingUri) {
    this(getHostCredsProviderFromTrackingUri(trackingUri));
  }

  /**
   * Create a new MlflowClient; users should prefer constructing ApiClients via
   * {@link #MlflowClient()} or {@link #MlflowClient(String)} if possible.
   */
  public MlflowClient(MlflowHostCredsProvider hostCredsProvider) {
    this.hostCredsProvider = hostCredsProvider;
    this.httpCaller = new MlflowHttpCaller(hostCredsProvider);
    this.artifactRepositoryFactory = new ArtifactRepositoryFactory(hostCredsProvider);
  }

  /**
   * Get metadata, params, tags, and metrics for a run. A single value is returned for each metric
   * key: the most recently logged metric value at the largest step.
   *
   * @return Run associated with the ID.
   */
  public Run getRun(String runId) {
    URIBuilder builder = newURIBuilder("runs/get")
      .setParameter("run_uuid", runId)
      .setParameter("run_id", runId);
    return mapper.toGetRunResponse(httpCaller.get(builder.toString())).getRun();
  }

  public List<Metric> getMetricHistory(String runId, String key) {
    URIBuilder builder = newURIBuilder("metrics/get-history")
      .setParameter("run_uuid", runId)
      .setParameter("run_id", runId)
      .setParameter("metric_key", key)
      .setParameter("max_results", "25000");

    GetMetricHistory.Response response = mapper
            .toGetMetricHistoryResponse(httpCaller.get(builder.toString()));
    List<Metric> metrics = new ArrayList<>(response.getMetricsList());
    String token = response.getNextPageToken();
    while (!token.isEmpty()) {
      URIBuilder bld = builder.setParameter("page_token", token);
      GetMetricHistory.Response resp = mapper
              .toGetMetricHistoryResponse(httpCaller.get(bld.toString()));
      metrics.addAll(resp.getMetricsList());
      token = resp.getNextPageToken();
    }
    return metrics;
  }

  /**
   * Create a new run under the default experiment with no application name.
   * @return RunInfo created by the server.
   */
  public RunInfo createRun() {
    return createRun(DEFAULT_EXPERIMENT_ID);
  }

  /**
   * Create a new run under the given experiment.
   * @return RunInfo created by the server.
   */
  public RunInfo createRun(String experimentId) {
    CreateRun.Builder request = CreateRun.newBuilder();
    request.setExperimentId(experimentId);
    request.setStartTime(System.currentTimeMillis());
    // userId is deprecated and will be removed in a future release.
    // It should be set as the `mlflow.user` tag instead.
    String username = System.getProperty("user.name");
    if (username != null) {
      request.setUserId(System.getProperty("user.name"));
    }
    return createRun(request.build());
  }

  /**
   * Create a new run. This method allows providing all possible fields of CreateRun, and can be
   * invoked as follows:
   *
   *   <pre>
   *   import org.mlflow.api.proto.Service.CreateRun;
   *   CreateRun.Builder request = CreateRun.newBuilder();
   *   request.setExperimentId(experimentId);
   *   request.setSourceVersion("my-version");
   *   createRun(request.build());
   *   </pre>
   *
   * @return RunInfo created by the server.
   */
  public RunInfo createRun(CreateRun request) {
    String ijson = mapper.toJson(request);
    String ojson = sendPost("runs/create", ijson);
    return mapper.toCreateRunResponse(ojson).getRun().getInfo();
  }

  /**
   * @return A list of all RunInfos associated with the given experiment.
   */
  public List<RunInfo> listRunInfos(String experimentId) {
    List<String> experimentIds = new ArrayList<>();
    experimentIds.add(experimentId);
    return searchRuns(experimentIds, null);
  }

  /**
   * Return RunInfos from provided list of experiments that satisfy the search query.
   * @deprecated As of 1.1.0 - please use {@link #searchRuns(List, String, ViewType, int)} or
   *                    similar that returns a page of Run results.
   *
   * @param experimentIds List of experiment IDs.
   * @param searchFilter SQL compatible search query string. Format of this query string is
   *                     similar to that specified on MLflow UI.
   *                     Example : "params.model = 'LogisticRegression' and metrics.acc = 0.9"
   *                     If null, the result will be equivalent to having an empty search filter.
   *
   * @return A list of all RunInfos that satisfy search filter.
   */
  public List<RunInfo> searchRuns(List<String> experimentIds, String searchFilter) {
    return searchRuns(experimentIds, searchFilter, ViewType.ACTIVE_ONLY, 1000).getItems().stream()
      .map(Run::getInfo).collect(Collectors.toList());
  }

  /**
   * Return RunInfos from provided list of experiments that satisfy the search query.
   * @deprecated As of 1.1.0 - please use {@link #searchRuns(List, String, ViewType, int)} or
   *                    similar that returns a page of Run results.
   *
   * @param experimentIds List of experiment IDs.
   * @param searchFilter SQL compatible search query string. Format of this query string is
   *                     similar to that specified on MLflow UI.
   *                     Example : "params.model = 'LogisticRegression' and metrics.acc != 0.9"
   *                     If null, the result will be equivalent to having an empty search filter.
   * @param runViewType ViewType for expected runs. One of (ACTIVE_ONLY, DELETED_ONLY, ALL)
   *                    If null, only runs with viewtype ACTIVE_ONLY will be searched.
   *
   * @return A list of all RunInfos that satisfy search filter.
   */
  public List<RunInfo> searchRuns(List<String> experimentIds,
                              String searchFilter,
                              ViewType runViewType) {
    return searchRuns(experimentIds, searchFilter, runViewType, 1000).getItems().stream()
      .map(Run::getInfo).collect(Collectors.toList());
  }

  /**
   * Return runs from provided list of experiments that satisfy the search query.
   *
   * @param experimentIds List of experiment IDs.
   * @param searchFilter SQL compatible search query string. Format of this query string is
   *                     similar to that specified on MLflow UI.
   *                     Example : "params.model = 'LogisticRegression' and metrics.acc != 0.9"
   *                     If null, the result will be equivalent to having an empty search filter.
   * @param runViewType ViewType for expected runs. One of (ACTIVE_ONLY, DELETED_ONLY, ALL)
   *                    If null, only runs with viewtype ACTIVE_ONLY will be searched.
   * @param maxResults Maximum number of runs desired in one page.
   *
   * @return A list of all Runs that satisfy search filter.
   */
  public RunsPage searchRuns(List<String> experimentIds,
                              String searchFilter,
                              ViewType runViewType,
                              int maxResults) {
    return searchRuns(experimentIds, searchFilter, runViewType, maxResults, new ArrayList<>(),
                      null);
  }

  /**
   * Return runs from provided list of experiments that satisfy the search query.
   *
   * @param experimentIds List of experiment IDs.
   * @param searchFilter SQL compatible search query string. Format of this query string is
   *                     similar to that specified on MLflow UI.
   *                     Example : "params.model = 'LogisticRegression' and metrics.acc != 0.9"
   *                     If null, the result will be equivalent to having an empty search filter.
   * @param runViewType ViewType for expected runs. One of (ACTIVE_ONLY, DELETED_ONLY, ALL)
   *                    If null, only runs with viewtype ACTIVE_ONLY will be searched.
   * @param maxResults Maximum number of runs desired in one page.
   * @param orderBy List of properties to order by. Example: "metrics.acc DESC".
   *
   * @return A list of all Runs that satisfy search filter.
   */
  public RunsPage searchRuns(List<String> experimentIds,
                              String searchFilter,
                              ViewType runViewType,
                              int maxResults,
                              List<String> orderBy) {
    return searchRuns(experimentIds, searchFilter, runViewType, maxResults, orderBy, null);
  }

  /**
   * Return runs from provided list of experiments that satisfy the search query.
   *
   * @param experimentIds List of experiment IDs.
   * @param searchFilter SQL compatible search query string. Format of this query string is
   *                     similar to that specified on MLflow UI.
   *                     Example : "params.model = 'LogisticRegression' and metrics.acc != 0.9"
   *                     If null, the result will be equivalent to having an empty search filter.
   * @param runViewType ViewType for expected runs. One of (ACTIVE_ONLY, DELETED_ONLY, ALL)
   *                    If null, only runs with viewtype ACTIVE_ONLY will be searched.
   * @param maxResults Maximum number of runs desired in one page.
   * @param orderBy List of properties to order by. Example: "metrics.acc DESC".
   * @param pageToken String token specifying the next page of results. It should be obtained from
   *             a call to {@link #searchRuns(List, String)}.
   *
   * @return A page of Runs that satisfy the search filter.
   */
  public RunsPage searchRuns(List<String> experimentIds,
                              String searchFilter,
                              ViewType runViewType,
                              int maxResults,
                              List<String> orderBy,
                              String pageToken) {
    SearchRuns.Builder builder = SearchRuns.newBuilder()
            .addAllExperimentIds(experimentIds)
            .addAllOrderBy(orderBy)
            .setMaxResults(maxResults);

    if (searchFilter != null) {
      builder.setFilter(searchFilter);
    }
    if (runViewType != null) {
      builder.setRunViewType(runViewType);
    }
    if (pageToken != null) {
      builder.setPageToken(pageToken);
    }
    SearchRuns request = builder.build();
    String ijson = mapper.toJson(request);
    String ojson = sendPost("runs/search", ijson);
    SearchRuns.Response response = mapper.toSearchRunsResponse(ojson);
    return new RunsPage(response.getRunsList(), response.getNextPageToken(), experimentIds,
      searchFilter, runViewType, maxResults, orderBy, this);
  }

  /**
   * Return experiments that satisfy the search query.
   *
   * @param searchFilter SQL compatible search query string.
   *                     Examples:
   *                         - "attribute.name = 'MyExperiment'"
   *                         - "tags.problem_type = 'iris_regression'"
   *                     If null, the result will be equivalent to having an empty search filter.
   * @param experimentViewType ViewType for expected experiments. One of
   *                           (ACTIVE_ONLY, DELETED_ONLY, ALL). If null, only experiments with
   *                           viewtype ACTIVE_ONLY will be searched.
   * @param maxResults Maximum number of experiments desired in one page.
   * @param orderBy List of properties to order by. Example: "metrics.acc DESC".
   *
   * @return A page of experiments that satisfy the search filter.
   */
  public ExperimentsPage searchExperiments(String searchFilter,
                                           ViewType experimentViewType,
                                           int maxResults,
                                           List<String> orderBy) {
    return searchExperiments(searchFilter, experimentViewType, maxResults, orderBy, null);
  }

  /**
   * Return up to 1000 active experiments.
   *
   * @return A page of active experiments with up to 1000 items.
   */
  public ExperimentsPage searchExperiments() {
    return searchExperiments("", null, 1000, new ArrayList<>(), null);
  }

  /**
   * Return up to the first 1000 active experiments that satisfy the search query.
   *
   * @param searchFilter SQL compatible search query string.
   *                     Examples:
   *                         - "attribute.name = 'MyExperiment'"
   *                         - "tags.problem_type = 'iris_regression'"
   *                     If null, the result will be equivalent to having an empty search filter.
   *
   * @return A page of up to active 1000 experiments that satisfy the search filter.
   */
  public ExperimentsPage searchExperiments(String searchFilter) {
    return searchExperiments(searchFilter, null, 1000, new ArrayList<>(), null);
  }

  /**
   * Return experiments that satisfy the search query.
   *
   * @param searchFilter SQL compatible search query string.
   *                     Examples:
   *                         - "attribute.name = 'MyExperiment'"
   *                         - "tags.problem_type = 'iris_regression'"
   *                     If null, the result will be equivalent to having an empty search filter.
   * @param experimentViewType ViewType for expected experiments. One of
   *                           (ACTIVE_ONLY, DELETED_ONLY, ALL). If null, only experiments with
   *                           viewtype ACTIVE_ONLY will be searched.
   * @param maxResults Maximum number of experiments desired in one page.
   * @param orderBy List of properties to order by. Example: "metrics.acc DESC".
   * @param pageToken String token specifying the next page of results. It should be obtained from
   *             a call to {@link #searchExperiments(String)}.
   *
   * @return A page of experiments that satisfy the search filter.
   */
  public ExperimentsPage searchExperiments(String searchFilter,
                                           ViewType experimentViewType,
                                           int maxResults,
                                           List<String> orderBy,
                                           String pageToken) {
    SearchExperiments.Builder builder = SearchExperiments.newBuilder()
            .addAllOrderBy(orderBy)
            .setMaxResults(maxResults);

    if (searchFilter != null) {
      builder.setFilter(searchFilter);
    }
    if (experimentViewType != null) {
      builder.setViewType(experimentViewType);
    } else {
      builder.setViewType(ViewType.ACTIVE_ONLY);
    }
    if (pageToken != null) {
      builder.setPageToken(pageToken);
    }
    SearchExperiments request = builder.build();
    String ijson = mapper.toJson(request);
    String ojson = sendPost("experiments/search", ijson);
    SearchExperiments.Response response = mapper.toSearchExperimentsResponse(ojson);
    return new ExperimentsPage(response.getExperimentsList(), response.getNextPageToken(),
      searchFilter, experimentViewType, maxResults, orderBy, this);
  }

  /** @return  An experiment with the given ID. */
  public Experiment getExperiment(String experimentId) {
    URIBuilder builder = newURIBuilder("experiments/get")
      .setParameter("experiment_id", experimentId);
    return mapper.toGetExperimentResponse(httpCaller.get(builder.toString())).getExperiment();
  }

  /** @return  The experiment associated with the given name or Optional.empty if none exists. */
  public Optional<Experiment> getExperimentByName(String experimentName) {
    URIBuilder builder = newURIBuilder("experiments/get-by-name")
      .setParameter("experiment_name", experimentName);
    try {
      return Optional.of(
          mapper.toGetExperimentByNameResponse(httpCaller.get(builder.toString())).getExperiment()
      );
    } catch (MlflowHttpException e) {
      if (e.getStatusCode() == 404) {
        return Optional.<Experiment>empty();
      } else {
        throw e;
      }
    }
  }

  /**
   * Create a new experiment using the default artifact location provided by the server.
   * @param experimentName Name of the experiment. This must be unique across all experiments.
   * @return Experiment ID of the newly created experiment.
   */
  public String createExperiment(String experimentName) {
    String ijson = mapper.makeCreateExperimentRequest(experimentName);
    String ojson = httpCaller.post("experiments/create", ijson);
    return mapper.toCreateExperimentResponse(ojson).getExperimentId();
  }

  /**
   * Create a new experiment. This method allows providing all possible
   * fields of CreateExperiment, and can be invoked as follows:
   *
   *   <pre>
   *   import org.mlflow.api.proto.Service.CreateExperiment;
   *   CreateExperiment.Builder request = CreateExperiment.newBuilder();
   *   request.setName(name);
   *   request.setArtifactLocation(artifactLocation);
   *   request.addTags(experimentTag);
   *   createExperiment(request.build());
   *   </pre>
   *
   * @return ID of the experiment created by the server.
   */
  public String createExperiment(CreateExperiment request) {
    String ijson = mapper.toJson(request);
    String ojson = sendPost("experiments/create", ijson);
    return mapper.toCreateExperimentResponse(ojson).getExperimentId();
  }

  /** Mark an experiment and associated runs, params, metrics, etc. for deletion. */
  public void deleteExperiment(String experimentId) {
    String ijson = mapper.makeDeleteExperimentRequest(experimentId);
    httpCaller.post("experiments/delete", ijson);
  }

  /** Restore an experiment marked for deletion. */
  public void restoreExperiment(String experimentId) {
    String ijson = mapper.makeRestoreExperimentRequest(experimentId);
    httpCaller.post("experiments/restore", ijson);
  }

  /** Update an experiment's name. The new name must be unique. */
  public void renameExperiment(String experimentId, String newName) {
    String ijson = mapper.makeUpdateExperimentRequest(experimentId, newName);
    httpCaller.post("experiments/update", ijson);
  }

  /**
   * Delete a run with the given ID.
   */
  public void deleteRun(String runId) {
    String ijson = mapper.makeDeleteRun(runId);
    httpCaller.post("runs/delete", ijson);
  }

  /**
   * Restore a deleted run with the given ID.
   */
  public void restoreRun(String runId) {
    String ijson = mapper.makeRestoreRun(runId);
    httpCaller.post("runs/restore", ijson);
  }

  /**
   * Log a parameter against the given run, as a key-value pair.
   * This cannot be called against the same parameter key more than once.
   */
  public void logParam(String runId, String key, String value) {
    sendPost("runs/log-parameter", mapper.makeLogParam(runId, key, value));
  }

  /**
   * Log a new metric against the given run, as a key-value pair. Metrics are recorded
   * against two axes: timestamp and step. This method uses the number of milliseconds
   * since the Unix epoch for the timestamp, and it uses the default step of zero.
   *
   * @param runId The ID of the run in which to record the metric.
   * @param key The key identifying the metric for which to record the specified value.
   * @param value The value of the metric.
   */
  public void logMetric(String runId, String key, double value) {
    logMetric(runId, key, value, System.currentTimeMillis(), 0);
  }

  /**
   * Log a new metric against the given run, as a key-value pair. Metrics are recorded
   * against two axes: timestamp and step.
   *
   * @param runId The ID of the run in which to record the metric.
   * @param key The key identifying the metric for which to record the specified value.
   * @param value The value of the metric.
   * @param timestamp The timestamp at which to record the metric value.
   * @param step The step at which to record the metric value.
   */
  public void logMetric(String runId, String key, double value, long timestamp, long step) {
    sendPost("runs/log-metric", mapper.makeLogMetric(runId, key, value, timestamp, step));
  }

  /**
   * Log a new tag against the given experiment as a key-value pair.
   * @param experimentId The ID of the experiment on which to set the tag
   * @param key The key used to identify the tag.
   * @param value The value of the tag.
   */
  public void setExperimentTag(String experimentId, String key, String value) {
    sendPost("experiments/set-experiment-tag",
            mapper.makeSetExperimentTag(experimentId, key, value));
  }

  /**
   * Log a new tag against the given run, as a key-value pair.
   * @param runId The ID of the run on which to set the tag
   * @param key The key used to identify the tag.
   * @param value The value of the tag.
   */
  public void setTag(String runId, String key, String value) {
    sendPost("runs/set-tag", mapper.makeSetTag(runId, key, value));
  }

  /**
   * Delete a tag on the run ID with a specific key. This is irreversible.
   * @param runId String ID of the run
   * @param key Name of the tag
   */
  public void deleteTag(String runId, String key) {
    sendPost("runs/delete-tag", mapper.makeDeleteTag(runId, key));
  }

  /**
   * Log multiple metrics, params, and/or tags against a given run (argument runId).
   * Argument metrics, params, and tag iterables can be nulls.
   */
  public void logBatch(String runId,
      Iterable<Metric> metrics,
      Iterable<Param> params,
      Iterable<RunTag> tags) {
    sendPost("runs/log-batch", mapper.makeLogBatch(runId, metrics, params, tags));
  }

  /** Set the status of a run to be FINISHED at the current time. */
  public void setTerminated(String runId) {
    setTerminated(runId, RunStatus.FINISHED);
  }

  /** Set the status of a run to be completed at the current time. */
  public void setTerminated(String runId, RunStatus status) {
    setTerminated(runId, status, System.currentTimeMillis());
  }

  /** Set the status of a run to be completed at the given endTime. */
  public void setTerminated(String runId, RunStatus status, long endTime) {
    sendPost("runs/update", mapper.makeUpdateRun(runId, status, endTime));
  }

  /**
   * Send a GET to the following path, including query parameters.
   * This is mostly an internal API, but allows making lower-level or unsupported requests.
   * @return JSON response from the server.
   */
  public String sendGet(String path) {
    return httpCaller.get(path);
  }

  /**
   * Send a POST to the following path, with a String-encoded JSON body.
   * This is mostly an internal API, but allows making lower-level or unsupported requests.
   * @return JSON response from the server.
   */
  public String sendPost(String path, String json) {
    return httpCaller.post(path, json);
  }

  public String sendPatch(String path, String json) {
    return httpCaller.patch(path, json);
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
   * Return the tracking URI from MLFLOW_TRACKING_URI or throws if not available.
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
   * Return the MlflowHostCredsProvider associated with the given tracking URI.
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
   * Upload the given local file or directory to the run's root artifact directory. For example,
   *
   *   <pre>
   *   logArtifact(runId, "/my/localModel")
   *   listArtifacts(runId) // returns "localModel"
   *   </pre>
   *
   * @param runId Run ID of an existing MLflow run.
   * @param localFile File or directory to upload. Must exist.
   */
  public void logArtifact(String runId, File localFile) {
    if (localFile.isDirectory()) {
      getArtifactRepository(runId).logArtifacts(localFile, localFile.getName());
    }
    else {
      getArtifactRepository(runId).logArtifact(localFile);
    }
  }

  /**
   * Upload the given local file or directory to an artifactPath
   * within the run's root directory. For example,
   *
   *   <pre>
   *   logArtifact(runId, "/my/localModel", "model")
   *   listArtifacts(runId, "model") // returns "model/localModel"
   *   </pre>
   *
   * (i.e., the localModel file is now available in model/localModel).
   * If logging a directory, the directory is renamed to artifactPath.
   *
   * @param runId Run ID of an existing MLflow run.
   * @param localFile File or directory to upload. Must exist.
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  public void logArtifact(String runId, File localFile, String artifactPath) {
    if (localFile.isDirectory()) {
      getArtifactRepository(runId).logArtifacts(localFile, artifactPath);
    }
    else {
      getArtifactRepository(runId).logArtifact(localFile, artifactPath);
    }
  }

  /**
   * Upload all files within the given local directory the run's root artifact directory.
   * For example, if /my/local/dir/ contains two files "file1" and "file2", then
   *
   *   <pre>
   *   logArtifacts(runId, "/my/local/dir")
   *   listArtifacts(runId) // returns "file1" and "file2"
   *   </pre>
   *
   * @param runId Run ID of an existing MLflow run.
   * @param localDir Directory to upload. Must exist, and must be a directory (not a simple file).
   */
  public void logArtifacts(String runId, File localDir) {
    getArtifactRepository(runId).logArtifacts(localDir);
  }

  /**
   * Upload all files within the given local director an artifactPath within the run's root
   * artifact directory. For example, if /my/local/dir/ contains two files "file1" and "file2", then
   *
   *   <pre>
   *   logArtifacts(runId, "/my/local/dir", "model")
   *   listArtifacts(runId, "model") // returns "model/file1" and "model/file2"
   *   </pre>
   *
   * (i.e., the contents of the local directory are now available in model/).
   *
   * @param runId Run ID of an existing MLflow run.
   * @param localDir Directory to upload. Must exist, and must be a directory (not a simple file).
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  public void logArtifacts(String runId, File localDir, String artifactPath) {
    getArtifactRepository(runId).logArtifacts(localDir, artifactPath);
  }

  /**
   * List the artifacts immediately under the run's root artifact directory. This does not
   * recursively list; instead, it will return FileInfos with isDir=true where further
   * listing may be done.
   * @param runId Run ID of an existing MLflow run.
   */
  public List<FileInfo> listArtifacts(String runId) {
    return getArtifactRepository(runId).listArtifacts();
  }

  /**
   * List the artifacts immediately under the given artifactPath within the run's root artifact
   * directory. This does not recursively list; instead, it will return FileInfos with isDir=true
   * where further listing may be done.
   * @param runId Run ID of an existing MLflow run.
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  public List<FileInfo> listArtifacts(String runId, String artifactPath) {
    return getArtifactRepository(runId).listArtifacts(artifactPath);
  }

  /**
   * Return a local directory containing *all* artifacts within the run's artifact directory.
   * Note that this will download the entire directory path, and so may be expensive if
   * the directory has a lot of data.
   * @param runId Run ID of an existing MLflow run.
   */
  public File downloadArtifacts(String runId) {
    return getArtifactRepository(runId).downloadArtifacts();
  }

  /**
   * Return a local file or directory containing all artifacts within the given artifactPath
   * within the run's root artifactDirectory. For example, if "model/file1" and "model/file2"
   * exist within the artifact directory, then
   *
   *   <pre>
   *   downloadArtifacts(runId, "model") // returns a local directory containing "file1" and "file2"
   *   downloadArtifacts(runId, "model/file1") // returns a local *file* with the contents of file1.
   *   </pre>
   *
   * Note that this will download the entire subdirectory path, and so may be expensive if
   * the subdirectory has a lot of data.
   *
   * @param runId Run ID of an existing MLflow run.
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  public File downloadArtifacts(String runId, String artifactPath) {
    return getArtifactRepository(runId).downloadArtifacts(artifactPath);
  }

  /**
   * @param runId Run ID of an existing MLflow run.
   * @return ArtifactRepository, capable of uploading and downloading MLflow artifacts.
   */
  private ArtifactRepository getArtifactRepository(String runId) {
    URI baseArtifactUri = URI.create(getRun(runId).getInfo().getArtifactUri());
    return artifactRepositoryFactory.getArtifactRepository(baseArtifactUri, runId);
  }


  // ********************
  // * Model Registry *
  // ********************

  /**
   * Return the latest model version for each stage.
   * The current available stages are: [None, Staging, Production, Archived].
   *
   *    <pre>
   *        import org.mlflow.api.proto.ModelRegistry.ModelVersion;
   *        List{@code <ModelVersion>} detailsList = getLatestVersions("model");
   *
   *        for (ModelVersion details : detailsList) {
   *            System.out.println("Model Name: " + details.getModelVersion()
   *                                                       .getRegisteredModel()
   *                                                       .getName());
   *            System.out.println("Model Version: " + details.getModelVersion().getVersion());
   *            System.out.println("Current Stage: " + details.getCurrentStage());
   *        }
   *    </pre>
   *
   * @param modelName The name of the model
   * @return A collection of {@link org.mlflow.api.proto.ModelRegistry.ModelVersion}
   */
  public List<ModelVersion> getLatestVersions(String modelName) {
      return getLatestVersions(modelName, Collections.emptyList());
  }

  /**
   * Return the latest model version for each stage requested.
   * The current available stages are: [None, Staging, Production, Archived].
   *
   *    <pre>
   *        import org.mlflow.api.proto.ModelRegistry.ModelVersion;
   *        List{@code <ModelVersion>} detailsList =
   *          getLatestVersions("model", Lists.newArrayList{@code <String>}("Staging"));
   *
   *        for (ModelVersion details : detailsList) {
   *            System.out.println("Model Name: " + details.getModelVersion()
   *                                                       .getRegisteredModel()
   *                                                       .getName());
   *            System.out.println("Model Version: " + details.getModelVersion().getVersion());
   *            System.out.println("Current Stage: " + details.getCurrentStage());
   *        }
   *    </pre>
   *
   * @param modelName The name of the model
   * @param stages A list of stages
   * @return The latest model version
   *         {@link org.mlflow.api.proto.ModelRegistry.ModelVersion}
   */
  public List<ModelVersion> getLatestVersions(String modelName, Iterable<String> stages) {
    String json = sendGet(mapper.makeGetLatestVersion(modelName, stages));
    GetLatestVersions.Response response =  mapper.toGetLatestVersionsResponse(json);
    return response.getModelVersionsList();
  }

  /**
   *
   *   <pre>
   *       import org.mlflow.api.proto.ModelRegistry.ModelVersion;
   *       ModelVersion modelVersion = getModelVersion("model", "version");
   *   </pre>
   *
   * @param modelName Name of the containing registered model. *
   * @param version Version number as a string of the model version.
   * @return a single model version
   *        {@link org.mlflow.api.proto.ModelRegistry.ModelVersion}
   */
  public ModelVersion getModelVersion(String modelName, String version) {
    String json = sendGet(mapper.makeGetModelVersion(modelName, version));
    GetModelVersion.Response response = mapper.toGetModelVersionResponse(json);
    return response.getModelVersion();
  }

  /**
   *  Returns a RegisteredModel from the model registry for the given model name.
   *   <pre>
   *       import org.mlflow.api.proto.ModelRegistry.RegisteredModel;
   *       RegisteredModel registeredModel = getRegisteredModel("model");
   *   </pre>
   *
   * @param modelName Name of the containing registered model. *
   * @return a registered model {@link org.mlflow.api.proto.ModelRegistry.RegisteredModel}
   */
  public RegisteredModel getRegisteredModel(String modelName) {
    String json = sendGet(mapper.makeGetRegisteredModel(modelName));
    GetRegisteredModel.Response response = mapper.toGetRegisteredModelResponse(json);
    return response.getRegisteredModel();
  }

  /**
   * Return the model URI containing for the given model version. The model URI can be used
   * to download the model version artifacts.
   *
   *    <pre>
   *        String modelUri = getModelVersionDownloadUri("model", 0);
   *    </pre>
   *
   * @param modelName The name of the model
   * @param version The version number of the model
   * @return The specified model version's URI.
   */
  public String getModelVersionDownloadUri(String modelName, String version) {
    String json = sendGet(mapper.makeGetModelVersionDownloadUri(modelName, version));
    return mapper.toGetModelVersionDownloadUriResponse(json);
  }

  /**
   * Returns a directory containing all artifacts within the given registered model
   * version. The method will download the model version artifacts to the local file system. Note
   * that this method will not work if the `download_uri` refers to a single file (and not a
   * directory) due to the way many ArtifactRepository's `download_artifacts` handle empty subpaths.
   *
   *    <pre>
   *        File modelVersionDir = downloadModelVersion("model", 0);
   *    </pre>
   *
   * @param modelName The name of the model
   * @param version The version number of the model
   * @return A directory ({@link java.io.File}) containing model artifacts
   */
  public File downloadModelVersion(String modelName, String version) {
    String path = modelName + "/" + version;
    URIBuilder downloadUriBuilder = new URIBuilder()
            .setScheme(DEFAULT_MODELS_ARTIFACT_REPOSITORY_SCHEME).setPath(path);
    CliBasedArtifactRepository repository = new CliBasedArtifactRepository(null, null,
            hostCredsProvider);
    return repository.downloadArtifactFromUri(downloadUriBuilder.toString());
  }

  /**
   * Returns a directory containing all artifacts within the latest registered
   * model version in the given stage. The method will download the model version artifacts
   * to the local file system.
   *
   *    <pre>
   *        File modelVersionDir = downloadLatestModelVersion("model", "Staging");
   *    </pre>
   *
   * (i.e., the contents of the local directory are now available).
   *
   * @param modelName The name of the model
   * @param stage The name of the stage
   * @return A directory ({@link java.io.File}) containing model artifacts
   */
  public File downloadLatestModelVersion(String modelName, String stage) {
      List<ModelVersion> versions = getLatestVersions(modelName, Lists.newArrayList(stage));

      if (versions.size() < 1) {
        throw new MlflowClientException("No model version found for " + modelName +
                "and stage " + stage);
      }

      ModelVersion details = versions.get(0);
      return downloadModelVersion(modelName, details.getVersion());
  }

  /**
   * Return model versions that satisfy the search query.
   *
   * @param searchFilter SQL compatible search query string.
   *                     Examples:
   *                         - "name = 'model_name'"
   *                         - "run_id = '...'"
   *                     If null, the result will be equivalent to having an empty search filter.
   * @param maxResults Maximum number of model versions desired in one page.
   * @param orderBy List of properties to order by. Example: "name DESC".
   *
   * @return A page of model versions that satisfy the search filter.
   */
  public ModelVersionsPage searchModelVersions(String searchFilter,
                                               int maxResults,
                                               List<String> orderBy) {
    return searchModelVersions(searchFilter, maxResults, orderBy, null);
  }

  /**
   * Return up to 1000 model versions.
   *
   * @return A page of model versions with up to 1000 items.
   */
  public ModelVersionsPage searchModelVersions() {
    return searchModelVersions("", 1000, new ArrayList<>(), null);
  }

  /**
   * Return up to 1000 model versions that satisfy the search query.
   *
   * @param searchFilter SQL compatible search query string.
   *                     Examples:
   *                         - "name = 'model_name'"
   *                         - "run_id = '...'"
   *                     If null, the result will be equivalent to having an empty search filter.
   *
   * @return A page of model versions with up to 1000 items.
   */
  public ModelVersionsPage searchModelVersions(String searchFilter) {
    return searchModelVersions(searchFilter, 1000, new ArrayList<>(), null);
  }

  /**
   * Return model versions that satisfy the search query.
   *
   * @param searchFilter SQL compatible search query string.
   *                     Examples:
   *                         - "name = 'model_name'"
   *                         - "run_id = '...'"
   *                     If null, the result will be equivalent to having an empty search filter.
   * @param maxResults Maximum number of model versions desired in one page.
   * @param orderBy List of properties to order by. Example: "name DESC".
   * @param pageToken String token specifying the next page of results. It should be obtained from
   *             a call to {@link #searchModelVersions(String)}.
   *
   * @return A page of model versions that satisfy the search filter.
   */
  public ModelVersionsPage searchModelVersions(String searchFilter,
                                               int maxResults,
                                               List<String> orderBy,
                                               String pageToken) {
    String json = sendGet(mapper.makeSearchModelVersions(
            searchFilter, maxResults, orderBy, pageToken
    ));
    SearchModelVersions.Response response = mapper.toSearchModelVersionsResponse(json);
    return new ModelVersionsPage(response.getModelVersionsList(), response.getNextPageToken(),
            searchFilter, maxResults, orderBy, this);
  }

  /**
   * Closes the MlflowClient and releases any associated resources.
   */
  public void close() {
    this.httpCaller.close();
  }
}
