package org.mlflow.tracking;

import org.apache.http.client.utils.URIBuilder;

import org.mlflow.api.proto.Service.*;
import org.mlflow.artifacts.ArtifactRepository;
import org.mlflow.artifacts.ArtifactRepositoryFactory;
import org.mlflow.tracking.creds.*;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.lang.Iterable;
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

  /**
   * Gets metadata, params, tags, and metrics for a run. In the case where multiple metrics with the
   * same key are logged for the run, returns only the value with the latest timestamp. If there are
   * multiple values with the latest timestamp, returns the maximum of these values.
   *
   * @return Run associated with the id.
   */
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
   * Creates a new run. This method allows providing all possible fields of CreateRun, and can be
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
   * @return RunInfo created by the server
   */
  public RunInfo createRun(CreateRun request) {
    String ijson = mapper.toJson(request);
    String ojson = sendPost("runs/create", ijson);
    return mapper.toCreateRunResponse(ojson).getRun().getInfo();
  }

  /**
   * @return a list of all RunInfos associated with the given experiment.
   */
  public List<RunInfo> listRunInfos(long experimentId) {
    List<Long> experimentIds = new ArrayList<>();
    experimentIds.add(experimentId);
    return searchRuns(experimentIds, null);
  }

  /**
   * Returns runs from provided list of experiments, that satisfy the search query.
   *
   * @param experimentIds List of experiment IDs
   * @param searchFilter SQL compatible search query string. Format of this query string is
   *                     similar to that specified on MLflow UI.
   *                     Example : "params.model = 'LogisticRegression' and metrics.acc = 0.9"
   *
   * @return a list of all RunInfos that satisfy search filter.
   */
  public List<RunInfo> searchRuns(List<Long> experimentIds, String searchFilter) {
    return searchRuns(experimentIds, searchFilter, ViewType.ACTIVE_ONLY);
  }

  /**
   * Returns runs from provided list of experiments, that satisfy the search query.
   *
   * @param experimentIds List of experiment IDs
   * @param searchFilter SQL compatible search query string. Format of this query string is
   *                     similar to that specified on MLflow UI.
   *                     Example : "params.model = 'LogisticRegression' and metrics.acc != 0.9"
   * @param runViewType ViewType for expected runs. One of (ACTIVE_ONLY, DELETED_ONLY, ALL)
   *                    Defaults to ACTIVE_ONLY.
   *
   * @return a list of all RunInfos that satisfy search filter.
   */
  public List<RunInfo> searchRuns(List<Long> experimentIds,
                                  String searchFilter,
                                  ViewType runViewType) {
    SearchRuns.Builder builder = SearchRuns.newBuilder().addAllExperimentIds(experimentIds);
    if (searchFilter != null) {
      builder.setFilter(searchFilter);
    }
    if (runViewType != null) {
      builder.setRunViewType(runViewType);
    }
    SearchRuns request = builder.build();
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

  /** Mark an experiment and associated runs, params, metrics, etc for deletion. */
  public void deleteExperiment(long experimentId) {
    String ijson = mapper.makeDeleteExperimentRequest(experimentId);
    httpCaller.post("experiments/delete", ijson);
  }

  /** Restore an experiment marked for deletion. */
  public void restoreExperiment(long experimentId) {
    String ijson = mapper.makeRestoreExperimentRequest(experimentId);
    httpCaller.post("experiments/restore", ijson);
  }

  /** Update an experiment's name. The new name must be unique. */
  public void renameExperiment(long experimentId, String newName) {
    String ijson = mapper.makeUpdateExperimentRequest(experimentId, newName);
    httpCaller.post("experiments/update", ijson);
  }

  /**
   * Deletes a run with the given ID.
   */
  public void deleteRun(String runId) {
    String ijson = mapper.makeDeleteRun(runId);
    httpCaller.post("runs/delete", ijson);
  }

  /**
   * Restores a deleted run with the given ID.
   */
  public void restoreRun(String runId) {
    String ijson = mapper.makeRestoreRun(runId);
    httpCaller.post("runs/restore", ijson);
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
  public void logMetric(String runUuid, String key, double value) {
    sendPost("runs/log-metric", mapper.makeLogMetric(runUuid, key, value,
      System.currentTimeMillis()));
  }

  /**
   * Logs a new tag against the given run, as a key-value pair.
   */
  public void setTag(String runUuid, String key, String value) {
    sendPost("runs/set-tag", mapper.makeSetTag(runUuid, key, value));
  }

  /**
   * Log multiple metrics, params, and/or tags against a given run (argument runUuid).
   * Argument metrics, params, and tag iterables can be nulls.
   */
  public void logBatch(String runUuid,
      Iterable<Metric> metrics,
      Iterable<Param> params,
      Iterable<RunTag> tags) {
    sendPost("runs/log-batch", mapper.makeLogBatch(runUuid, metrics, params, tags));
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
   * Uploads the given local file to the run's root artifact directory. For example,
   *
   *   <pre>
   *   logArtifact(runId, "/my/localModel")
   *   listArtifacts(runId) // returns "localModel"
   *   </pre>
   *
   * @param runId Run ID of an existing MLflow run.
   * @param localFile File to upload. Must exist, and must be a simple file (not a directory).
   */
  public void logArtifact(String runId, File localFile) {
    getArtifactRepository(runId).logArtifact(localFile);
  }

  /**
   * Uploads the given local file to an artifactPath within the run's root directory. For example,
   *
   *   <pre>
   *   logArtifact(runId, "/my/localModel", "model")
   *   listArtifacts(runId, "model") // returns "model/localModel"
   *   </pre>
   *
   * (i.e., the localModel file is now available in model/localModel).
   *
   * @param runId Run ID of an existing MLflow run.
   * @param localFile File to upload. Must exist, and must be a simple file (not a directory).
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  public void logArtifact(String runId, File localFile, String artifactPath) {
    getArtifactRepository(runId).logArtifact(localFile, artifactPath);
  }

  /**
   * Uploads all files within the given local directory the run's root artifact directory.
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
   * Uploads all files within the given local director an artifactPath within the run's root
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
   * Lists the artifacts immediately under the run's root artifact directory. This does not
   * recursively list; instead, it will return FileInfos with isDir=true where further
   * listing may be done.
   * @param runId Run ID of an existing MLflow run.
   */
  public List<FileInfo> listArtifacts(String runId) {
    return getArtifactRepository(runId).listArtifacts();
  }

  /**
   * Lists the artifacts immediately under the given artifactPath within the run's root artifact
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
   * Returns a local directory containing *all* artifacts within the run's artifact directory.
   * Note that this will download the entire directory path, and so may be expensive if
   * the directory has a lot of data.
   * @param runId Run ID of an existing MLflow run.
   */
  public File downloadArtifacts(String runId) {
    return getArtifactRepository(runId).downloadArtifacts();
  }

  /**
   * Returns a local file or directory containing all artifacts within the given artifactPath
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
}
