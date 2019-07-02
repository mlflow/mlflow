package org.mlflow.tracking;

import org.mlflow.api.proto.Service.*;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class ActiveRun {
  private transient MlflowClient client;
  private RunInfo runInfo;
  boolean isTerminated;

  public String getId() {
    return runInfo.getRunId();
  }

  ActiveRun(RunInfo runInfo, MlflowClient client) {
    this.runInfo = runInfo;
    this.client = client;
  }

  /**
   * Synchronous and will throw MlflowClientException on failures.
   */
  public void logParam(String key, String value) {
    client.logParam(getId(), key, value);
  }

  public void setTag(String key, String value) {
    client.setTag(getId(), key, value);
  }

  public void logMetric(String key, double value) {
    logMetric(key, value, 0);
  }

  public void logMetric(String key, double value, int step) {
    client.logMetric(getId(), key, value, System.currentTimeMillis(), step);
  }

  public void logMetrics(Map<String, Double> metrics) {
      logMetrics(metrics, 0);
  }

  public void logMetrics(Map<String, Double> metrics, int step) {
    List<Metric> protoMetrics = metrics.entrySet().stream()
      .map((metric) ->
        Metric.newBuilder()
          .setKey(metric.getKey())
          .setValue(metric.getValue())
          .setTimestamp(System.currentTimeMillis())
          .setStep(step)
          .build()
      ).collect(Collectors.toList());
    client.logBatch(getId(), protoMetrics, Collections.emptyList(), Collections.emptyList());
  }

  public void logParams(Map<String, String> params) {
    List<Param> protoParams = params.entrySet().stream().map((param) ->
      Param.newBuilder()
        .setKey(param.getKey())
        .setValue(param.getValue())
        .build()
    ).collect(Collectors.toList());
    client.logBatch(getId(), Collections.emptyList(), protoParams, Collections.emptyList());
  }

  public void setTags(Map<String, String> tags) {
    List<RunTag> protoTags = tags.entrySet().stream().map((tag) ->
      RunTag.newBuilder().setKey(tag.getKey()).setValue(tag.getValue()).build()
    ).collect(Collectors.toList());
    client.logBatch(getId(), Collections.emptyList(), Collections.emptyList(), protoTags);
  }

  public void logArtifact(Path localPath) {
    client.logArtifact(getId(), localPath.toFile());
  }

  public void logArtifact(Path localPath, String artifactPath) {
    client.logArtifact(getId(), localPath.toFile(), artifactPath);
  }

  public String getArtifactUri() {
    return this.runInfo.getArtifactUri();
  }

  public ActiveRun endRun() {
    return endRun(RunStatus.FINISHED);
  }

  public ActiveRun endRun(RunStatus status) {
    isTerminated = true;
    client.setTerminated(getId(), status);
    return this;
  }
}
