package org.mlflow.tracking;

import org.mlflow.MlflowMetric;
import org.mlflow.MlflowParam;
import org.mlflow.MlflowTag;
import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.utils.DatabricksContext;
import org.mlflow.tracking.utils.MlflowTagConstants;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class ActiveRun {
    private MlflowClient client;
    private RunInfo runInfo;
    private String experimentId;
    boolean isTerminated;

    public String getId() {
        return runInfo.getRunId();
    }

    ActiveRun(RunInfo runInfo, MlflowClient client, String experimentId) {
        this.runInfo = runInfo;
        this.client = client;
        this.experimentId = experimentId;
        this.isTerminated = false;
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

    public void logMetrics(Iterable<MlflowMetric> metrics) {
        logMetrics(metrics, 0);
    }

    // TODO(andrew): Should this be it's own object in org.mlflow or should it be just the proto
    // object.
    public void logMetrics(Iterable<MlflowMetric> metrics, int step) {
        List<Metric> protoMetrics = StreamSupport
                .stream(metrics.spliterator(), false)
                .map((metric) ->
                Metric.newBuilder()
                        .setKey(metric.key)
                        .setValue(metric.value)
                        .setTimestamp(System.currentTimeMillis())
                        .setStep(step)
                        .build()
        ).collect(Collectors.toList());
        client.logBatch(getId(), protoMetrics, Collections.emptyList(), Collections.emptyList());
    }

    public void logParams(Iterable<MlflowParam> params) {
        List<Param> protoParams = StreamSupport.stream(params.spliterator(), false).map((param) ->
                Param.newBuilder()
                        .setKey(param.key)
                        .setValue(param.value)
                        .build()
        ).collect(Collectors.toList());
        client.logBatch(getId(), Collections.emptyList(), protoParams, Collections.emptyList());
    }

    public void setTags(Iterable<MlflowTag> tags) {
        List<RunTag> protoTags = StreamSupport.stream(tags.spliterator(), false).map((tag) ->
                RunTag.newBuilder().setKey(tag.key).setValue(tag.value).build()
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

    public ActiveRun startChildRun(String runName) {
        Map<String, String> tags = new HashMap<>();
        tags.put(MlflowTagConstants.RUN_NAME, runName);
        tags.put(MlflowTagConstants.USER, System.getProperty("user.name"));
        tags.put(MlflowTagConstants.SOURCE_TYPE, "LOCAL");
        tags.put(MlflowTagConstants.PARENT_RUN_ID, getId());

        // Add tags from DatabricksContext if they exist
        DatabricksContext databricksContext = DatabricksContext.createIfAvailable();
        if (databricksContext != null) {
            tags.putAll(databricksContext.getTags());
        }

        CreateRun.Builder createRunBuilder = CreateRun.newBuilder()
          .setExperimentId(experimentId)
          .setStartTime(System.currentTimeMillis());
        for (Map.Entry<String, String> tag: tags.entrySet()) {
            createRunBuilder.addTags(
              RunTag.newBuilder().setKey(tag.getKey()).setValue(tag.getValue()).build());
        }
        RunInfo runInfo = client.createRun(createRunBuilder.build());

        return new ActiveRun(runInfo, client, experimentId);
    }

    public ActiveRun endRun() {
        return endRun(RunStatus.FINISHED);
    }

    public ActiveRun endRun(RunStatus status) {
        client.setTerminated(getId(), status);
        this.isTerminated = true;
        return this;
    }
}
