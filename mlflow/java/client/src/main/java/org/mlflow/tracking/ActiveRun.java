package org.mlflow.tracking;

import org.mlflow.MlflowMetric;
import org.mlflow.MlflowParam;
import org.mlflow.MlflowTag;
import org.mlflow.api.proto.Service.*;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class ActiveRun {
    protected MlflowClient client;
    protected RunInfo runInfo;

    public String getId() {
        return runInfo.getRunId();
    }

    public ActiveRun(RunInfo runInfo, MlflowClient client) {
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
}
