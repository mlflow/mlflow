package org.mlflow.tracking;

import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageOrBuilder;
import com.google.protobuf.util.JsonFormat;

import java.lang.Iterable;
import java.net.URISyntaxException;
import java.util.List;

import org.apache.http.client.utils.URIBuilder;
import org.mlflow.api.proto.ModelRegistry.*;
import org.mlflow.api.proto.Service.*;

class MlflowProtobufMapper {

  String makeCreateExperimentRequest(String expName) {
    CreateExperiment.Builder builder = CreateExperiment.newBuilder();
    builder.setName(expName);
    return print(builder);
  }

  String makeDeleteExperimentRequest(String experimentId) {
    DeleteExperiment.Builder builder = DeleteExperiment.newBuilder();
    builder.setExperimentId(experimentId);
    return print(builder);
  }

  String makeRestoreExperimentRequest(String experimentId) {
    RestoreExperiment.Builder builder = RestoreExperiment.newBuilder();
    builder.setExperimentId(experimentId);
    return print(builder);
  }

  String makeUpdateExperimentRequest(String experimentId, String newExperimentName) {
    UpdateExperiment.Builder builder = UpdateExperiment.newBuilder();
    builder.setExperimentId(experimentId);
    builder.setNewName(newExperimentName);
    return print(builder);
  }

  String makeLogParam(String runId, String key, String value) {
    LogParam.Builder builder = LogParam.newBuilder();
    builder.setRunUuid(runId);
    builder.setRunId(runId);
    builder.setKey(key);
    builder.setValue(value);
    return print(builder);
  }

  String makeLogMetric(String runId, String key, double value, long timestamp, long step) {
    LogMetric.Builder builder = LogMetric.newBuilder();
    builder.setRunUuid(runId);
    builder.setRunId(runId);
    builder.setKey(key);
    builder.setValue(value);
    builder.setTimestamp(timestamp);
    builder.setStep(step);
    return print(builder);
  }

  String makeSetExperimentTag(String expId, String key, String value) {
    SetExperimentTag.Builder builder = SetExperimentTag.newBuilder();
    builder.setExperimentId(expId);
    builder.setKey(key);
    builder.setValue(value);
    return print(builder);
  }

  String makeSetTag(String runId, String key, String value) {
    SetTag.Builder builder = SetTag.newBuilder();
    builder.setRunUuid(runId);
    builder.setRunId(runId);
    builder.setKey(key);
    builder.setValue(value);
    return print(builder);
  }

  String makeDeleteTag(String runId, String key) {
    DeleteTag.Builder builder = DeleteTag.newBuilder();
    builder.setRunId(runId);
    builder.setKey(key);
    return print(builder);
  }

  String makeLogBatch(String runId,
      Iterable<Metric> metrics,
      Iterable<Param> params,
      Iterable<RunTag> tags) {
    LogBatch.Builder builder = LogBatch.newBuilder();
    builder.setRunId(runId);
    if (metrics != null) {
      builder.addAllMetrics(metrics);
    }
    if (params != null) {
      builder.addAllParams(params);
    }
    if (tags != null) {
      builder.addAllTags(tags);
    }
    return print(builder);
  }

  String makeUpdateRun(String runId, RunStatus status, long endTime) {
    UpdateRun.Builder builder = UpdateRun.newBuilder();
    builder.setRunUuid(runId);
    builder.setRunId(runId);
    builder.setStatus(status);
    builder.setEndTime(endTime);
    return print(builder);
  }

  String makeDeleteRun(String runId) {
    DeleteRun.Builder builder = DeleteRun.newBuilder();
    builder.setRunId(runId);
    return print(builder);
  }

  String makeRestoreRun(String runId) {
    RestoreRun.Builder builder = RestoreRun.newBuilder();
    builder.setRunId(runId);
    return print(builder);
  }

  String makeGetLatestVersion(String modelName, Iterable<String> stages) {
    try {
      URIBuilder builder = new URIBuilder("registered-models/get-latest-versions")
              .addParameter("name", modelName);
      if (stages != null) {
        for( String stage: stages) {
          builder.addParameter("stages", stage);
        }
      }
      return builder.build().toString();
    } catch (URISyntaxException e) {
      throw new MlflowClientException("Failed to construct request URI for get latest versions.",
              e);
    }
  }

  String makeUpdateModelVersion(String modelName, String version) {
    return print(UpdateModelVersion.newBuilder().setName(modelName).setVersion(version));
  }

  String makeTransitionModelVersionStage(String modelName, String version, String stage) {
    return print(TransitionModelVersionStage.newBuilder()
            .setName(modelName).setVersion(version).setStage(stage));
  }

  String makeCreateModel(String modelName) {
    CreateRegisteredModel.Builder builder = CreateRegisteredModel.newBuilder()
            .setName(modelName);
    return print(builder);
  }

  String makeCreateModelVersion(String modelName, String runId, String source) {
    CreateModelVersion.Builder builder = CreateModelVersion.newBuilder()
            .setName(modelName)
            .setRunId(runId)
            .setSource(source);
    return print(builder);
  }

  String makeGetRegisteredModel(String modelName) {
    try {
      return new URIBuilder("registered-models/get")
          .addParameter("name", modelName)
          .build()
          .toString();
    } catch (URISyntaxException e) {
      throw new MlflowClientException("Failed to construct request URI for get model version.", e);
    }
  }

  String makeGetModelVersion(String modelName, String modelVersion) {
    try {
      return new URIBuilder("model-versions/get")
          .addParameter("name", modelName)
          .addParameter("version", modelVersion)
          .build()
          .toString();
    } catch (URISyntaxException e) {
      throw new MlflowClientException("Failed to construct request URI for get model version.", e);
    }
  }

  String makeGetModelVersionDownloadUri(String modelName, String modelVersion) {
    try {
      return new URIBuilder("model-versions/get-download-uri")
              .addParameter("name", modelName)
              .addParameter("version", modelVersion).build().toString();
    } catch (URISyntaxException e) {
      throw new MlflowClientException(
              "Failed to construct request URI for get version download uri.",
              e);
    }
  }

  String makeSearchModelVersions(String searchFilter,
                                 int maxResults,
                                 List<String> orderBy,
                                 String pageToken) {
    try {
      URIBuilder builder = new URIBuilder("model-versions/search")
              .addParameter("max_results", Integer.toString(maxResults));
      if (searchFilter != null && searchFilter != "") {
        builder.addParameter("filter", searchFilter);
      }
      if (pageToken != null && pageToken != "") {
        builder.addParameter("page_token", pageToken);
      }
      for( String order: orderBy) {
        builder.addParameter("order_by", order);
      }
      return builder.build().toString();
    } catch (URISyntaxException e) {
      throw new MlflowClientException(
              "Failed to construct request URI for search model version.",
              e);
    }
  }

  String toJson(MessageOrBuilder mb) {
    return print(mb);
  }

  GetExperiment.Response toGetExperimentResponse(String json) {
    GetExperiment.Response.Builder builder = GetExperiment.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  GetExperimentByName.Response toGetExperimentByNameResponse(String json) {
    GetExperimentByName.Response.Builder builder = GetExperimentByName.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  SearchExperiments.Response toSearchExperimentsResponse(String json) {
    SearchExperiments.Response.Builder builder = SearchExperiments.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  CreateExperiment.Response toCreateExperimentResponse(String json) {
    CreateExperiment.Response.Builder builder = CreateExperiment.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  GetRun.Response toGetRunResponse(String json) {
    GetRun.Response.Builder builder = GetRun.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  GetMetricHistory.Response toGetMetricHistoryResponse(String json) {
    GetMetricHistory.Response.Builder builder = GetMetricHistory.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  CreateRun.Response toCreateRunResponse(String json) {
    CreateRun.Response.Builder builder = CreateRun.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  SearchRuns.Response toSearchRunsResponse(String json) {
    SearchRuns.Response.Builder builder = SearchRuns.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  GetLatestVersions.Response toGetLatestVersionsResponse(String json) {
    GetLatestVersions.Response.Builder builder = GetLatestVersions.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  GetModelVersion.Response toGetModelVersionResponse(String json) {
    GetModelVersion.Response.Builder builder = GetModelVersion.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  GetRegisteredModel.Response toGetRegisteredModelResponse(String json) {
    GetRegisteredModel.Response.Builder builder = GetRegisteredModel.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  String toGetModelVersionDownloadUriResponse(String json) {
    GetModelVersionDownloadUri.Response.Builder builder = GetModelVersionDownloadUri.Response
            .newBuilder();
    merge(json, builder);
    return builder.getArtifactUri();
  }

  SearchModelVersions.Response toSearchModelVersionsResponse(String json) {
    SearchModelVersions.Response.Builder builder = SearchModelVersions.Response.newBuilder();
    merge(json, builder);
    return builder.build();
  }

  private String print(MessageOrBuilder message) {
    try {
      return JsonFormat.printer().preservingProtoFieldNames().print(message);
    } catch (InvalidProtocolBufferException e) {
      throw new MlflowClientException("Failed to serialize message " + message, e);
    }
  }

  private void merge(String json, com.google.protobuf.Message.Builder builder) {
    try {
      JsonFormat.parser().ignoringUnknownFields().merge(json, builder);
    } catch (InvalidProtocolBufferException e) {
      throw new MlflowClientException("Failed to serialize json " + json + " into " + builder, e);
    }
  }
}
