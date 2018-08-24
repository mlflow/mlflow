package org.mlflow.client.objects;

import com.google.protobuf.MessageOrBuilder;
import com.google.protobuf.util.JsonFormat;

import org.mlflow.api.proto.Service.*;

public class FromProtobufMapper {

  public String makeCreateExperimentRequest(String expName) throws Exception {
    CreateExperiment.Builder builder = CreateExperiment.newBuilder();
    builder.setName(expName);
    return JsonFormat.printer().print(builder);
  }

  public String makeLogParam(String runUuid, String key, String value) throws Exception {
    LogParam.Builder builder = LogParam.newBuilder();
    builder.setRunUuid(runUuid);
    builder.setKey(key);
    builder.setValue(value);
    return JsonFormat.printer().print(builder);
  }

  public String makeLogMetric(String runUuid, String key, float value) throws Exception {
    LogMetric.Builder builder = LogMetric.newBuilder();
    builder.setRunUuid(runUuid);
    builder.setKey(key);
    builder.setValue(value);
    return JsonFormat.printer().print(builder);
  }

  public String makeUpdateRun(String runUuid, RunStatus status, long endTime) throws Exception {
    UpdateRun.Builder builder = UpdateRun.newBuilder();
    builder.setRunUuid(runUuid);
    builder.setStatus(status);
    builder.setEndTime(endTime);
    return JsonFormat.printer().print(builder);
  }

  public String toJson(MessageOrBuilder mb) throws Exception {
    return JsonFormat.printer().print(mb);
  }
}
