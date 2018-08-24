package org.mlflow.client.objects;

import java.util.*;
import java.util.stream.Collectors;

import com.google.protobuf.MessageOrBuilder;

import org.mlflow.api.proto.Service.*;

import com.google.protobuf.util.JsonFormat;

public class ObjectUtils {

  public static CreateRun makeCreateRun(
      long experimentId,
      String runName,
      SourceType sourceType,
      String sourceName,
      long startTime,
      String userId) {
    CreateRun.Builder request = CreateRun.newBuilder();
    request.setExperimentId(experimentId);
    request.setRunName(runName);
    request.setSourceType(sourceType);
    request.setSourceName(sourceName);
    request.setStartTime(startTime);
    request.setUserId(userId);
    return request.build();
  }

  public String toJson(MessageOrBuilder mb) throws Exception {
    return JsonFormat.printer().print(mb);
  }
}
