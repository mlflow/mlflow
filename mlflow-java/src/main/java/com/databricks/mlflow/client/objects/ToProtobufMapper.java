package com.databricks.mlflow.client.objects;

import java.util.*;
import com.google.protobuf.MessageOrBuilder;
import com.google.protobuf.util.JsonFormat;
import com.databricks.api.proto.mlflow.Service.*;

public class ToProtobufMapper {

    public GetExperiment.Response toGetExperimentResponse(String json) throws Exception {
       GetExperiment.Response.Builder builder = GetExperiment.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
    }

    public ListExperiments.Response toListExperimentsResponse(String json) throws Exception {
       ListExperiments.Response.Builder builder = ListExperiments.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
    }

    public CreateExperiment.Response toCreateExperimentResponse(String json) throws Exception {
       CreateExperiment.Response.Builder builder = CreateExperiment.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
    }

    public String makeCreateExperimentRequest(String expName) throws Exception {
       CreateExperiment.Builder builder = CreateExperiment.newBuilder();
       builder.setName(expName);
       return JsonFormat.printer().print(builder);
    }

    public GetRun.Response toGetRunResponse(String json) throws Exception {
       GetRun.Response.Builder builder = GetRun.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
    }

    public CreateRun.Response toCreateRunResponse(String json) throws Exception {
       CreateRun.Response.Builder builder = CreateRun.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
    }

    public GetMetric.Response toGetMetricResponse(String json) throws Exception {
       GetMetric.Response.Builder builder = GetMetric.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
    }

    public GetMetricHistory.Response toGetMetricHistoryResponse(String json) throws Exception {
       GetMetricHistory.Response.Builder builder = GetMetricHistory.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
    }

    public ListArtifacts.Response toListArtifactsResponse(String json) throws Exception {
       ListArtifacts.Response.Builder builder = ListArtifacts.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
   }

    public SearchRuns.Response toSearchRunsResponse(String json) throws Exception {
       SearchRuns.Response.Builder builder = SearchRuns.Response.newBuilder();
       JsonFormat.parser().merge(json,builder);
       return builder.build();
   }
}
