package com.databricks.mlflow.client.objects;

import java.util.*;
import java.util.stream.Collectors;
import com.google.protobuf.MessageOrBuilder;
import com.databricks.api.proto.mlflow.Service.*;
import com.google.protobuf.util.JsonFormat;

public class ObjectUtils {

    public static  CreateRun makeCreateRun(long experimentId, String runName, SourceType sourceType, String sourceName, long startTime, String userId) {
        CreateRun.Builder request = CreateRun.newBuilder();
        request.setExperimentId(experimentId);
        request.setRunName(runName);
        request.setSourceType(sourceType);
        request.setSourceName(sourceName);
        request.setStartTime(startTime);
        request.setUserId(userId);
        return request.build();
    }

    public static SearchRuns makeSearchRequest(long [] experimentIds, BaseSearch[] clauses) {
        SearchRuns.Builder builder = SearchRuns.newBuilder();
        List<Long> expIds = Arrays.stream(experimentIds).boxed().collect(Collectors.toList());
        builder.addAllExperimentIds(expIds);
        for (BaseSearch cl: clauses) {
            if (cl instanceof ParameterSearch) {
                SearchExpression.Builder expr = SearchExpression.newBuilder();
                ParameterSearchExpression.Builder param = ParameterSearchExpression.newBuilder();
                StringClause.Builder cloz = StringClause.newBuilder();
                cloz.setValue(((ParameterSearch)cl).getValue());
                cloz.setComparator(cl.getComparator());
                param.setKey(cl.getKey());
                param.setString(cloz);
                expr.setParameter(param);
                builder.addAndedExpressions(expr);
            } else {
                SearchExpression.Builder expr = SearchExpression.newBuilder();
                MetricSearchExpression.Builder param = MetricSearchExpression.newBuilder();
                FloatClause.Builder cloz = FloatClause.newBuilder();
                cloz.setValue(((MetricSearch)cl).getValue());
                cloz.setComparator(cl.getComparator());
                param.setKey(cl.getKey());
                param.setFloat(cloz);
                expr.setMetric(param);
	            builder.addAndedExpressions(expr);
            }
        }
        return builder.build();
    }

    public String toJson(MessageOrBuilder mb) throws Exception {
         return JsonFormat.printer().print(mb);
    }
}
