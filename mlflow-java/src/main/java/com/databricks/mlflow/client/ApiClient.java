package com.databricks.mlflow.client;

import java.util.*;
import org.apache.http.client.utils.URIBuilder;
import com.databricks.api.proto.mlflow.Service.*;
import com.databricks.mlflow.client.objects.BaseSearch;
import com.databricks.mlflow.client.objects.ObjectUtils;
import com.databricks.mlflow.client.objects.FromProtobufMapper;
import com.databricks.mlflow.client.objects.ToProtobufMapper;

public class ApiClient {
    private String basePath = "api/2.0/preview/mlflow";
    private ToProtobufMapper toMapper = new ToProtobufMapper();
    private FromProtobufMapper fromMapper = new FromProtobufMapper();
    private HttpCaller httpCaller ;

    public ApiClient(String baseApiUri) throws Exception {
        String apiUri = baseApiUri + "/" + basePath;
        httpCaller = new HttpCaller(apiUri);
    }

    public ApiClient(String baseApiUri, String user, String password) throws Exception {
        String apiUri = baseApiUri + "/" + basePath;
        httpCaller = new HttpCaller(apiUri, user, password);
    }

    public void setVerbose(boolean verbose) {
        httpCaller.setVerbose(verbose);
    }

    public GetExperiment.Response getExperiment(long experimentId) throws Exception {
        URIBuilder builder = httpCaller.makeURIBuilder("experiments/get").setParameter("experiment_id",""+experimentId);
        return toMapper.toGetExperimentResponse(httpCaller._get(builder));
    }

    public List<Experiment> listExperiments() throws Exception {
        return toMapper.toListExperimentsResponse(httpCaller.get("experiments/list")).getExperimentsList();
    }

    public long createExperiment(String experimentName) throws Exception {
        String ijson = fromMapper.makeCreateExperimentRequest(experimentName);
        String ojson = post("experiments/create",ijson);
        return toMapper.toCreateExperimentResponse(ojson).getExperimentId();
    }

    public Run getRun(String runUuid) throws Exception {
        URIBuilder builder = httpCaller.makeURIBuilder("runs/get").setParameter("run_uuid",runUuid);
        return toMapper.toGetRunResponse(httpCaller._get(builder)).getRun();
    }

    public RunInfo createRun(CreateRun request) throws Exception {
        String ijson = fromMapper.toJson(request);
        String ojson = post("runs/create",ijson);
        return toMapper.toCreateRunResponse(ojson).getRun().getInfo();
    }

    public void updateRun(String runUuid, RunStatus status, long endTime) throws Exception {
        post("runs/update", fromMapper.makeUpdateRun(runUuid, status, endTime));
    }

    public void logParameter(String runUuid, String key, String value) throws Exception {
        post("runs/log-parameter",fromMapper.makeLogParam(runUuid, key, value));
    }
    public void logMetric(String runUuid, String key, float value) throws Exception {
        post("runs/log-metric", fromMapper.makeLogMetric(runUuid, key, value));
    }

    public Metric getMetric(String runUuid, String metricKey) throws Exception {
        URIBuilder builder = httpCaller.makeURIBuilder("metrics/get")
            .setParameter("run_uuid",runUuid)
            .setParameter("metric_key",metricKey);
        return toMapper.toGetMetricResponse(httpCaller._get(builder)).getMetric();
    }

    public List<Metric> getMetricHistory(String runUuid, String metricKey) throws Exception {
        URIBuilder builder = httpCaller.makeURIBuilder("metrics/get-history")
            .setParameter("run_uuid",runUuid)
            .setParameter("metric_key",metricKey);
        return toMapper.toGetMetricHistoryResponse(httpCaller._get(builder)).getMetricsList();
    }

    public ListArtifacts.Response listArtifacts(String runUuid, String path) throws Exception {
        URIBuilder builder = httpCaller.makeURIBuilder("artifacts/list")
            .setParameter("run_uuid",runUuid)
            .setParameter("path",path);
        return toMapper.toListArtifactsResponse(httpCaller._get(builder));
    }

    public byte [] getArtifact(String runUuid, String path) throws Exception {
        URIBuilder builder = httpCaller.makeURIBuilder("artifacts/get")
            .setParameter("run_uuid",runUuid)
            .setParameter("path",path);
        return httpCaller._getAsBytes(builder.toString());
    }

    public SearchRuns.Response search(long [] experimentIds, BaseSearch[] clauses) throws Exception {
        SearchRuns search =  ObjectUtils.makeSearchRequest(experimentIds, clauses);
        String ijson = fromMapper.toJson(search);
        String ojson = post("runs/search",ijson);
        return toMapper.toSearchRunsResponse(ojson);
    }

    public Optional<Experiment> getExperimentByName(String experimentName) throws Exception {
        return listExperiments().stream().filter(e -> e.getName().equals(experimentName)).findFirst();
    }

    public long getOrCreateExperimentId(String experimentName) throws Exception {
        Optional<Experiment> opt = getExperimentByName(experimentName);
        return opt.isPresent() ? opt.get().getExperimentId() : createExperiment(experimentName);
    }

    public String get(String path) throws Exception {
        return httpCaller.get(path);
    }

    public String post(String path, String json) throws Exception {
        return httpCaller.post(path,json);
    }
}
