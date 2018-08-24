package com.databricks.mlflow.client;

import java.util.*;
import org.testng.Assert;
import com.databricks.api.proto.mlflow.Service.*;
import static com.databricks.mlflow.client.TestUtils.*;

public class TestShared {
    static float auc = 2.0F;
    static float accuracy_score = 0.9733333333333334F;
    static float zero_one_loss = 0.026666666666666616F;
    static String min_samples_leaf = "2";
    static String max_depth = "3";

    static public void assertParamsAndMetrics(ApiClient client, Run run, String runId) throws Exception {
        List<Param> params = run.getData().getParamsList();
        Assert.assertEquals(params.size(),2);
        assertParam(params,"min_samples_leaf",min_samples_leaf);
        assertParam(params,"max_depth",max_depth);

        List<Metric> metrics = run.getData().getMetricsList();
        Assert.assertEquals(metrics.size(),3);
        assertMetric(metrics,"auc",auc);
        assertMetric(metrics,"accuracy_score",accuracy_score);
        assertMetric(metrics,"zero_one_loss",zero_one_loss);

        Metric m = client.getMetric(runId,"auc");
        Assert.assertEquals(m.getKey(),"auc");
        Assert.assertEquals(m.getValue(),auc);

        metrics = client.getMetricHistory(runId,"auc");
        Assert.assertEquals(metrics.size(),1);
        m = metrics.get(0);
        Assert.assertEquals(m.getKey(),"auc");
        Assert.assertEquals(m.getValue(),auc);
    }
}
