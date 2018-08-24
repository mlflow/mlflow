package com.databricks.mlflow.client;

import java.util.*;
import org.testng.Assert;
import org.testng.annotations.*;
import com.databricks.api.proto.mlflow.Service.*;

/** 
  Check the results of a Python sckit-learn program run in the test Docker container.
  See $ROOT/docker: sklearn_sample.py and run_all.sh.
*/
public class PythonScikitLearnTest extends BaseTest {
    long expId = 0 ;
    String runId;
    float auc = 2.0F ;
    float accuracy_score = 0.9733333333333334F ;
    float zero_one_loss = 0.026666666666666616F ;

    @Test
    public void checkExperiment() throws Exception {
        GetExperiment.Response rsp = client.getExperiment(expId);
        List<RunInfo> runs = rsp.getRunsList();
        Assert.assertEquals(runs.size(),1);
        RunInfo run = runs.get(0);
        runId = run.getRunUuid();
    }

    @Test (dependsOnMethods={"checkExperiment"})
    public void checkParamsAndMetrics() throws Exception {
        TestShared.assertParamsAndMetrics(client, client.getRun(runId), runId);
    }

    @Test (dependsOnMethods={"checkExperiment"})
    public void checkListArtifacts() throws Exception {
        ListArtifacts.Response rsp = client.listArtifacts(runId,"");
        List<FileInfo> files = rsp.getFilesList();
        Assert.assertEquals(files.size(),3);
        assertFile(files,"confusion_matrix.txt");
        assertFile(files,"classification_report.txt");
        assertFile(files,"model");
    }

    @Test (dependsOnMethods={"checkExperiment"})
    public void checkGetArtifact() throws Exception {
        byte[] bytes = client.getArtifact(runId,"model/model.pkl");
        Assert.assertTrue(bytes.length > 0);
        bytes = client.getArtifact(runId,"confusion_matrix.txt");
        Assert.assertTrue(bytes.length > 0);
    }

    private void assertFile(List<FileInfo> files, String path) {
        Assert.assertTrue(files.stream().filter(e -> e.getPath().equals(path)).findFirst().isPresent());
    }
}
