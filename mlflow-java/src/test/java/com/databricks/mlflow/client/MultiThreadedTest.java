package com.databricks.mlflow.client;

import java.util.*;
import org.testng.Assert;
import org.testng.annotations.*;
import com.databricks.api.proto.mlflow.Service.*;
import com.databricks.mlflow.client.objects.ObjectUtils;
import static com.databricks.mlflow.client.TestUtils.*;

public class MultiThreadedTest extends BaseTest {
    List<String> runIds = Collections.synchronizedList(new ArrayList<String>());
    long expId ;
    String expName ;
    Random random = new Random();
    static final int invocationCount = 10;

    @BeforeClass
    public void beforeClass() throws Exception {
        expName = createExperimentName() + "_MultiThreaded";
        expId = client.createExperiment(expName);
    }

    @Test(threadPoolSize = 3, invocationCount = invocationCount,  timeOut = 10000)
    public void testRunsInParallel() throws Exception {
        long startTime = System.currentTimeMillis();
        String user = "foo";
        String sourceFile = "MyFile.java";

        CreateRun request = ObjectUtils.makeCreateRun(expId, "run_for_"+expId, SourceType.LOCAL, sourceFile, startTime, user);   
        RunInfo runInfo = client.createRun(request);
        String runId = runInfo.getRunUuid();
        runIds.add(runId);

        float fval = random.nextFloat() * 100;
        int n = random.nextInt(10);
        for (int j=0 ; j < n ; j++) {
          client.logParameter(runId, "p"+j, ""+(fval+1));
        }
        for (int j=0 ; j < n+1 ; j++) {
          client.logMetric(runId, "m"+j, fval+1);
        }

        // Update finished run
        client.updateRun(runId, RunStatus.FINISHED, startTime+1001);
 
        // Assert run from getExperiment
        GetExperiment.Response expResponse = client.getExperiment(expId);
        Experiment exp = expResponse.getExperiment() ;
        Assert.assertEquals(exp.getName(),expName);
        assertRunInfo(expResponse.getRunsList().get(0), expId, user, sourceFile);

        // Assert run from getRun
        Run run = client.getRun(runId);
        runInfo = run.getInfo();
        assertRunInfo(runInfo, expId, user, sourceFile);

        // Assert run params
        List<Param> params = run.getData().getParamsList();
        Assert.assertEquals(params.size(),n);
        for (int j=0 ; j < n ; j++) {
            assertParam(params,"p"+j,""+(fval+1));
        }

        // Assert run metrics
        List<Metric> metrics = run.getData().getMetricsList();
        Assert.assertEquals(metrics.size(),n+1);
        for (int j=0 ; j < n+1 ; j++) {
            assertMetric(metrics,"m"+j,fval+1);
        }
    }

    @Test (dependsOnMethods={"testRunsInParallel"})
    public void checkNumberOfRunIds() {
        Assert.assertEquals(runIds.size(),invocationCount);
    }
}
