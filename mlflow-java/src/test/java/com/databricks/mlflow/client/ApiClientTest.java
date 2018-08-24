package com.databricks.mlflow.client;

import java.util.*;
import org.apache.log4j.Logger;
import org.testng.Assert;
import org.testng.annotations.*;
import static com.databricks.mlflow.client.TestUtils.*;
import com.databricks.api.proto.mlflow.Service.*;
import com.databricks.mlflow.client.objects.*;

public class ApiClientTest extends BaseTest {
    private static final Logger logger = Logger.getLogger(ApiClientTest.class);
    long expIdPythonScikitLearnTest = 0 ;
    String runId ;

    @Test
    public void getCreateExperimentTest() throws Exception {
        String expName = createExperimentName();
        long expId = client.createExperiment(expName);
        GetExperiment.Response exp = client.getExperiment(expId);
        Assert.assertEquals(exp.getExperiment().getName(),expName);
    }

    @Test (expectedExceptions = HttpServerException.class) // TODO: server should throw 406
    public void createExistingExperiment() throws Exception {
        String expName = createExperimentName();
        client.createExperiment(expName);
        client.createExperiment(expName);
    }

    @Test
    public void listExperimentsTest() throws Exception {
        List<Experiment> expsBefore = client.listExperiments();

        String expName = createExperimentName();
        long expId = client.createExperiment(expName);

        List<Experiment> exps = client.listExperiments();
        Assert.assertEquals(exps.size(),1+expsBefore.size());

        java.util.Optional<Experiment> opt = getExperimentByName(exps,expName);
        Assert.assertTrue(opt.isPresent());
        Experiment expList = opt.get();
        Assert.assertEquals(expList.getName(),expName);

        GetExperiment.Response expGet = client.getExperiment(expId);
        Assert.assertEquals(expGet.getExperiment(),expList);
    }

    @Test
    public void addGetRun() throws Exception {
        // Create exp 
        String expName = createExperimentName();
        long expId = client.createExperiment(expName);
        logger.debug(">> TEST.0");
    
        // Create run 
        String user = System.getenv("USER");
        long startTime = System.currentTimeMillis();
        String sourceFile = "MyFile.java";
        CreateRun request = ObjectUtils.makeCreateRun(expId, "run_for_"+expId, SourceType.LOCAL, sourceFile, startTime, user);   

        RunInfo runCreated = client.createRun(request);
        runId = runCreated.getRunUuid();
        logger.debug("runId="+runId);

        // Log parameters
        client.logParameter(runId, "min_samples_leaf", TestShared.min_samples_leaf);
        client.logParameter(runId, "max_depth", TestShared.max_depth);
    
        // Log metrics
        client.logMetric(runId, "auc", TestShared.auc);
        client.logMetric(runId, "accuracy_score", TestShared.accuracy_score);
        client.logMetric(runId, "zero_one_loss", TestShared.zero_one_loss);

        // Update finished run
        client.updateRun(runId, RunStatus.FINISHED, startTime+1001);
  
        // Assert run from getExperiment
        GetExperiment.Response expResponse = client.getExperiment(expId);
        Experiment exp = expResponse.getExperiment() ;
        Assert.assertEquals(exp.getName(),expName);
        assertRunInfo(expResponse.getRunsList().get(0), expId, user, sourceFile);
        
        // Assert run from getRun
        Run run = client.getRun(runId);
        RunInfo runInfo = run.getInfo();
        assertRunInfo(runInfo, expId, user, sourceFile);
    }

    @Test (dependsOnMethods={"addGetRun"})
    public void checkParamsAndMetrics() throws Exception {
        TestShared.assertParamsAndMetrics(client, client.getRun(runId), runId);
    }

	@DataProvider
	public Object[][] searchParameterRequests() {
		return new Object[][]{
			{ "=",  "max_depth", "3" , 1},
            { "!=", "max_depth", "x" , 1},
			{ "=",  "max_depth", "x" , 0},
            { "!=", "max_depth", "3" , 0}
		};
    }

	@Test(dependsOnMethods={"addGetRun"}, dataProvider = "searchParameterRequests")
	public void testSearchParameters(String comparator, String key, String value, int numResults) throws Exception {
        String expectedValue = "3";
        SearchRuns.Response rsp = client.search(new long[] {expIdPythonScikitLearnTest}, new ParameterSearch[] { new ParameterSearch(key,comparator,value) });
        List<Run> runs = rsp.getRunsList();
        Assert.assertEquals(runs.size(),numResults);
        if (numResults > 0) {
            assertParam(runs.get(0).getData().getParamsList(),key,expectedValue);
        }
    }

	@DataProvider
	public Object[][] searchMetricRequests() {
		return new Object[][]{
			{ "=",   "auc", 2 , 1},
			{ ">",   "auc", 1 , 1},
			{ ">=",  "auc", 1 , 1},
			{ ">=",  "auc", 2 , 1},
			{ "<",   "auc", 3 , 1},
			{ "<=",  "auc", 3 , 1},
			{ "<=",  "auc", 2 , 1},
			{ "!=",  "auc", 1 , 1},
			{ "=",   "auc", 9 , 0},
			{ "!=",  "auc", 2 , 0},
			{ ">",   "auc", 9 , 0},
			{ ">=",  "auc", 9 , 0},
			{ "<",   "auc", 1 , 0},
			{ "<=",  "auc", 1 , 0}
		};
    }
    @Test(dependsOnMethods={"addGetRun"}, dataProvider = "searchMetricRequests")
    public void checkSearchMetrics(String comparator, String key, float value, int numResults) throws Exception {
        float expectedValue = 2F;
        SearchRuns.Response rsp = client.search(new long[] {expIdPythonScikitLearnTest}, new MetricSearch[] { new MetricSearch(key,comparator,value) });
        List<Run> runs = rsp.getRunsList();
        Assert.assertEquals(runs.size(),numResults);
        if (numResults > 0) {
            assertMetric(runs.get(0).getData().getMetricsList(),key,expectedValue);
        }
    }
    @DataProvider
    public Object[][] searchMixedRequests() {
        return new Object[][]{
            { "=",  "max_depth", "3" , "=",  "auc", 2 , 1},
            { "=",  "max_depth", "3" , "=",  "auc", 9 , 0},
            { "=",  "max_depth", "9" , "=",  "auc", 2 , 0},
            { "!=", "max_depth", "9" , "=",  "auc", 2 , 1},
            { "!=", "max_depth", "9" , "!=", "auc", 9 , 1},
            { "=",  "max_depth", "3" , ">=", "auc", 2 , 1},
            { "=",  "max_depth", "3" , "<=", "auc", 2 , 1},
            { "=",  "max_depth", "3" , ">",  "auc", 2 , 0},
            { "=",  "max_depth", "3" , "<",  "auc", 2 , 0},
        };
    }
    @Test(dependsOnMethods={"addGetRun"}, dataProvider = "searchMixedRequests")
    public void checkSearchMixed(String comparator1, String key1, String value1, String comparator2, String key2, float value2, int numResults) throws Exception {
        String expectedValue1 = "3";
        float expectedValue2 = 2F;
        SearchRuns.Response rsp = client.search(new long[] {expIdPythonScikitLearnTest}, new BaseSearch[] {
            new ParameterSearch(key1,comparator1,value1),
            new MetricSearch(key2,comparator2,value2) });
        List<Run> runs = rsp.getRunsList();
        Assert.assertEquals(runs.size(),numResults);
        if (numResults > 0) {
            assertParam(runs.get(0).getData().getParamsList(),key1,expectedValue1);
            assertMetric(runs.get(0).getData().getMetricsList(),key2,expectedValue2);
        }
    }
}
