package org.mlflow.client;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import org.apache.log4j.Logger;
import org.testng.Assert;
import org.testng.annotations.*;

import static org.mlflow.client.TestUtils.*;

import org.mlflow.api.proto.Service.*;
import org.mlflow.client.objects.*;

public class ApiClientTest {
  private static final Logger logger = Logger.getLogger(ApiClientTest.class);

  private static float ACCURACY_SCORE = 0.9733333333333334F;
  private static float ZERO_ONE_LOSS = 0.026666666666666616F;
  private static String MIN_SAMPLES_LEAF = "2";
  private static String MAX_DEPTH = "3";

  private final TestClientProvider testClientProvider = new TestClientProvider();
  private String runId;

  private ApiClient client;

  @BeforeSuite
  public void beforeAll() throws IOException, InterruptedException {
    client = testClientProvider.initializeClientAndServer();
  }

  @AfterSuite
  public void afterAll() throws InterruptedException {
    testClientProvider.cleanupClientAndServer();
  }

  @Test
  public void getCreateExperimentTest() throws Exception {
    String expName = createExperimentName();
    long expId = client.createExperiment(expName);
    GetExperiment.Response exp = client.getExperiment(expId);
    Assert.assertEquals(exp.getExperiment().getName(), expName);
  }

  @Test(expectedExceptions = HttpServerException.class) // TODO: server should throw 406
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
    Assert.assertEquals(exps.size(), 1 + expsBefore.size());

    java.util.Optional<Experiment> opt = getExperimentByName(exps, expName);
    Assert.assertTrue(opt.isPresent());
    Experiment expList = opt.get();
    Assert.assertEquals(expList.getName(), expName);

    GetExperiment.Response expGet = client.getExperiment(expId);
    Assert.assertEquals(expGet.getExperiment(), expList);
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
    CreateRun request = ObjectUtils.makeCreateRun(expId, "run_for_" + expId, SourceType.LOCAL, sourceFile, startTime, user);

    RunInfo runCreated = client.createRun(request);
    runId = runCreated.getRunUuid();
    logger.debug("runId=" + runId);

    // Log parameters
    client.logParameter(runId, "min_samples_leaf", MIN_SAMPLES_LEAF);
    client.logParameter(runId, "max_depth", MAX_DEPTH);

    // Log metrics
    client.logMetric(runId, "accuracy_score", ACCURACY_SCORE);
    client.logMetric(runId, "zero_one_loss", ZERO_ONE_LOSS);

    // Update finished run
    client.updateRun(runId, RunStatus.FINISHED, startTime + 1001);

    // Assert run from getExperiment
    GetExperiment.Response expResponse = client.getExperiment(expId);
    Experiment exp = expResponse.getExperiment();
    Assert.assertEquals(exp.getName(), expName);
    assertRunInfo(expResponse.getRunsList().get(0), expId, user, sourceFile);

    // Assert run from getRun
    Run run = client.getRun(runId);
    RunInfo runInfo = run.getInfo();
    assertRunInfo(runInfo, expId, user, sourceFile);
  }

  @Test(dependsOnMethods = {"addGetRun"})
  public void checkParamsAndMetrics() throws Exception {

    Run run = client.getRun(runId);
    List<Param> params = run.getData().getParamsList();
    Assert.assertEquals(params.size(), 2);
    assertParam(params, "min_samples_leaf", MIN_SAMPLES_LEAF);
    assertParam(params, "max_depth", MAX_DEPTH);

    List<Metric> metrics = run.getData().getMetricsList();
    Assert.assertEquals(metrics.size(), 2);
    assertMetric(metrics, "accuracy_score", ACCURACY_SCORE);
    assertMetric(metrics, "zero_one_loss", ZERO_ONE_LOSS);

    Metric m = client.getMetric(runId, "accuracy_score");
    Assert.assertEquals(m.getKey(), "accuracy_score");
    Assert.assertEquals(m.getValue(), ACCURACY_SCORE);

    metrics = client.getMetricHistory(runId, "accuracy_score");
    Assert.assertEquals(metrics.size(), 1);
    m = metrics.get(0);
    Assert.assertEquals(m.getKey(), "accuracy_score");
    Assert.assertEquals(m.getValue(), ACCURACY_SCORE);
  }
}
