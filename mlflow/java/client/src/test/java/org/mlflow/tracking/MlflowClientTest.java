package org.mlflow.tracking;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.HashSet;
import java.util.Stack;
import java.util.Vector;
import java.util.LinkedList;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

import org.mlflow.api.proto.Service.*;

import static org.mlflow.tracking.TestUtils.*;

public class MlflowClientTest {
  private static final Logger logger = LoggerFactory.getLogger(MlflowClientTest.class);

  private static double ACCURACY_SCORE = 0.9733333333333334D;
  // NB: This can only be represented as a double (not float)
  private static double ZERO_ONE_LOSS = 123.456789123456789D;
  private static String MIN_SAMPLES_LEAF = "2";
  private static String MAX_DEPTH = "3";
  private static String USER_EMAIL = "some@email.com";

  private final TestClientProvider testClientProvider = new TestClientProvider();
  private String runId;

  private MlflowClient client;

  @BeforeSuite
  public void beforeAll() throws IOException {
    client = testClientProvider.initializeClientAndServer();
  }

  @AfterSuite
  public void afterAll() throws InterruptedException {
    testClientProvider.cleanupClientAndServer();
  }

  @Test
  public void getCreateExperimentTest() {
    String expName = createExperimentName();
    long expId = client.createExperiment(expName);
    GetExperiment.Response exp = client.getExperiment(expId);
    Assert.assertEquals(exp.getExperiment().getName(), expName);
  }

  @Test(expectedExceptions = MlflowClientException.class) // TODO: server should throw 406
  public void createExistingExperiment() {
    String expName = createExperimentName();
    client.createExperiment(expName);
    client.createExperiment(expName);
  }

  @Test
  public void deleteAndRestoreExperiments() {
    String expName = createExperimentName();
    long expId = client.createExperiment(expName);
    Assert.assertEquals(client.getExperiment(expId).getExperiment().getLifecycleStage(), "active");

    client.deleteExperiment(expId);
    Assert.assertEquals(client.getExperiment(expId).getExperiment().getLifecycleStage(), "deleted");

    client.restoreExperiment(expId);
    Assert.assertEquals(client.getExperiment(expId).getExperiment().getLifecycleStage(), "active");
  }

  @Test
  public void renameExperiment() {
    String expName = createExperimentName();
    String newName = createExperimentName();

    long expId = client.createExperiment(expName);
    Assert.assertEquals(client.getExperiment(expId).getExperiment().getName(), expName);

    client.renameExperiment(expId, newName);
    Assert.assertEquals(client.getExperiment(expId).getExperiment().getName(), newName);
  }

  @Test
  public void listExperimentsTest() {
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
  public void addGetRun() {
    // Create exp
    String expName = createExperimentName();
    long expId = client.createExperiment(expName);
    logger.debug(">> TEST.0");

    // Create run
    String user = System.getenv("USER");
    long startTime = System.currentTimeMillis();
    String sourceFile = "MyFile.java";

    RunInfo runCreated = client.createRun(expId, sourceFile);
    runId = runCreated.getRunUuid();
    logger.debug("runId=" + runId);

    List<RunInfo> runInfos = client.listRunInfos(expId);
    Assert.assertEquals(runInfos.size(), 1);
    Assert.assertEquals(runInfos.get(0).getSourceType(), SourceType.LOCAL);
    Assert.assertEquals(runInfos.get(0).getStatus(), RunStatus.RUNNING);

    // Log parameters
    client.logParam(runId, "min_samples_leaf", MIN_SAMPLES_LEAF);
    client.logParam(runId, "max_depth", MAX_DEPTH);

    // Log metrics
    client.logMetric(runId, "accuracy_score", ACCURACY_SCORE);
    client.logMetric(runId, "zero_one_loss", ZERO_ONE_LOSS);

    // Log tag
    client.setTag(runId, "user_email", USER_EMAIL);

    // Update finished run
    client.setTerminated(runId, RunStatus.FINISHED, startTime + 1001);

    List<RunInfo> updatedRunInfos = client.listRunInfos(expId);
    Assert.assertEquals(updatedRunInfos.size(), 1);
    Assert.assertEquals(updatedRunInfos.get(0).getStatus(), RunStatus.FINISHED);

    // Assert run from getExperiment
    GetExperiment.Response expResponse = client.getExperiment(expId);
    Experiment exp = expResponse.getExperiment();
    Assert.assertEquals(exp.getName(), expName);

    // Assert run from getRun
    Run run = client.getRun(runId);
    RunInfo runInfo = run.getInfo();
    assertRunInfo(runInfo, expId, sourceFile);

    // Assert parent run ID is not set.
    Assert.assertTrue(run.getData().getTagsList().stream().noneMatch(
            tag -> tag.getKey().equals("mlflow.parentRunId")));
  }

  @Test
  public void searchRuns() {
    // Create exp
    String expName = createExperimentName();
    long expId = client.createExperiment(expName);
    logger.debug(">> TEST.0");

    // Create run
    String user = System.getenv("USER");
    long startTime = System.currentTimeMillis();
    String sourceFile = "MyFile.java";

    RunInfo runCreated_1 = client.createRun(expId, sourceFile);
    String runId_1 = runCreated_1.getRunUuid();
    logger.debug("runId=" + runId_1);

    RunInfo runCreated_2 = client.createRun(expId, sourceFile);
    String runId_2 = runCreated_2.getRunUuid();
    logger.debug("runId=" + runId_2);

    // Log parameters
    client.logParam(runId_1, "min_samples_leaf", MIN_SAMPLES_LEAF);
    client.logParam(runId_2, "min_samples_leaf", MIN_SAMPLES_LEAF);

    client.logParam(runId_1, "max_depth", "5");
    client.logParam(runId_2, "max_depth", "15");

    // Log metrics
    client.logMetric(runId_1, "accuracy_score", 0.1);
    client.logMetric(runId_1, "accuracy_score", 0.4);
    client.logMetric(runId_2, "accuracy_score", 0.9);

    // Log tag
    client.setTag(runId_1, "user_email", USER_EMAIL);
    client.setTag(runId_1, "test", "works");
    client.setTag(runId_2, "test", "also works");

    List<Long> experimentIds = Arrays.asList(expId);

    // metrics based searches
    List<RunInfo> searchResult = client.searchRuns(experimentIds, "metrics.accuracy_score < 0");
    Assert.assertEquals(searchResult.size(), 0);

    searchResult = client.searchRuns(experimentIds, "metrics.accuracy_score > 0");
    Assert.assertEquals(searchResult.size(), 2);

    searchResult = client.searchRuns(experimentIds, "metrics.accuracy_score < 0.3");
    Assert.assertEquals(searchResult.size(), 0);

    searchResult = client.searchRuns(experimentIds, "metrics.accuracy_score < 0.5");
    Assert.assertEquals(searchResult.get(0).getRunUuid(), runId_1);

    searchResult = client.searchRuns(experimentIds, "metrics.accuracy_score > 0.5");
    Assert.assertEquals(searchResult.get(0).getRunUuid(), runId_2);

    // parameter based searches
    searchResult = client.searchRuns(experimentIds,
            "params.min_samples_leaf = '" + MIN_SAMPLES_LEAF + "'");
    Assert.assertEquals(searchResult.size(), 2);
    searchResult = client.searchRuns(experimentIds,
            "params.min_samples_leaf != '" + MIN_SAMPLES_LEAF + "'");
    Assert.assertEquals(searchResult.size(), 0);
    searchResult = client.searchRuns(experimentIds, "params.max_depth = '5'");
    Assert.assertEquals(searchResult.get(0).getRunUuid(), runId_1);

    searchResult = client.searchRuns(experimentIds, "params.max_depth = '15'");
    Assert.assertEquals(searchResult.get(0).getRunUuid(), runId_2);

    // tag based search
    searchResult = client.searchRuns(experimentIds, "tag.user_email = '" + USER_EMAIL + "'");
    Assert.assertEquals(searchResult.get(0).getRunUuid(), runId_1);

    searchResult = client.searchRuns(experimentIds, "tag.user_email != '" + USER_EMAIL + "'");
    Assert.assertEquals(searchResult.size(), 0);

    searchResult = client.searchRuns(experimentIds, "tag.test = 'works'");
    Assert.assertEquals(searchResult.get(0).getRunUuid(), runId_1);

    searchResult = client.searchRuns(experimentIds, "tag.test = 'also works'");
    Assert.assertEquals(searchResult.get(0).getRunUuid(), runId_2);
  }

  @Test
  public void createRunWithParent() {
    String expName = createExperimentName();
    long expId = client.createExperiment(expName);
    RunInfo parentRun = client.createRun(expId);
    String parentRunId = parentRun.getRunUuid();
    RunInfo childRun = client.createRun(CreateRun.newBuilder()
    .setExperimentId(expId)
    .setParentRunId(parentRunId)
    .build());
    List<RunTag> childTags = client.getRun(childRun.getRunUuid()).getData().getTagsList();
    String parentRunIdTagValue = childTags.stream()
      .filter(t -> t.getKey().equals("mlflow.parentRunId"))
      .findFirst()
      .get()
      .getValue();
    Assert.assertEquals(parentRunIdTagValue, parentRunId);
  }

  @Test(dependsOnMethods = {"addGetRun"})
  public void checkParamsAndMetrics() {

    Run run = client.getRun(runId);
    List<Param> params = run.getData().getParamsList();
    Assert.assertEquals(params.size(), 2);
    assertParam(params, "min_samples_leaf", MIN_SAMPLES_LEAF);
    assertParam(params, "max_depth", MAX_DEPTH);

    List<Metric> metrics = run.getData().getMetricsList();
    Assert.assertEquals(metrics.size(), 2);
    assertMetric(metrics, "accuracy_score", ACCURACY_SCORE);
    assertMetric(metrics, "zero_one_loss", ZERO_ONE_LOSS);
    assert(metrics.get(0).getTimestamp() > 0) : metrics.get(0).getTimestamp();

    List<RunTag> tags = run.getData().getTagsList();
    Assert.assertEquals(tags.size(), 1);
    assertTag(tags, "user_email", USER_EMAIL);
  }

  @Test
  public void testBatchedLogging() {
    // Create exp
    String expName = createExperimentName();
    long expId = client.createExperiment(expName);
    logger.debug(">> TEST.0");

    // Test logging just metrics
    {
      RunInfo runCreated = client.createRun(expId);
      String runUuid = runCreated.getRunUuid();
      logger.debug("runUuid=" + runUuid);

      List<Metric> metrics = new ArrayList<>(Arrays.asList(createMetric("met1", 0.081D, 10),
        createMetric("metric2", 82.3D, 100)));
      client.logBatch(runUuid, metrics, null, null);

      Run run = client.getRun(runUuid);
      Assert.assertEquals(run.getInfo().getRunUuid(), runUuid);

      List<Metric> loggedMetrics = run.getData().getMetricsList();
      Assert.assertEquals(loggedMetrics.size(), 2);
      assertMetric(loggedMetrics, "met1", 0.081D);
      assertMetric(loggedMetrics, "metric2", 82.3D);
    }

    // Test logging just params
    {
      RunInfo runCreated = client.createRun(expId);
      String runUuid = runCreated.getRunUuid();
      logger.debug("runUuid=" + runUuid);

      Set<Param> params = new HashSet<Param>(Arrays.asList(
        createParam("p1", "this is a param string"),
        createParam("p2", "a b"),
        createParam("3", "x")));
      client.logBatch(runUuid, null, params, null);

      Run run = client.getRun(runUuid);
      Assert.assertEquals(run.getInfo().getRunUuid(), runUuid);

      List<Param> loggedParams = run.getData().getParamsList();
      Assert.assertEquals(loggedParams.size(), 3);
      assertParam(loggedParams, "p1", "this is a param string");
      assertParam(loggedParams, "p2", "a b");
      assertParam(loggedParams, "3", "x");
    }

    // Test logging just tags
    {
      RunInfo runCreated = client.createRun(expId);
      String runUuid = runCreated.getRunUuid();
      logger.debug("runUuid=" + runUuid);

      Stack<RunTag> tags = new Stack();
      tags.push(createTag("t1", "tagtagtag"));
      client.logBatch(runUuid, null, null, tags);

      Run run = client.getRun(runUuid);
      Assert.assertEquals(run.getInfo().getRunUuid(), runUuid);

      List<RunTag> loggedTags = run.getData().getTagsList();
      Assert.assertEquals(loggedTags.size(), 1);
      assertTag(loggedTags, "t1", "tagtagtag");
    }

    // All
    {
      RunInfo runCreated = client.createRun(expId);
      String runUuid = runCreated.getRunUuid();
      logger.debug("runUuid=" + runUuid);

      List<Metric> metrics = new LinkedList<>(Arrays.asList(createMetric("m1", 32.23D, 12)));
      Vector<Param> params = new Vector<>(Arrays.asList(createParam("p1", "param1"),
        createParam("p2", "another param")));
      Set<RunTag> tags = new HashSet<>(Arrays.asList(createTag("t1", "t1"),
        createTag("t2", "xx"),
        createTag("t3", "xx")));
      client.logBatch(runUuid, metrics, params, tags);

      Run run = client.getRun(runUuid);
      Assert.assertEquals(run.getInfo().getRunUuid(), runUuid);

      List<Metric> loggedMetrics = run.getData().getMetricsList();
      Assert.assertEquals(loggedMetrics.size(), 1);
      assertMetric(loggedMetrics, "m1", 32.23D);

      List<Param> loggedParams = run.getData().getParamsList();
      Assert.assertEquals(loggedParams.size(), 2);
      assertParam(loggedParams, "p1", "param1");
      assertParam(loggedParams, "p2", "another param");

      List<RunTag> loggedTags = run.getData().getTagsList();
      Assert.assertEquals(loggedTags.size(), 3);
      assertTag(loggedTags, "t1", "t1");
      assertTag(loggedTags, "t2", "xx");
      assertTag(loggedTags, "t3", "xx");
    }
  }

  @Test
  public void deleteAndRestoreRun() {
    String expName = createExperimentName();
    long expId = client.createExperiment(expName);

    String sourceFile = "MyFile.java";

    RunInfo runCreated = client.createRun(expId, sourceFile);
    Assert.assertEquals(runCreated.getLifecycleStage(), "active");
    String deleteRunId = runCreated.getRunUuid();
    client.deleteRun(deleteRunId);
    Assert.assertEquals(client.getRun(deleteRunId).getInfo().getLifecycleStage(), "deleted");
    client.restoreRun(deleteRunId);
    Assert.assertEquals(client.getRun(deleteRunId).getInfo().getLifecycleStage(), "active");
  }

  @Test
  public void testUseArtifactRepository() throws IOException {
    String content = "Hello, Worldz!";

    File tempFile = Files.createTempFile(getClass().getSimpleName(), ".txt").toFile();
    FileUtils.writeStringToFile(tempFile, content, StandardCharsets.UTF_8);
    client.logArtifact(runId, tempFile);

    File downloadedArtifact = client.downloadArtifacts(runId, tempFile.getName());
    String downloadedContent = FileUtils.readFileToString(downloadedArtifact,
      StandardCharsets.UTF_8);
    Assert.assertEquals(content, downloadedContent);
  }
}
