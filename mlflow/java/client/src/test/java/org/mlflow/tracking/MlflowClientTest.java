package org.mlflow.tracking;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.Stack;
import java.util.Vector;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

import org.mlflow.api.proto.Service.CreateRun;
import org.mlflow.api.proto.Service.CreateExperiment;
import org.mlflow.api.proto.Service.Experiment;
import org.mlflow.api.proto.Service.ExperimentTag;
import org.mlflow.api.proto.Service.Metric;
import org.mlflow.api.proto.Service.Param;
import org.mlflow.api.proto.Service.Run;
import org.mlflow.api.proto.Service.RunInfo;
import org.mlflow.api.proto.Service.RunStatus;
import org.mlflow.api.proto.Service.RunTag;
import org.mlflow.api.proto.Service.ViewType;

import static org.mlflow.tracking.TestUtils.assertMetric;
import static org.mlflow.tracking.TestUtils.assertMetricHistory;
import static org.mlflow.tracking.TestUtils.assertParam;
import static org.mlflow.tracking.TestUtils.assertRunInfo;
import static org.mlflow.tracking.TestUtils.assertTag;
import static org.mlflow.tracking.TestUtils.createExperimentName;
import static org.mlflow.tracking.TestUtils.createMetric;
import static org.mlflow.tracking.TestUtils.createParam;
import static org.mlflow.tracking.TestUtils.createTag;
import static org.mlflow.tracking.TestUtils.getExperimentByName;

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
    String expId = client.createExperiment(expName);
    Experiment exp = client.getExperiment(expId);
    Assert.assertEquals(exp.getName(), expName);
  }

  @Test
  public void createExperimentWithTagsTest() {
    String expName = createExperimentName();
    CreateExperiment.Builder request = CreateExperiment.newBuilder();
    request.setName(expName);
    request.addTags(ExperimentTag.newBuilder().setKey("key1").setValue("val1").build());
    request.addTags(ExperimentTag.newBuilder().setKey("key2").setValue("val2").build());
    String expId = client.createExperiment(request.build());
    Experiment exp = client.getExperiment(expId);
    Assert.assertEquals(exp.getTagsCount(), 2);
    for (ExperimentTag tag : exp.getTagsList()) {
      if (tag.getKey().equals("key1")) {
        Assert.assertTrue(tag.getValue().equals("val1"));
      }
    }
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
    String expId = client.createExperiment(expName);
    Assert.assertEquals(client.getExperiment(expId).getLifecycleStage(), "active");

    client.deleteExperiment(expId);
    Assert.assertEquals(client.getExperiment(expId).getLifecycleStage(), "deleted");

    client.restoreExperiment(expId);
    Assert.assertEquals(client.getExperiment(expId).getLifecycleStage(), "active");
  }

  @Test
  public void renameExperiment() {
    String expName = createExperimentName();
    String newName = createExperimentName();

    String expId = client.createExperiment(expName);
    Assert.assertEquals(client.getExperiment(expId).getName(), expName);

    client.renameExperiment(expId, newName);
    Assert.assertEquals(client.getExperiment(expId).getName(), newName);
  }

  @Test
  public void searchExperimentsTest() {
    List<Experiment> expsBefore = client.searchExperiments().getItems();

    String expName1 = createExperimentName();
    String expId1 = client.createExperiment(expName1);
    client.setExperimentTag(expId1, "test", "test");
    client.setExperimentTag(expId1, "expgroup", "group1");

    String expName2 = createExperimentName();
    String expId2 = client.createExperiment(expName2);
    client.setExperimentTag(expId2, "test", "test");

    String expName3 = createExperimentName();
    String expId3 = client.createExperiment(expName3);
    client.setExperimentTag(expId3, "test", "test");
    client.setExperimentTag(expId3, "expgroup", "group1");

    List<Experiment> exps = client.searchExperiments().getItems();
    Assert.assertEquals(exps.size(), 3 + expsBefore.size());

    String exp1Filter = String.format("attribute.name = '%s'", expName1);
    List<Experiment> exps1 = client.searchExperiments(exp1Filter).getItems();
    Assert.assertEquals(exps1.size(), 1);
    Assert.assertEquals(exps1.get(0).getExperimentId(), expId1);

    String exp2Filter = String.format("attribute.name = '%s'", expName2);
    List<Experiment> exps2 = client.searchExperiments(exp2Filter).getItems();
    Assert.assertEquals(exps2.size(), 1);
    Assert.assertEquals(exps2.get(0).getExperimentId(), expId2);

    String expGroupFilter = String.format("tags.expgroup = 'group1'");
    List<Experiment> expGroup = client.searchExperiments(expGroupFilter).getItems();
    Assert.assertEquals(
      expGroup.stream().map(exp -> exp.getExperimentId()).collect(Collectors.toSet()),
      new HashSet<>(Arrays.asList(expId1, expId3))
    );

    client.deleteExperiment(expId2);

    List<Experiment> activeExps = client.searchExperiments("").getItems();
    Set<String> activeExpIds = activeExps.stream().map(
        exp -> exp.getExperimentId()
    ).collect(Collectors.toSet());
    Assert.assertTrue(activeExpIds.contains(expId1));
    Assert.assertTrue(activeExpIds.contains(expId3));
    Assert.assertFalse(activeExpIds.contains(expId2));

    List<Experiment> deletedExps = client.searchExperiments(
        "", ViewType.DELETED_ONLY, 10, new ArrayList<>()
    ).getItems();
    Assert.assertEquals(deletedExps.size(), 1);
    Assert.assertEquals(deletedExps.get(0).getExperimentId(), expId2);

    List<String> orderedExpNames = Arrays.asList(expName1, expName2, expName3);
    Collections.sort(orderedExpNames);

    ExperimentsPage page1 = client.searchExperiments(
      "tags.test = 'test'", ViewType.ALL, 1, Arrays.asList("attribute.name")
    );
    Assert.assertEquals(page1.getItems().size(), 1);
    Assert.assertEquals(page1.getItems().get(0).getName(), orderedExpNames.get(0));
    Assert.assertTrue(page1.getNextPageToken().isPresent());

    ExperimentsPage page2 = client.searchExperiments(
      "tags.test = 'test'",
      ViewType.ALL,
      2,
      Arrays.asList("attribute.name"),
      page1.getNextPageToken().get()
    );
    Assert.assertEquals(page2.getItems().size(), 2);
    Assert.assertEquals(page2.getItems().get(0).getName(), orderedExpNames.get(1));
    Assert.assertEquals(page2.getItems().get(1).getName(), orderedExpNames.get(2));
    Assert.assertFalse(page2.getNextPageToken().isPresent());

    ExperimentsPage nextPageFromPrevPage = (ExperimentsPage) page1.getNextPage();
    Assert.assertEquals(nextPageFromPrevPage.getItems().size(), 1);
    Assert.assertEquals(nextPageFromPrevPage.getItems().get(0).getName(), orderedExpNames.get(1));
    Assert.assertTrue(nextPageFromPrevPage.getNextPageToken().isPresent());
  }

  @Test
  public void addGetRun() {
    // Create exp
    String expName = createExperimentName();
    String expId = client.createExperiment(expName);
    logger.debug(">> TEST.0");

    // Create run
    long startTime = System.currentTimeMillis();

    RunInfo runCreated = client.createRun(expId);
    runId = runCreated.getRunUuid();
    logger.debug("runId=" + runId);

    List<RunInfo> runInfos = client.listRunInfos(expId);
    Assert.assertEquals(runInfos.size(), 1);
    Assert.assertEquals(runInfos.get(0).getStatus(), RunStatus.RUNNING);

    // Log parameters
    client.logParam(runId, "min_samples_leaf", MIN_SAMPLES_LEAF);
    client.logParam(runId, "max_depth", MAX_DEPTH);

    // Log metrics
    client.logMetric(runId, "accuracy_score", ACCURACY_SCORE);
    client.logMetric(runId, "zero_one_loss", ZERO_ONE_LOSS);
    client.logMetric(runId, "multi_log_default_step_ts", 2.0);
    client.logMetric(runId, "multi_log_default_step_ts", -1.0);
    client.logMetric(runId, "multi_log_specified_step_ts", 1.0, 1000, 1);
    client.logMetric(runId, "multi_log_specified_step_ts", 2.0, 2000, -5);
    client.logMetric(runId, "multi_log_specified_step_ts", -3.0, 3000, 4);
    client.logMetric(runId, "multi_log_specified_step_ts", 4.0, 2999, 4);

    // Log NaNs and Infs
    client.logMetric(runId, "nan_metric", java.lang.Double.NaN);
    client.logMetric(runId, "pos_inf", java.lang.Double.POSITIVE_INFINITY);
    client.logMetric(runId, "neg_inf", java.lang.Double.NEGATIVE_INFINITY);

    // Log tag
    client.setTag(runId, "user_email", USER_EMAIL);

    // Update finished run
    client.setTerminated(runId, RunStatus.FINISHED);
    long endTime = System.currentTimeMillis();

    List<RunInfo> updatedRunInfos = client.listRunInfos(expId);
    Assert.assertEquals(updatedRunInfos.size(), 1);
    Assert.assertEquals(updatedRunInfos.get(0).getStatus(), RunStatus.FINISHED);

    // Assert run from getExperiment
    Experiment exp = client.getExperiment(expId);
    Assert.assertEquals(exp.getName(), expName);

    // Assert run from getRun
    Run run = client.getRun(runId);
    RunInfo runInfo = run.getInfo();
    assertRunInfo(runInfo, expId);
    // verify run start and end are set in ms
    Assert.assertTrue(runInfo.getStartTime() >= startTime);
    Assert.assertTrue(runInfo.getEndTime() <= endTime);

    // Assert parent run ID is not set.
    Assert.assertTrue(run.getData().getTagsList().stream().noneMatch(
            tag -> tag.getKey().equals("mlflow.parentRunId")));
  }

  @Test
  public void setExperimentTag() {
    // Create experiment
    String expName = createExperimentName();
    String expId = client.createExperiment(expName);
    client.setExperimentTag(expId, "dataset", "imagenet1K");
    Experiment exp = client.getExperiment(expId);
    Assert.assertTrue(exp.getTagsCount() == 1);
    Assert.assertTrue(exp.getTags(0).getKey().equals("dataset"));
    Assert.assertTrue(exp.getTags(0).getValue().equals("imagenet1K"));
    // test updating experiment tag
    client.setExperimentTag(expId, "dataset", "birdbike");
    exp = client.getExperiment(expId);
    Assert.assertTrue(exp.getTagsCount() == 1);
    Assert.assertTrue(exp.getTags(0).getKey().equals("dataset"));
    Assert.assertTrue(exp.getTags(0).getValue().equals("birdbike"));
    // test that setting a tag on 1 experiment does not impact another experiment.
    String expId2 = client.createExperiment("randomExperimentName");
    Experiment exp2 = client.getExperiment(expId2);
    Assert.assertTrue(exp2.getTagsCount() == 0);
    // test that setting a tag on different experiments maintain different values across experiments
    client.setExperimentTag(expId2, "dataset", "birds200");
    exp = client.getExperiment(expId);
    exp2 = client.getExperiment(expId2);
    Assert.assertTrue(exp.getTagsCount() == 1);
    Assert.assertTrue(exp.getTags(0).getKey().equals("dataset"));
    Assert.assertTrue(exp.getTags(0).getValue().equals("birdbike"));
    Assert.assertTrue(exp2.getTagsCount() == 1);
    Assert.assertTrue(exp2.getTags(0).getKey().equals("dataset"));
    Assert.assertTrue(exp2.getTags(0).getValue().equals("birds200"));
    // test can set multi-line tags
    client.setExperimentTag(expId, "multiline tag", "value2\nvalue2\nvalue2");
    exp = client.getExperiment(expId);
    Assert.assertTrue(exp.getTagsCount() == 2);
    for (ExperimentTag tag : exp.getTagsList()) {
      if (tag.getKey().equals("multiline tag")) {
        Assert.assertTrue(tag.getValue().equals("value2\nvalue2\nvalue2"));
      }
    }
  }

  @Test
  public void deleteTag() {
    // Create experiment
    String expName = createExperimentName();
    String expId = client.createExperiment(expName);

    // Create run
    RunInfo runCreated = client.createRun(expId);
    String runId = runCreated.getRunUuid();
    client.setTag(runId, "tag0", "val0");
    client.setTag(runId, "tag1", "val1");
    client.deleteTag(runId, "tag0");
    Run run = client.getRun(runId);
    // test that the tag was correctly deleted.
    for (RunTag rt : run.getData().getTagsList()) {
      Assert.assertTrue(!rt.getKey().equals("tag0"));
    }
    // test that you can't re-delete the old tag
    try {
      client.deleteTag(runId, "tag0");
      Assert.fail();
    } catch (MlflowClientException e) {
      Assert.assertTrue(e.getMessage().contains(String.format("No tag with name: tag0 in run with id %s", runId)));
    }
    // test that you can't delete a tag that doesn't already exist.
    try {
      client.deleteTag(runId, "fakeTag");
      Assert.fail();
    } catch (MlflowClientException e) {
      Assert.assertTrue(e.getMessage().contains(String.format("No tag with name: fakeTag in run with id %s", runId)));
    }
    // test that you can't delete a tag on a nonexistent run.
    try {
      client.deleteTag("fakeRunId", "fakeTag");
      Assert.fail();
    } catch (MlflowClientException e) {
      Assert.assertTrue(e.getMessage().contains(String.format("Run '%s' not found", "fakeRunId")));
    }
  }

  @Test
  public void searchRuns() {
    // Create exp
    String expName = createExperimentName();
    String expId = client.createExperiment(expName);
    logger.debug(">> TEST.0");

    // Create run
    String user = System.getenv("USER");
    long startTime = System.currentTimeMillis();
    String sourceFile = "MyFile.java";

    RunInfo runCreated_1 = client.createRun(expId);
    String runId_1 = runCreated_1.getRunUuid();
    logger.debug("runId=" + runId_1);

    RunInfo runCreated_2 = client.createRun(expId);
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

    List<String> experimentIds = Arrays.asList(expId);

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

    // Paged searchRuns

    List<Run> searchRuns = Lists.newArrayList(client.searchRuns(experimentIds, "", 
            ViewType.ACTIVE_ONLY, 1000, Lists.newArrayList("metrics.accuracy_score")).getItems());
    Assert.assertEquals(searchRuns.get(0).getInfo().getRunUuid(), runId_1);
    Assert.assertEquals(searchRuns.get(1).getInfo().getRunUuid(), runId_2);

    searchRuns = Lists.newArrayList(client.searchRuns(experimentIds, "", ViewType.ACTIVE_ONLY,
            1000, Lists.newArrayList("params.min_samples_leaf", "metrics.accuracy_score DESC"))
            .getItems());
    Assert.assertEquals(searchRuns.get(1).getInfo().getRunUuid(), runId_1);
    Assert.assertEquals(searchRuns.get(0).getInfo().getRunUuid(), runId_2);

    Page<Run> page = client.searchRuns(experimentIds, "", ViewType.ACTIVE_ONLY, 1000);
    Assert.assertEquals(page.getPageSize(), 2);
    Assert.assertEquals(page.hasNextPage(), false);
    Assert.assertEquals(page.getNextPageToken(), Optional.empty());

    page = client.searchRuns(experimentIds, "", ViewType.ACTIVE_ONLY, 1);
    Assert.assertEquals(page.getPageSize(), 1);
    Assert.assertEquals(page.hasNextPage(), true);
    Assert.assertNotEquals(page.getNextPageToken(), Optional.empty());

    Page<Run> page2 = page.getNextPage();
    Assert.assertEquals(page2.getPageSize(), 1);
    Assert.assertEquals(page2.hasNextPage(), false);
    Assert.assertEquals(page2.getNextPageToken(), Optional.empty());

    Page<Run> page3 = page2.getNextPage();
    Assert.assertEquals(page3.getPageSize(), 0);
    Assert.assertEquals(page3.getNextPageToken(), Optional.empty());
  }

  @Test
  public void createRunWithParent() {
    String expName = createExperimentName();
    String expId = client.createExperiment(expName);
    RunInfo parentRun = client.createRun(expId);
    String parentRunId = parentRun.getRunUuid();
    RunInfo childRun = client.createRun(CreateRun.newBuilder()
    .setExperimentId(expId)
    .build());
    client.setTag(childRun.getRunUuid(), "mlflow.parentRunId", parentRunId);
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
    Assert.assertEquals(metrics.size(), 7);
    assertMetric(metrics, "accuracy_score", ACCURACY_SCORE);
    assertMetric(metrics, "zero_one_loss", ZERO_ONE_LOSS);
    assertMetric(metrics, "multi_log_default_step_ts", -1.0);
    assertMetric(metrics, "multi_log_specified_step_ts", -3.0);
    assertMetric(metrics, "nan_metric", Double.NaN);
    assertMetric(metrics, "pos_inf", Double.POSITIVE_INFINITY);
    assertMetric(metrics, "neg_inf", Double.NEGATIVE_INFINITY);
    assert(metrics.get(0).getTimestamp() > 0) : metrics.get(0).getTimestamp();

    List<Metric> multiDefaultMetricHistory = client.getMetricHistory(
      runId, "multi_log_default_step_ts");
    assertMetricHistory(multiDefaultMetricHistory, "multi_log_default_step_ts",
      Arrays.asList(2.0, -1.0), Arrays.asList(0L, 0L));

    List<Metric> multiSpecifiedMetricHistory = client.getMetricHistory(
      runId, "multi_log_specified_step_ts");
    assertMetricHistory(multiSpecifiedMetricHistory, "multi_log_specified_step_ts",
      Arrays.asList(1.0, 2.0, -3.0, 4.0), Arrays.asList(1000L, 2000L, 3000L, 2999L),
      Arrays.asList(1L, -5L, 4L, 4L));

    List<RunTag> tags = run.getData().getTagsList();
    Assert.assertEquals(tags.size(), 2);
    assertTag(tags, "user_email", USER_EMAIL);
  }

  @Test
  public void getMetricHistoryPagination() {
    String expId = client.createExperiment("getMetricHistoryPagination");
    RunInfo run = client.createRun(expId);
    String runId = run.getRunUuid();
    List<Long> steps = LongStream.range(0, 26000).boxed().collect(Collectors.toList());
    for (Long step: steps) {
      client.logMetric(runId, "random_metric", step, step, step);
    }
    List<Metric> metrics = client.getMetricHistory(runId, "random_metric");
    List<Double> values = steps.stream().mapToDouble(x -> x).boxed().collect(Collectors.toList());
    assertMetricHistory(metrics, "random_metric", values, steps);
  }

  @Test
  public void testBatchedLogging() {
    // Create exp
    String expName = createExperimentName();
    String expId = client.createExperiment(expName);
    logger.debug(">> TEST.0");

    // Test logging just metrics
    {
      RunInfo runCreated = client.createRun(expId);
      String runUuid = runCreated.getRunId();
      logger.debug("runUuid=" + runUuid);

      List<Metric> metrics = new ArrayList<>(Arrays.asList(createMetric("met1", 0.081D, 10, 0),
        createMetric("metric2", 82.3D, 100, 73), createMetric("metric3", 1.0D, 1000, 1),
        createMetric("metric3", 2.0D, 2000, 3), createMetric("metric3", 3.0D, 0, -2)));
      client.logBatch(runUuid, metrics, null, null);

      Run run = client.getRun(runUuid);
      Assert.assertEquals(run.getInfo().getRunId(), runUuid);

      List<Metric> loggedMetrics = run.getData().getMetricsList();
      Assert.assertEquals(loggedMetrics.size(), 3);
      assertMetric(loggedMetrics, "met1", 0.081D, 10, 0);
      assertMetric(loggedMetrics, "metric2", 82.3D, 100, 73);
      assertMetric(loggedMetrics, "metric3", 2.0D, 2000, 3);
    }

    // Test logging just params
    {
      RunInfo runCreated = client.createRun(expId);
      String runUuid = runCreated.getRunId();
      logger.debug("runUuid=" + runUuid);

      Set<Param> params = new HashSet<Param>(Arrays.asList(
        createParam("p1", "this is a param string"),
        createParam("p2", "a b"),
        createParam("3", "x")));
      client.logBatch(runUuid, null, params, null);

      Run run = client.getRun(runUuid);
      Assert.assertEquals(run.getInfo().getRunId(), runUuid);

      List<Param> loggedParams = run.getData().getParamsList();
      Assert.assertEquals(loggedParams.size(), 3);
      assertParam(loggedParams, "p1", "this is a param string");
      assertParam(loggedParams, "p2", "a b");
      assertParam(loggedParams, "3", "x");
    }

    // Test logging just tags
    {
      RunInfo runCreated = client.createRun(expId);
      String runUuid = runCreated.getRunId();
      logger.debug("runUuid=" + runUuid);

      Stack<RunTag> tags = new Stack();
      tags.push(createTag("t1", "tagtagtag"));
      tags.push(createTag("mlflow.runName", "myRun"));
      client.logBatch(runUuid, null, null, tags);

      Run run = client.getRun(runUuid);
      Assert.assertEquals(run.getInfo().getRunId(), runUuid);

      List<RunTag> loggedTags = run.getData().getTagsList();
      Assert.assertEquals(loggedTags.size(), 2);
      assertTag(loggedTags, "t1", "tagtagtag");
      assertTag(loggedTags, "mlflow.runName", "myRun");
    }

    // All
    {
      RunInfo runCreated = client.createRun(expId);
      String runUuid = runCreated.getRunId();
      logger.debug("runUuid=" + runUuid);

      List<Metric> metrics = new LinkedList<>(Arrays.asList(createMetric("m1", 32.23D, 12, 0)));
      Vector<Param> params = new Vector<>(Arrays.asList(createParam("p1", "param1"),
        createParam("p2", "another param")));
      Set<RunTag> tags = new HashSet<>(Arrays.asList(createTag("t1", "t1"),
        createTag("t2", "xx"),
        createTag("t3", "xx"),
        createTag("mlflow.runName", "myRun")));
      client.logBatch(runUuid, metrics, params, tags);

      Run run = client.getRun(runUuid);
      Assert.assertEquals(run.getInfo().getRunId(), runUuid);

      List<Metric> loggedMetrics = run.getData().getMetricsList();
      Assert.assertEquals(loggedMetrics.size(), 1);
      assertMetric(loggedMetrics, "m1", 32.23D);

      List<Param> loggedParams = run.getData().getParamsList();
      Assert.assertEquals(loggedParams.size(), 2);
      assertParam(loggedParams, "p1", "param1");
      assertParam(loggedParams, "p2", "another param");

      List<RunTag> loggedTags = run.getData().getTagsList();
      Assert.assertEquals(loggedTags.size(), 4);
      assertTag(loggedTags, "t1", "t1");
      assertTag(loggedTags, "t2", "xx");
      assertTag(loggedTags, "t3", "xx");
      assertTag(loggedTags, "mlflow.runName", "myRun");
    }
  }

  @Test
  public void deleteAndRestoreRun() {
    String expName = createExperimentName();
    String expId = client.createExperiment(expName);

    String sourceFile = "MyFile.java";

    RunInfo runCreated = client.createRun(expId);
    Assert.assertEquals(runCreated.getLifecycleStage(), "active");
    String deleteRunId = runCreated.getRunId();
    client.deleteRun(deleteRunId);
    Assert.assertEquals(client.getRun(deleteRunId).getInfo().getLifecycleStage(), "deleted");
    client.restoreRun(deleteRunId);
    Assert.assertEquals(client.getRun(deleteRunId).getInfo().getLifecycleStage(), "active");
  }

  @Test
  public void testUseArtifactRepository() throws IOException {
    String content = "Hello, Worldz!";

    File tempFile = Files.createTempFile(getClass().getSimpleName(), ".txt").toFile();
    File tempDir = Files.createTempDirectory("tempDir").toFile();
    File tempFileForDir = Files.createTempFile(tempDir.toPath(), "file", ".txt").toFile();

    FileUtils.writeStringToFile(tempFileForDir, content, StandardCharsets.UTF_8);
    FileUtils.writeStringToFile(tempFile, content, StandardCharsets.UTF_8);
    client.logArtifact(runId, tempFile);
    client.logArtifact(runId, tempDir);

    File downloadedArtifact = client.downloadArtifacts(runId, tempFile.getName());
    File downloadedArtifactFromDir = client.downloadArtifacts(runId, tempDir.getName() + "/" +
      tempFileForDir.getName());
    String downloadedContent = FileUtils.readFileToString(downloadedArtifact,
      StandardCharsets.UTF_8);
    String downloadedContentFromDir = FileUtils.readFileToString(downloadedArtifactFromDir,
      StandardCharsets.UTF_8);
    Assert.assertEquals(content, downloadedContent);
    Assert.assertEquals(content, downloadedContentFromDir);
  }
}
