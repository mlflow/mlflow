package com.databricks.mlflow.client.scala

import scala.collection.JavaConversions._
import org.testng.annotations._
import org.testng.Assert._
import com.databricks.api.proto.mlflow.Service._
import com.databricks.mlflow.client.objects.{ObjectUtils,ParameterSearch}
import com.databricks.mlflow.client.TestShared

class ApiClientTest extends BaseTest {
  val expIdPythonScikitLearnTest = 0L
  var runId: String = _
  var run: Run = _

  @Test
  def getCreateExperimentTest() {
    val expName = createExperimentName()
    val expId = client.createExperiment(expName)
    val exp = client.getExperiment(expId)
    assertEquals(exp.getExperiment().getName(),expName)
  }

  @Test
  def listExperimentsTest() {
    val expsBefore = client.listExperiments()

    val expName = createExperimentName()
    val expId = client.createExperiment(expName)

    val exps = client.listExperiments()
    assertEquals(exps.size(), 1+expsBefore.size())

    val opt = getExperimentByName(exps,expName)
    assertFalse(opt == None)
    val expList = opt.get

    val expGet = client.getExperiment(expId).getExperiment()
    assertEquals(expGet,expList)
  }

  @Test
  def addGetRun() {
    // Create experiment
    val expName = createExperimentName()
    val expId = client.createExperiment(expName)

    // Create run 
    val user = System.getenv("USER")
    val startTime = System.currentTimeMillis()
    val sourceFile = "MyFile.java"
    val request = ObjectUtils.makeCreateRun(expId, "run_for_"+expId, SourceType.LOCAL, sourceFile, startTime, user)     
    val runCreated = client.createRun(request)
    runId = runCreated.getRunUuid()

    // Log parameters
    client.logParameter(runId, "min_samples_leaf", "2")
    client.logParameter(runId, "max_depth", "3")

    // Log metrics
    client.logMetric(runId, "auc", 2.12F)
    client.logMetric(runId, "accuracy_score", 3.12F)
    client.logMetric(runId, "zero_one_loss", 4.12F)

    // Update finished run
    client.updateRun(runId, RunStatus.FINISHED, startTime+1001)
    
    // Get run details
    run = client.getRun(runId)

    val runInfo = run.getInfo()
    assertEquals(runInfo.getExperimentId(),expId)
    assertEquals(runInfo.getUserId(),user)
    assertEquals(runInfo.getExperimentId(),expId)
    assertEquals(runInfo.getSourceName(),sourceFile)
  }

  @Test(dependsOnMethods = Array("addGetRun"))
  def checkParamsAndMetrics() {
    val params = run.getData().getParamsList()
    val metrics = run.getData().getMetricsList()
    assertEquals(params.size,2)
    assertEquals(metrics.size,3)
  }

  @Test(dependsOnMethods = Array("addGetRun"))
  def checkSearch() {
    import com.databricks.mlflow.client.TestUtils

    val rsp = client.search(Array(expIdPythonScikitLearnTest), Array(new ParameterSearch("max_depth","=","3")))
    assertEquals(rsp.getRunsList().size(),1)
    val runData = rsp.getRunsList().get(0).getData()
    TestUtils.assertParam(runData.getParamsList(),"max_depth","3")
  }

  def getExperimentByName(exps: Seq[Experiment], experimentName: String) = {
    exps find (_.getName == experimentName)
  }

}
