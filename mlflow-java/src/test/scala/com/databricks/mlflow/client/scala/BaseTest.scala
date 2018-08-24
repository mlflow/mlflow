package com.databricks.mlflow.client.scala

import org.testng.Assert._
import org.testng.annotations._
import com.databricks.api.proto.mlflow.Service._
import com.databricks.mlflow.client.ApiClient
import scala.util.Properties

class BaseTest() {
  val apiUriDefault = "http://localhost:5001"
  var client: ApiClient = null 

  @BeforeClass
  def BeforeClass() {
      val apiUri = Properties.envOrElse("MLFLOW_TRACKING_URI",apiUriDefault)
      client = new ApiClient(apiUri)
  }

  def createExperimentName() = "TestScala_"+System.currentTimeMillis.toString
}
