package com.databricks.mlflow.client.samples

import  com.databricks.mlflow.client.ApiClient
import  com.databricks.mlflow.client.objects._
import scala.collection.JavaConversions._

object ScalaDriver {
  def main(args: Array[String]) {
    val client = new ApiClient(args(0))
    val exps = client.listExperiments()
    println("Experiments")
    for (exp <- exps) println(s"  $exp")
  }
}
