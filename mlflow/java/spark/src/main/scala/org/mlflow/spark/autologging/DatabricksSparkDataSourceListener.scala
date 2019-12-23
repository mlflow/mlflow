package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, LogicalPlan}
import org.apache.spark.sql.execution.ui.{SparkListenerSQLExecutionEnd, SparkListenerSQLExecutionStart}
import org.apache.spark.sql.execution.{QueryExecution, SQLExecution}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * Implementation of the SparkListener interface used to detect Spark datasource reads.
 * and notify subscribers.
 */
class DatabricksSparkDataSourceListener(
    publisher: MlflowAutologEventPublisherImpl = MlflowAutologEventPublisher)
  extends SparkDataSourceListener(publisher) {
  private val executionIdToReplId = mutable.Map[Long, String]()


  private[autologging] def getProperties(event: SparkListenerJobStart): Map[String, String] = {
    Option(event.properties).map(_.asScala.toMap).getOrElse(Map.empty)
  }

  override def onJobStart(event: SparkListenerJobStart): Unit = {
    val properties = getProperties(event)
    val executionIdOpt = properties.get(SQLExecution.EXECUTION_ID_KEY).map(_.toLong)
    if (executionIdOpt.isEmpty) {
      return
    }
    val executionId = executionIdOpt.get
    val replIdOpt = properties.get("spark.databricks.replId")
    replIdOpt.foreach { replId =>
      executionIdToReplId.put(executionId, replId)
    }
  }

  override protected def getReplIdOpt(event: SparkListenerSQLExecutionEnd): Option[String] = {
    executionIdToReplId.remove(event.executionId)
  }
}
