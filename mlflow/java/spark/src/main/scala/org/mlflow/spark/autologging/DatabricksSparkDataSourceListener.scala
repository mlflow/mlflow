package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, LogicalPlan}
import org.apache.spark.sql.execution.ui.{SparkListenerSQLExecutionEnd, SparkListenerSQLExecutionStart}
import org.apache.spark.sql.execution.{QueryExecution, SQLExecution}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * SparkListener implementation that attempts to extract Delta table information & notify the PySpark process
  * TODO: maybe pull query-plan-parsing logic out so that we can later add autologging for users of the Java client
  * as well
  */
class DatabricksSparkDataSourceListener() extends SparkDataSourceListener {

  // Order of callbacks is onSQLExecutionStart, onJobStart, onSQLExecutionEnd
  // So we can get the table infos to log in onSQLExecutionStart, figure out where to log them in onJobStart,
  // and remove them in onSQLExecutionEnd

  // A QueryExecution has many JobIDs via associatedJobs = mutable.Set[Int]()
  // A SparkListenerJobStart event gives you a mapping from JobIds to repl IDs (so you know the repl associated with each job)
  // So should be able to capture the first mapping on Job start, then on query execution end look up
  // the repls associated with each Spark job that was completed, and notify those repls
  private val executionIdToTableInfos = mutable.Map[Long, Seq[SparkTableInfo]]()

  private def addTableInfos(executionId: Long, tableInfos: Seq[SparkTableInfo]): Unit = synchronized {
    val tableInfosOpt = executionIdToTableInfos.get(executionId)
    if (tableInfosOpt.isDefined) {
      throw new RuntimeException(
        s"Unexpected error trying to associate " +
          s"execution ID ${executionId} -> table infos. Found existing table infos.")
    } else {
      executionIdToTableInfos(executionId) = tableInfos
    }
  }

  private def removeExecutionIdToTableInfos(executionId: Long): Unit = synchronized {
    executionIdToTableInfos.remove(executionId)
  }

  override def onJobStart(event: SparkListenerJobStart): Unit = {
    // Find corresponding execution, and
    // If found, check if we have an associated active SQLExecutionAdvisor, and set its job group.
    // This is needed, as in onSQLExecutionStart we don't know the SQLExecution's job group.
    val properties: Map[String, String] = Option(event.properties).map(_.asScala.toMap).getOrElse(Map.empty)
    val executionIdOpt = properties.get(SQLExecution.EXECUTION_ID_KEY).map(_.toLong)
    if (executionIdOpt.isEmpty) {
      return
    }
    val executionId = executionIdOpt.get
    val replIdOpt = properties.get("spark.databricks.replId")
    replIdOpt.foreach { replId =>
      executionIdToTableInfos.get(executionId).map { tableInfosToLog =>
        tableInfosToLog.map(tableInfo => SparkDataSourceEventPublisher.publishEvent(Option(replId), tableInfo))
      }
    }
  }

  // Populate a map of execution ID to list of table infos to log under
  private def onSQLExecutionStart(event: SparkListenerSQLExecutionStart): Unit = {
    val qe: QueryExecution = event.getClass.getDeclaredFields.find(_.getName == "qe")
        .map(_.get(event).asInstanceOf[QueryExecution]).getOrElse {
      throw new RuntimeException("Unable to get QueryExecution field")
    }

    if (qe != null) {
      val leafNodes = getLeafNodes(qe.analyzed)
      val tableInfosToLog = leafNodes.flatMap(
        DatabricksDatasourceAttributeExtractor.getTableInfoToLog)
      addTableInfos(event.executionId, tableInfosToLog)
    }
  }

  override protected def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    removeExecutionIdToTableInfos(event.executionId)
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionStart =>
        onSQLExecutionStart(e)
      case e: SparkListenerSQLExecutionEnd =>
        onSQLExecutionEnd(e)
      case _ =>
    }
  }
}