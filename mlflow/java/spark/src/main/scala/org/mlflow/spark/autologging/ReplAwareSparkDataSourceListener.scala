package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, LogicalPlan}
import org.apache.spark.sql.execution.ui.{SparkListenerSQLExecutionEnd, SparkListenerSQLExecutionStart}
import org.apache.spark.sql.execution.{QueryExecution, SQLExecution}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * Implementation of the SparkListener interface used to detect Spark datasource reads.
 * and notify subscribers. Used in REPL-ID aware environments (e.g. Databricks)
 */
class ReplAwareSparkDataSourceListener(
    publisher: MlflowAutologEventPublisherImpl = MlflowAutologEventPublisher)
  extends SparkDataSourceListener(publisher) {
  private val executionIdToReplId = mutable.Map[Long, String]()

  override protected def getDatasourceAttributeExtractor: DatasourceAttributeExtractorBase = {
    ReplAwareDatasourceAttributeExtractor
  }

  private[autologging] def getProperties(event: SparkListenerJobStart): Map[String, String] = {
    Option(event.properties).map(_.asScala.toMap).getOrElse(Map.empty)
  }

  override def onJobStart(event: SparkListenerJobStart): Unit = {
    val properties = getProperties(event)
    val executionIdOpt = properties.get(SQLExecution.EXECUTION_ID_KEY).map(_.toLong)
    val replIdOpt = properties.get("spark.databricks.replId")

    (executionIdOpt, replIdOpt) match {
      case (Some(executionId), Some(replId)) =>
        executionIdToReplId.put(executionId, replId)
      case _ =>
        logger.trace(s"Skipping datasource autolog - required properties not available")
    }
  }

  protected[autologging] override def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    val extractor = getDatasourceAttributeExtractor
    val tableInfos = extractor.getTableInfos(event)
    val replIdOpt = popReplIdOpt(event)
    if (replIdOpt.isDefined) {
      tableInfos.foreach { tableInfo =>
        publisher.publishEvent(replIdOpt = replIdOpt, sparkTableInfo = tableInfo)
      }
    }
  }

  private def popReplIdOpt(event: SparkListenerSQLExecutionEnd): Option[String] = {
    executionIdToReplId.remove(event.executionId)
  }
}
