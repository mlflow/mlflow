package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.ui.{SparkListenerSQLExecutionEnd, SparkListenerSQLExecutionStart}
import org.apache.spark.sql.execution.{QueryExecution, SQLExecution}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * Implementation of the SparkListener interface used to detect Spark datasource reads.
 * and notify subscribers. Similar to SparkDataSourceListener, but attempts to route datasource-read
 * events to subscribers based on REPL ID.
 */
private[autologging] class DatabricksSparkDataSourceListener(
    publisher: MlflowAutologEventPublisherImpl,
    isSpark2: Boolean)
  extends SparkListener {
  private val logger = LoggerFactory.getLogger(getClass)
  private val executionIdToTableInfos = mutable.Map[Long, Seq[SparkTableInfo]]()

  override def onJobStart(event: SparkListenerJobStart): Unit = {
    logger.info("@SID in onJobStart")
    val properties = Option(event.properties).map(_.asScala.toMap).getOrElse(Map.empty)
    val executionIdOpt = properties.get(SQLExecution.EXECUTION_ID_KEY).map(_.toLong)
    if (executionIdOpt.isEmpty) {
      return
    }
    val executionId = executionIdOpt.get
    val replIdOpt = properties.get("spark.databricks.replId")
    logger.info(s"@SID in onJobStart, ${executionId}, ${replIdOpt}")
    replIdOpt.foreach { replId =>
      executionIdToTableInfos.get(executionId).foreach { tableInfosToLog =>
        tableInfosToLog.foreach(
          tableInfo => publisher.publishEvent(Option(replId), tableInfo)
        )
      }
    }
  }

  private def addTableInfos(executionId: Long, tableInfos: Seq[SparkTableInfo]): Unit = synchronized {
    val tableInfosOpt = executionIdToTableInfos.get(executionId)
    if (tableInfosOpt.isDefined) {
      throw new RuntimeException(
        s"Unexpected error trying to associate " +
          s"execution ID $executionId -> table infos. Found existing table infos.")
    } else {
      executionIdToTableInfos(executionId) = tableInfos
    }
  }

  // Populate a map of execution ID to list of table infos to log under
  private def onSQLExecutionStart(event: SparkListenerSQLExecutionStart): Unit = {
    val extractor = if (isSpark2) DatabricksDatasourceAttributeExtractorSpark2 else DatabricksDatasourceAttributeExtractor
    val tableInfos = SparkDataSourceListener.getTableInfos(event, extractor)
    addTableInfos(event.executionId, tableInfos)
  }


  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    logger.info(s"@SID in onOtherEvent, event ${event.getClass.getName}")
    event match {
      case e: SparkListenerSQLExecutionStart =>
        // Defensively catch exceptions while attempting to extract datasource read information
        // from the SparkListenerSQLExecutionEnd event. In particular, we do this to defend
        // against changes in the internal APIs we access (e.g. changes in Delta table classnames
        // or removal of the QueryExecution field from SparkListenerSQLExecutionEnd) in future
        // Spark versions. As of the time of writing, Spark seems to also catch these exceptions,
        // but we defensively catch here to be safe & give the user a better error message.
        ExceptionUtils.tryAndLogUnexpectedError(
          logger, "when attempting to handle SparkListenerSQLExecutionStart event", {
            onSQLExecutionStart(e)
          })
      case e: SparkListenerSQLExecutionEnd =>
        this.synchronized {
          executionIdToTableInfos.remove(e.executionId)
        }
      case _ =>
    }
  }
}
