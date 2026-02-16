package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.slf4j.LoggerFactory
import scala.util.control.NonFatal


/**
 * Implementation of the SparkListener interface used to detect Spark datasource reads.
 * and notify subscribers.
 */
class SparkDataSourceListener(
    publisher: MlflowAutologEventPublisherImpl = MlflowAutologEventPublisher) extends SparkListener {
  protected val logger = LoggerFactory.getLogger(getClass)

  protected def getDatasourceAttributeExtractor: DatasourceAttributeExtractorBase = {
    DatasourceAttributeExtractor
  }

  protected[autologging] def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    val extractor = getDatasourceAttributeExtractor
    val tableInfos = extractor.getTableInfos(event)
    tableInfos.foreach { tableInfo =>
      publisher.publishEvent(replIdOpt = None, sparkTableInfo = tableInfo)
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionEnd =>
        try {
          onSQLExecutionEnd(e)
        } catch {
          case NonFatal(ex) =>
            logger.trace(s"Skipping datasource autolog: ${ex.getMessage}")
        }
      case _ =>
    }
  }
}
