package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, LogicalPlan}
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.apache.spark.sql.SparkAutologgingUtils
import org.slf4j.LoggerFactory

import scala.util.control.NonFatal

/**
 * Implementation of the SparkListener interface used to detect Spark datasource reads.
 * and notify subscribers.
 */
class SparkDataSourceListener(
    publisher: MlflowAutologEventPublisherImpl = MlflowAutologEventPublisher) extends SparkListener {
  private val logger = LoggerFactory.getLogger(getClass)

  protected def getLeafNodes(lp: LogicalPlan): Seq[LogicalPlan] = {
    if (lp == null) {
      return Seq.empty
    }
    if (lp.isInstanceOf[LeafNode]) {
      Seq(lp)
    } else {
      lp.children.flatMap { child =>
        child match {
          case l: LeafNode =>
            Seq(l)
          case other: LogicalPlan => getLeafNodes(other)
        }
      }
    }
  }

  protected def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    val qe = SparkAutologgingUtils.getQueryExecution(event)
    if (qe != null) {
      val leafNodes = getLeafNodes(qe.analyzed)
      val tableInfosToLog = leafNodes.flatMap(DatasourceAttributeExtractor.getTableInfoToLog)
      tableInfosToLog.foreach { tableInfo =>
        publisher.publishEvent(None, tableInfo)
      }
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionEnd =>
        try {
          onSQLExecutionEnd(e)
        } catch {
          // Defensively catch exceptions while attempting to extract datasource read information
          // from the SparkListenerSQLExecutionEnd event. In particular, we do this to defend
          // against changes in the internal APIs we access (e.g. changes in Delta table classnames
          // or removal of the QueryExecution field from SparkListenerSQLExecutionEnd) in future
          // Spark versions. As of the time of writing, Spark seems to also catch these exceptions,
          // but we defensively catch here to be safe & give the user a better error message.
          case NonFatal(exc) =>
            logger.error(s"Unexpected exception when attempting to handle " +
              s"SparkListenerSQLExecutionEnd event. Please report this error, along with the " +
              s"following stacktrace, on https://github.com/mlflow/mlflow/issues:\n" +
              s"${ExceptionUtils.serializeException(exc)}")
        }
      case _ =>
    }
  }
}
