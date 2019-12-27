package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, LogicalPlan}
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.apache.spark.sql.execution.QueryExecution
import org.slf4j.LoggerFactory


/**
 * Implementation of the SparkListener interface used to detect Spark datasource reads.
 * and notify subscribers.
 */
class SparkDataSourceListener(
    publisher: MlflowAutologEventPublisherImpl = MlflowAutologEventPublisher) extends SparkListener {
  private val logger = LoggerFactory.getLogger(getClass)


  // Exposed for testing
  private[autologging] def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    val tableInfos = SparkDataSourceListener.getTableInfos(event, DatasourceAttributeExtractor)
    tableInfos.foreach { tableInfo =>
      publisher.publishEvent(None, tableInfo)
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionEnd =>
        // Defensively catch exceptions while attempting to extract datasource read information
        // from the SparkListenerSQLExecutionEnd event. In particular, we do this to defend
        // against changes in the internal APIs we access (e.g. changes in Delta table classnames
        // or removal of the QueryExecution field from SparkListenerSQLExecutionEnd) in future
        // Spark versions. As of the time of writing, Spark seems to also catch these exceptions,
        // but we defensively catch here to be safe & give the user a better error message.
        ExceptionUtils.tryAndLogUnexpectedError(
          logger, "when attempting to handle SparkListenerSQLExecutionEnd event", {
          onSQLExecutionEnd(e)
        })
      case _ =>
    }
  }
}

private[autologging] object SparkDataSourceListener {
  private def getLeafNodes(lp: LogicalPlan): Seq[LogicalPlan] = {
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

  // TODO: probably this method should live in DatasourceAttributeExtractorBase
  // The SparkListeners can then just construct the right DatasourceAttributeExtractorBase subclass
  // & call this method on that subclass
  def getTableInfos(
      event: SparkListenerEvent,
      extractor: DatasourceAttributeExtractorBase): Seq[SparkTableInfo] = {
    val qe = ReflectionUtils.getField(event, "qe").asInstanceOf[QueryExecution]
    println(s"@SID got qe ${qe} from event ${event.getClass.getName}")
    if (qe != null) {
      val leafNodes = getLeafNodes(qe.analyzed)
      leafNodes.flatMap(extractor.getTableInfoToLog)
    } else {
      Seq.empty
    }
  }
}
