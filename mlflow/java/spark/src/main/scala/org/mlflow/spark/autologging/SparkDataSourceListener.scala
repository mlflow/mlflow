package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, LogicalPlan}
import org.apache.spark.sql.execution.ui.{SparkListenerSQLExecutionEnd, SparkListenerSQLExecutionStart}
import org.apache.spark.sql.{SparkAutologgingUtils, SparkSession}

class SparkDataSourceListener extends SparkListener {
  protected def getLeafNodes(lp: LogicalPlan): Seq[LogicalPlan] = {
    if (lp == null) {
      return Seq.empty
    }
    println(s"Traversing getLeafNodes, got ${lp.getClass.getName}, children: ${lp.children.length}")
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
        SparkDataSourceEventPublisher.publishEvent(None, tableInfo)
      }
    }
  }


  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionEnd =>
        onSQLExecutionEnd(e)
      case _ =>
    }
  }
}
