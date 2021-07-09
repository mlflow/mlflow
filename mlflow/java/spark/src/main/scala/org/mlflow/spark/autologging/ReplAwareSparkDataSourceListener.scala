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
    replId: String,
    publisher: MlflowAutologEventPublisherImpl = MlflowAutologEventPublisher)
  extends SparkDataSourceListener(publisher) {

  override protected def getDatasourceAttributeExtractor: DatasourceAttributeExtractorBase = {
    ReplAwareDatasourceAttributeExtractor
  }

  override protected def getReplIdOpt(event: SparkListenerSQLExecutionEnd): Option[String] = {
    return Some(replId);
  }
}
