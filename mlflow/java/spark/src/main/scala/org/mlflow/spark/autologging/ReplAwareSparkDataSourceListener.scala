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
    sparkSessionUUID: String,
    publisher: MlflowAutologEventPublisherImpl = MlflowAutologEventPublisher)
  extends SparkDataSourceListener(publisher) {

  override protected def getDatasourceAttributeExtractor: DatasourceAttributeExtractorBase = {
    ReplAwareDatasourceAttributeExtractor
  }

  override protected def getReplIdOpt(event: SparkListenerSQLExecutionEnd): Option[String] = {
    // NB: We directly return the Spark Session UUID under the assumption that a data source
    // listener can only be attached to a single Spark Session at a time and that the Spark Session
    // UUID uniquely identifies a REPL
    return Some(sparkSessionUUID);
  }
}
