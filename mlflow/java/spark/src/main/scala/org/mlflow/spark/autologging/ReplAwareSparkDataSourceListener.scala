package org.mlflow.spark.autologging

import org.apache.spark.scheduler._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.apache.spark.sql.execution.QueryExecution

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * Implementation of the SparkListener interface used to detect Spark datasource reads.
 * and notify subscribers. Used in REPL-ID aware environments (e.g. Databricks)
 */
class ReplAwareSparkDataSourceListener(
    publisher: MlflowAutologEventPublisherImpl = MlflowAutologEventPublisher)
  extends SparkDataSourceListener(publisher) {

  override protected def getDatasourceAttributeExtractor: DatasourceAttributeExtractorBase = {
    ReplAwareDatasourceAttributeExtractor
  }

  override protected def getReplIdOpt(event: SparkListenerSQLExecutionEnd): Option[String] = {
    // NB: We compute and return the Spark Session UUID under the assumption that a data source
    // listener can only be attached to a single Spark Session at a time and that the Spark Session
    // UUID uniquely identifies a REPL
    val sessionUUID = SparkUtils.getSparkSessionUUID(SparkSession.getActiveSession.get)
    Some(sessionUUID)
  }
}
