package org.apache.spark.sql

import org.apache.spark.sql.execution.ui._
import org.apache.spark.sql.execution.QueryExecution


/**
 * MLflow-internal object used to access Spark-private fields in the implementation of
 * autologging Spark datasource information.
 */
object SparkAutologgingUtils {
  def getQueryExecution(sqlExecution: SparkListenerSQLExecutionEnd): QueryExecution = {
    sqlExecution.qe
  }
}
