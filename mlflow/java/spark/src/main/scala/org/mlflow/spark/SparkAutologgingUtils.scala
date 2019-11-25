package org.apache.spark.sql

import org.apache.spark.sql.execution.QueryExecution
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd

object SparkAutologgingUtils {
  def getQueryExecution(sqlExecution: SparkListenerSQLExecutionEnd): QueryExecution = {
    sqlExecution.qe
  }
}
