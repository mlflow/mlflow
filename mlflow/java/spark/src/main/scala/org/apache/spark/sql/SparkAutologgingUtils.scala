package org.apache.spark.sql

import org.apache.spark.sql.execution.ui._
import org.apache.spark.sql.execution.QueryExecution


object SparkAutologgingUtils {
  def getQueryExecution(sqlExecution: SparkListenerSQLExecutionEnd): QueryExecution = {
    println(sqlExecution.executionId)
    println(sqlExecution.executionFailure)
    sqlExecution.qe
  }
}
