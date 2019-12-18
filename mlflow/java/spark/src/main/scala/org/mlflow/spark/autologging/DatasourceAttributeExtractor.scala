package org.mlflow.spark.autologging

import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.datasources.v2.{DataSourceV2Relation, FileTable}
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.connector.catalog.Table
import org.slf4j.LoggerFactory

import scala.util.control.NonFatal

/** Case class wrapping information on a Spark datasource that was read. */
private[autologging] case class SparkTableInfo(
    path: String,
    versionOpt: Option[String],
    formatOpt: Option[String])

/** Helper object for extracting Spark datasource attributes from a Spark logical plan. */
private[autologging] object DatasourceAttributeExtractor {
  private val logger = LoggerFactory.getLogger(getClass)

  private def getSparkTableInfoFromTable(table: Table): Option[SparkTableInfo] = {
    table match {
      case fileTable: FileTable =>
        val tableName = fileTable.name
        val splitName = tableName.split(" ")
        val lowercaseFormat = fileTable.formatName.toLowerCase()
        if (splitName.headOption.exists(head => head.toLowerCase == lowercaseFormat)) {
          Option(SparkTableInfo(splitName.tail.mkString(" "), None, Option(lowercaseFormat)))
        } else {
          Option(SparkTableInfo(fileTable.name, None, Option(fileTable.formatName)))
        }
      case other: Table =>
        Option(SparkTableInfo(other.name, None, None))
    }
  }

  /**
   * Get SparkTableInfo representing the datasource that was read from leaf node of a Spark SQL
   * query plan
   */
  def getTableInfoToLog(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    leafNode match {
      case relation: DataSourceV2Relation =>
        getSparkTableInfoFromTable(relation.table)
      case other =>
        None
    }
  }
}
