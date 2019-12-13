package org.mlflow.spark.autologging

import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.datasources.v2.{DataSourceV2Relation, FileTable}
import org.apache.spark.sql.execution.datasources.{FileFormat, HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.connector.catalog.Table
import org.apache.spark.sql.execution.datasources.csv.CSVFileFormat
import org.apache.spark.sql.execution.datasources.json.JsonFileFormat
import org.apache.spark.sql.execution.datasources.orc.OrcFileFormat
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.execution.datasources.text.TextFileFormat
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
      case LogicalRelation(HadoopFsRelation(index, _, _, _, fileFormat, _), _, _, _) =>
        fileFormat match {
          case _: CSVFileFormat => None
          case _: ParquetFileFormat => None
          case _: JsonFileFormat => None
          case _: OrcFileFormat => None
          case _: TextFileFormat => None
          case other: FileFormat =>
            val path: String = index.rootPaths.headOption.map(_.toString).getOrElse("unknown")
            println(s"@SID Got fileformat ${other.getClass.getName} for path ${path}")
            Option(SparkTableInfo(path, None, None))
        }
      case other =>
        None
    }
  }
}
