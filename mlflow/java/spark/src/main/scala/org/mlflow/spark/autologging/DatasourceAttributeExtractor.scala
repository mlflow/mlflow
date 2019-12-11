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
    val deltaInfoOpt = maybeGetDeltaTableInfo(leafNode)
    if (deltaInfoOpt.isDefined) {
      deltaInfoOpt
    } else {
      leafNode match {
        case relation: DataSourceV2Relation =>
          getSparkTableInfoFromTable(relation.table)
        case LogicalRelation(HadoopFsRelation(index, _, _, _, _, _), _, _, _) =>
          val path: String = index.rootPaths.headOption.map(_.toString).getOrElse("unknown")
          Option(SparkTableInfo(path, None, None))
        case other =>
          None
      }
    }
  }

  /**
   * Return an option containing a SparkTableInfo representing a Delta table read if the passed-in
   * query plan leafNode corresponds to a read of a Delta table. If the leafNode does not correspond
   * to a Delta table read, returns None.
   */
  def maybeGetDeltaTableInfo(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    leafNode match {
      case lr: LogicalRelation =>
        // First, check whether LogicalRelation is a Delta table
        // We use reflection so that we do not need to depend on the Delta package in our JAR
        try {
          val obj = ReflectionUtils.getScalaObjectByName("org.apache.spark.sql.delta.DeltaTable")
          val deltaFileIndexOpt = ReflectionUtils.callMethod(obj, "unapply", Seq(lr)).asInstanceOf[Option[Any]]
          deltaFileIndexOpt.map(fileIndex => {
            val path = ReflectionUtils.getField(fileIndex, "path").toString
            val versionOpt = Option(ReflectionUtils.getField(fileIndex, "tableVersion")).map(_.toString)
            SparkTableInfo(path, versionOpt, Option("delta"))
          })
        } catch {
          case _: ScalaReflectionException =>
            None
          case NonFatal(e) =>
            logger.error(s"Unexpected exception when attempting to extract Delta table info from" +
              s"Spark SQL query plan. Please report this error, along with the " +
              s"following stacktrace, on https://github.com/mlflow/mlflow/issues:\n" +
              s"${ExceptionUtils.serializeException(e)}")
            None
        }
      case other => None
    }
  }
}
