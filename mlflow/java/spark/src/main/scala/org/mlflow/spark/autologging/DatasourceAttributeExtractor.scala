package org.mlflow.spark.autologging

import scala.collection.JavaConverters._
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
private[autologging] trait DatasourceAttributeExtractorBase {
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

  protected def maybeGetDeltaTableInfo(plan: LogicalPlan): Option[SparkTableInfo]

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
//        case LogicalRelation(HadoopFsRelation(index, _, _, _, _, _), _, _, _) =>
//          val path: String = index.rootPaths.headOption.map(_.toString).getOrElse("unknown")
//          Option(SparkTableInfo(path, None, None))
        case other =>
          None
      }
    }
  }
}

object DatasourceAttributeExtractor extends DatasourceAttributeExtractorBase {
  override def maybeGetDeltaTableInfo(leafNode: LogicalPlan): Option[SparkTableInfo] = None
}


object DatabricksDatasourceAttributeExtractor extends DatasourceAttributeExtractorBase {
  override def maybeGetDeltaTableInfo(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    leafNode match {
      case lr: LogicalRelation =>
        // First, check whether LogicalRelation is a Delta table
        val obj = ReflectionUtils.getScalaObjectByName("com.databricks.sql.transaction.tahoe.DeltaTable")
        val deltaFileIndexOpt = ReflectionUtils.callMethod(obj, "unapply", Seq(lr)).asInstanceOf[Option[Any]]
        deltaFileIndexOpt.map(fileIndex => {
          val path = ReflectionUtils.getField(fileIndex, "path").toString
          val versionOpt = Option(ReflectionUtils.getField(fileIndex, "tableVersion")).map(_.toString)
          SparkTableInfo(path, versionOpt, Option("delta"))
        })
      case other => None
    }
  }

  override def getTableInfoToLog(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    val rawTableInfo = super.getTableInfoToLog(leafNode)
    // TODO apply redaction to path
    rawTableInfo.map(identity)
  }
}
