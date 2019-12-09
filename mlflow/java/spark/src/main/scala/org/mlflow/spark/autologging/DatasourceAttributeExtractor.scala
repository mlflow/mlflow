package org.mlflow.spark.autologging

import scala.collection.JavaConverters._
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.datasources.v2.{DataSourceV2Relation, FileTable}
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.connector.catalog.Table

case class SparkTableInfo(path: String, versionOpt: Option[String], formatOpt: Option[String])

/**
 * Interface for extracting Spark datasource attributes from a Spark logical plan.
 */
trait DatasourceAttributeExtractorBase {
  def maybeGetDeltaTableInfo(plan: LogicalPlan): Option[SparkTableInfo]

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

  // Get SparkTableInfo of info to log from leaf node of a query plan
  def getTableInfoToLog(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    val deltaInfoOpt = maybeGetDeltaTableInfo(leafNode)
    if (deltaInfoOpt.isDefined) {
      deltaInfoOpt
    } else {
      println(s"LeafNode isn't a delta relation, it has type ${leafNode.getClass.getName}")
      leafNode match {
        case relation: DataSourceV2Relation =>
          getSparkTableInfoFromTable(relation.table)
        // TODO: having these seems to break Spark 3.0 OSS build, but probably works in
        // Databricks?
        case LogicalRelation(HadoopFsRelation(index, _, _, _, _, _), _, _, _) =>
          val path: String = index.rootPaths.headOption.map(_.toString).getOrElse("unknown")
          Option(SparkTableInfo(path, None, None))
        case other =>
          println(s"Got LeafNode of other class type ${leafNode.getClass.getName}")
          None
      }
    }
  }
}

object DatasourceAttributeExtractor extends DatasourceAttributeExtractorBase {
  override def maybeGetDeltaTableInfo(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    leafNode match {
      case lr: LogicalRelation =>
        // First, check whether LogicalRelation is a Delta table
        val obj = ReflectionUtils.getScalaObjectByName("org.apache.spark.sql.delta.DeltaTable")
        val deltaFileIndexOpt = ReflectionUtils.callMethod(obj, "unapply", Seq(lr)).asInstanceOf[Option[Any]]
        deltaFileIndexOpt.map(fileIndex => {
          val path = ReflectionUtils.getField(fileIndex, "path").toString
          val versionOpt = Option(ReflectionUtils.getField(fileIndex, "tableVersion")).map(_.toString)
          SparkTableInfo(path, versionOpt, Option("delta"))
        })
      case other => None
    }
  }
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
