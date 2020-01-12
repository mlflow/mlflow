package org.mlflow.spark.autologging

import org.apache.spark.sql.SparkAutologgingUtils
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, LogicalPlan}
import org.apache.spark.sql.execution.datasources.v2.{DataSourceV2Relation, FileTable}
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.connector.catalog.Table
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.apache.spark.sql.sources.DataSourceRegister
import org.slf4j.{Logger, LoggerFactory}

import scala.util.control.NonFatal

/** Case class wrapping information on a Spark datasource that was read. */
private[autologging] case class SparkTableInfo(
    path: String,
    versionOpt: Option[String],
    formatOpt: Option[String])

/** Base trait for extracting Spark datasource attributes from a Spark logical plan. */
private[autologging] trait DatasourceAttributeExtractorBase {
  protected val logger: Logger = LoggerFactory.getLogger(getClass)

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
  protected def getTableInfoToLog(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    val deltaInfoOpt = maybeGetDeltaTableInfo(leafNode)
    if (deltaInfoOpt.isDefined) {
      deltaInfoOpt
    } else {
      leafNode match {
        case relation: DataSourceV2Relation =>
          getSparkTableInfoFromTable(relation.table)
        case other =>
          None
      }
    }
  }

  private def getLeafNodes(lp: LogicalPlan): Seq[LogicalPlan] = {
    if (lp == null) {
      return Seq.empty
    }
    if (lp.isInstanceOf[LeafNode]) {
      Seq(lp)
    } else {
      lp.children.flatMap(getLeafNodes)
    }
  }

  /**
   * Get SparkTableInfo representing the datasource(s) that were read from a SparkListenerEvent
   * assumed to have a QueryExecution field named "qe".
   */
  def getTableInfos(event: SparkListenerSQLExecutionEnd): Seq[SparkTableInfo] = {
    val qe = SparkAutologgingUtils.getQueryExecution(event)
    if (qe != null) {
      val leafNodes = getLeafNodes(qe.analyzed)
      leafNodes.flatMap(getTableInfoToLog)
    } else {
      Seq.empty
    }
  }
}

/** Default datasource attribute extractor */
object DatasourceAttributeExtractor extends DatasourceAttributeExtractorBase {
  // TODO: attempt to detect Delta table info when Delta Lake becomes compatible with Spark 3.0
  override def maybeGetDeltaTableInfo(leafNode: LogicalPlan): Option[SparkTableInfo] = None
}

/** Datasource attribute extractor for REPL-ID aware environments (e.g. Databricks) */
object ReplAwareDatasourceAttributeExtractor extends DatasourceAttributeExtractorBase {
  override protected def maybeGetDeltaTableInfo(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    leafNode match {
      case lr: LogicalRelation =>
        // First, check whether LogicalRelation is a Delta table
        val obj = ReflectionUtils.getScalaObjectByName("com.databricks.sql.transaction.tahoe.DeltaTable")
        val deltaFileIndexOpt = ReflectionUtils.callMethod(obj, "unapply", Seq(lr)).asInstanceOf[Option[Any]]
        deltaFileIndexOpt.map(fileIndex => {
          val path = ReflectionUtils.getField(fileIndex, "path").toString
          val versionOpt = Option(ReflectionUtils.callMethod(fileIndex, "tableVersion", Seq.empty)).map(_.toString)
          SparkTableInfo(path, versionOpt, Option("delta"))
        })
      case other => None
    }
  }

  override def getTableInfoToLog(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    val res = super.getTableInfoToLog(leafNode)
    if (res.isDefined) {
      res
    } else {
      leafNode match {
        case LogicalRelation(HadoopFsRelation(index, _, _, _, fileFormat, _), _, _, _) =>
          val path: String = index.rootPaths.headOption.map(_.toString).getOrElse("unknown")
          val formatOpt = fileFormat match {
            case format: DataSourceRegister => Option(format.shortName)
            case _ => None
          }
          Option(SparkTableInfo(path, None, formatOpt))
        case _ => None
      }
    }
  }

  // Attempts to apply redaction to sensitive data within datasource paths (e.g. S3 keys)
  private def tryRedactString(value: String): String = {
    try {
      val redactor = ReflectionUtils.getScalaObjectByName(
        "com.databricks.spark.util.DatabricksSparkLogRedactor")
      ReflectionUtils.callMethod(redactor, "redact", Seq(value)).asInstanceOf[String]
    } catch {
      case NonFatal(e) =>
        val msg = ExceptionUtils.getUnexpectedExceptionMessage(e, "while applying redaction to " +
          "datasource paths")
        logger.error(msg)
        throw e
    }
  }

  private def applyRedaction(tableInfo: SparkTableInfo): SparkTableInfo = {
    tableInfo match {
      case SparkTableInfo(path, versionOpt, formatOpt) =>
        SparkTableInfo(tryRedactString(path), versionOpt, formatOpt)
    }
  }

  override def getTableInfos(event: SparkListenerSQLExecutionEnd): Seq[SparkTableInfo] = {
    super.getTableInfos(event).map(applyRedaction)
  }
}
